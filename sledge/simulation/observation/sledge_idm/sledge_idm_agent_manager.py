from typing import Dict, List

import numpy as np
from shapely.geometry.base import CAP_STYLE

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.geometry.transform import rotate_angle
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import StopLine
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.planning.simulation.observation.idm.idm_states import IDMLeadAgentState
from nuplan.planning.metrics.utils.expert_comparisons import principal_value
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.occupancy_map.abstract_occupancy_map import OccupancyMap
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring

from sledge.simulation.observation.sledge_idm.sledge_idm_agent import (
    SledgeIDMAgent,
    SledgePedestrianAgent,
    SledgeObjectAgent,
)

UniqueIDMAgents = Dict[str, SledgeIDMAgent]
UniquePedestrians = Dict[str, SledgePedestrianAgent]


class SledgeIDMAgentManager:

    def __init__(
        self,
        vehicles: Dict[str, SledgeIDMAgent],
        pedestrians: Dict[str, SledgePedestrianAgent],
        static_objects: Dict[str, StaticObject],
        occupancy: OccupancyMap,
        map_api: AbstractMap,
        radius: float,
    ):
        """
        Constructor for SledgeIDMAgentManager.
        :param vehicles: dictionary of vehicle identifier and IDM states
        :param pedestrians: dictionary of pedestrian identifier and states.
        :param static_objects: dictionary of static object identifier and states.
        :param occupancy: occupancy map of current objects in scene (for IDM).
        :param map_api: map interface of simulation
        :param radius: distance around ego to propagate agents in meter.
        """
        self.vehicles: Dict[str, SledgeIDMAgent] = vehicles
        self.pedestrians: Dict[str, SledgePedestrianAgent] = pedestrians
        self.static_objects: Dict[str, SledgeObjectAgent] = static_objects

        self.occupancy = occupancy
        self._map_api = map_api
        self._radius = radius

    def propagate_agents(
        self,
        ego_state: EgoState,
        tspan: float,
        traffic_light_status: Dict[TrafficLightStatusType, List[str]],
    ) -> None:
        """
        Propagates agents in agent manager.
        :param ego_state: object of ego vehicle state in nuPlan
        :param tspan: time interval to propagate agents in seconds
        :param traffic_light_status: dictionary of traffic light types and lane ids.
        """

        self.occupancy.set("ego", ego_state.car_footprint.geometry)
        relative_distance = None
        for agent_token, agent in self.vehicles.items():
            if agent.is_active(ego_state, self._radius) and agent.has_valid_path():
                agent.plan_route(traffic_light_status)

                # Add stop lines into occupancy map if they are impacting the agent
                stop_lines = self._get_relevant_stop_lines(agent, traffic_light_status)
                # Keep track of the stop lines that were inserted. This is to remove them for each agent
                inactive_stop_line_tokens = self._insert_stop_lines_into_occupancy_map(stop_lines)

                # Check for agents that intersects THIS agent's path
                agent_path_buffer = path_to_linestring(agent.get_path_to_go()).buffer(
                    (agent.width / 2),
                    cap_style=CAP_STYLE.flat,
                )
                intersecting_agents = self.occupancy.intersects(agent_path_buffer)

                assert intersecting_agents.contains(agent_token), "Agent's baseline does not intersect the agent itself"

                # Checking if there are agents intersecting THIS agent's baseline.
                # Hence, we are checking for at least 2 intersecting agents.
                if intersecting_agents.size > 1:

                    nearest_id, nearest_agent_polygon, relative_distance = intersecting_agents.get_nearest_entry_to(
                        agent_token
                    )
                    agent_heading = agent.to_se2().heading

                    if "ego" in nearest_id:
                        ego_velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d
                        longitudinal_velocity = np.hypot(ego_velocity.x, ego_velocity.y)
                        relative_heading = ego_state.rear_axle.heading - agent_heading
                    elif "stop_line" in nearest_id:
                        longitudinal_velocity = 0.0
                        relative_heading = 0.0
                    elif nearest_id in self.vehicles:
                        nearest_agent = self.vehicles[nearest_id]
                        longitudinal_velocity = nearest_agent.velocity
                        relative_heading = nearest_agent.to_se2().heading - agent_heading
                    else:
                        longitudinal_velocity = 0.0
                        relative_heading = 0.0

                    # Wrap angle to [-pi, pi]
                    relative_heading = principal_value(relative_heading)
                    # take the longitudinal component of the projected velocity
                    projected_velocity = rotate_angle(StateSE2(longitudinal_velocity, 0, 0), relative_heading).x

                    # relative_distance already takes the vehicle dimension into account.
                    # Therefore there is no need to pass in the length_rear.
                    length_rear = 0
                else:
                    # Free road case: no leading vehicle
                    projected_velocity = 0.0
                    relative_distance = agent.get_progress_to_go()
                    length_rear = agent.length / 2

                if relative_distance is None:
                    relative_distance = 1.0
                agent.propagate(
                    IDMLeadAgentState(
                        progress=relative_distance,
                        velocity=projected_velocity,
                        length_rear=length_rear,
                    ),
                    tspan,
                )
                self.occupancy.set(agent_token, agent.projected_footprint)
                self.occupancy.remove(inactive_stop_line_tokens)

        for pedestrian_token, pedestrian in self.pedestrians.items():
            if pedestrian.is_active(ego_state, 15):
                pedestrian.propagate(tspan)
                self.occupancy.set(pedestrian_token, pedestrian.get_pedestrian().box.geometry)

    def get_active_agents(self, ego_state: EgoState) -> DetectionsTracks:
        """
        Collect all active agents and return in detection dataclass.
        :param ego_state: object of ego vehicle state in nuPlan
        :return: detected objects in current scene.
        """
        vehicles = [
            vehicle.get_vehicle() for vehicle in self.vehicles.values() if vehicle.is_active(ego_state, self._radius)
        ]

        pedestrians = [
            pedestrian.get_pedestrian()
            for pedestrian in self.pedestrians.values()
            if pedestrian.is_active(ego_state, self._radius)
        ]

        static_objects = [
            static_object.get_object()
            for static_object in self.static_objects.values()
            if static_object.is_active(ego_state, self._radius)
        ]

        return DetectionsTracks(TrackedObjects(vehicles + pedestrians + static_objects))

    def _get_relevant_stop_lines(
        self, agent: SledgeIDMAgent, traffic_light_status: Dict[TrafficLightStatusType, List[str]]
    ) -> List[StopLine]:
        """
        Retrieve the stop lines that are affecting the given agent.
        :param agent: The vehicle IDM agent of interest.
        :param traffic_light_status: {traffic_light_status: lane_connector_ids} A dictionary containing traffic light information.
        :return: A list of stop lines associated with the given traffic light status.
        """
        relevant_lane_connectors = list(
            {segment.id for segment in agent.get_route()} & set(traffic_light_status[TrafficLightStatusType.RED])
        )
        lane_connectors = [
            self._map_api.get_map_object(lc_id, SemanticMapLayer.LANE_CONNECTOR) for lc_id in relevant_lane_connectors
        ]
        return [stop_line for lc in lane_connectors if lc for stop_line in lc.stop_lines]

    def _insert_stop_lines_into_occupancy_map(self, stop_lines: List[StopLine]) -> List[str]:
        """
        Insert stop lines into the occupancy map.
        :param stop_lines: A list of stop lines to be inserted.
        :return: A list of token corresponding to the inserted stop lines.
        """
        stop_line_tokens: List[str] = []
        for stop_line in stop_lines:
            stop_line_token = f"stop_line_{stop_line.id}"
            if not self.occupancy.contains(stop_line_token):
                self.occupancy.set(stop_line_token, stop_line.polygon)
                stop_line_tokens.append(stop_line_token)

        return stop_line_tokens
