import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.occupancy_map.abstract_occupancy_map import OccupancyMap
from nuplan.planning.simulation.occupancy_map.strtree_occupancy_map import STRTreeOccupancyMapFactory
from nuplan.planning.simulation.observation.idm.idm_policy import IDMPolicy

from sledge.simulation.observation.sledge_idm.sledge_idm_agent_manager import UniqueIDMAgents
from sledge.simulation.observation.sledge_idm.sledge_idm_agent import (
    SledgeIDMAgent,
    SledgeIDMInitialState,
    SledgePedestrianAgent,
    SledgeObjectAgent,
)

logger = logging.getLogger(__name__)


def get_starting_segment(
    agent: Agent, map_api: AbstractMap
) -> Tuple[Optional[LaneGraphEdgeMapObject], Optional[float]]:
    """
    Gets the map object that the agent is on and the progress along the segment.
    :param agent: The agent of interested.
    :param map_api: An AbstractMap instance.
    :return: GraphEdgeMapObject and progress along the segment. If no map object is found then None.
    """
    if map_api.is_in_layer(agent.center, SemanticMapLayer.LANE):
        layer = SemanticMapLayer.LANE
    elif map_api.is_in_layer(agent.center, SemanticMapLayer.INTERSECTION):
        layer = SemanticMapLayer.LANE_CONNECTOR
    else:
        return None, None

    segments: List[LaneGraphEdgeMapObject] = map_api.get_all_map_objects(agent.center, layer)
    if not segments:
        return None, None

    # Get segment with the closest heading to the agent
    heading_diff = [
        segment.baseline_path.get_nearest_pose_from_position(agent.center).heading - agent.center.heading
        for segment in segments
    ]
    closest_segment = segments[np.argmin(np.abs(heading_diff))]

    progress = closest_segment.baseline_path.get_nearest_arc_length_from_position(agent.center)
    return closest_segment, progress


def build_idm_agents_on_map_rails(
    target_velocity: float,
    min_gap_to_lead_agent: float,
    headway_time: float,
    accel_max: float,
    decel_max: float,
    minimum_path_length: float,
    scenario: AbstractScenario,
    static_detections_types: List[TrackedObjectType],
) -> Tuple[UniqueIDMAgents, OccupancyMap]:
    """
    Build unique agents from a scenario. InterpolatedPaths are created for each agent according to their driven path
    TODO: Refactor
    :param target_velocity: Desired velocity in free traffic [m/s]
    :param min_gap_to_lead_agent: Minimum relative distance to lead vehicle [m]
    :param headway_time: Desired time headway. The minimum possible time to the vehicle in front [s]
    :param accel_max: maximum acceleration [m/s^2]
    :param decel_max: maximum deceleration (positive value) [m/s^2]
    :param minimum_path_length: [m] The minimum path length
    :param scenario: scenario
    :param static_detections_types: The static objects to include during simulation
    :return: a tuple of unique vehicles to simulation and a scene occupancy map
    """
    unique_vehicles: Dict[str, SledgeIDMAgent] = {}
    unique_pedestrians: Dict[str, SledgePedestrianAgent] = {}
    unique_static_objects: Dict[str, SledgeObjectAgent] = {}

    detections = scenario.initial_tracked_objects

    map_api = scenario.map_api
    ego_agent = scenario.get_ego_state_at_iteration(0).agent

    static_detections = detections.tracked_objects.get_tracked_objects_of_types(static_detections_types)
    occupancy_map = STRTreeOccupancyMapFactory.get_from_boxes(static_detections)
    occupancy_map.insert(ego_agent.token, ego_agent.box.geometry)
    for static_detection in static_detections:
        unique_static_objects[static_detection.track_token] = SledgeObjectAgent(static_detection)

    for pedestrian in detections.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.PEDESTRIAN):
        occupancy_map.insert(pedestrian.track_token, pedestrian.box.geometry)
        unique_pedestrians[pedestrian.track_token] = SledgePedestrianAgent(pedestrian)

    agent: Agent
    for agent in detections.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE):
        # filter for only vehicles
        if agent.track_token not in unique_vehicles:

            route, progress = get_starting_segment(agent, map_api)

            # Ignore agents that a baseline path cannot be built for
            if route is None:
                continue

            # Snap agent to baseline path
            state_on_path = route.baseline_path.get_nearest_pose_from_position(agent.center.point)
            box_on_baseline = OrientedBox.from_new_pose(
                agent.box, StateSE2(state_on_path.x, state_on_path.y, state_on_path.heading)
            )

            # Check for collision
            if not occupancy_map.intersects(box_on_baseline.geometry).is_empty():
                continue

            # Add to init_agent_occupancy for collision checking
            occupancy_map.insert(agent.track_token, box_on_baseline.geometry)

            # Project velocity into local frame
            if np.isnan(agent.velocity.array).any():
                ego_state = scenario.get_ego_state_at_iteration(0)
                logger.debug(
                    f"Agents has nan velocity. Setting velocity to ego's velocity of "
                    f"{ego_state.dynamic_car_state.speed}"
                )
                velocity = StateVector2D(ego_state.dynamic_car_state.speed, 0.0)
            else:
                velocity = StateVector2D(np.hypot(agent.velocity.x, agent.velocity.y), 0)

            initial_state = SledgeIDMInitialState(
                metadata=agent.metadata,
                tracked_object_type=agent.tracked_object_type,
                box=box_on_baseline,
                velocity=velocity,
                path_progress=progress,
                predictions=agent.predictions,
            )
            target_velocity = route.speed_limit_mps or target_velocity
            unique_vehicles[agent.track_token] = SledgeIDMAgent(
                start_iteration=0,
                initial_state=initial_state,
                route=[route],
                policy=IDMPolicy(target_velocity, min_gap_to_lead_agent, headway_time, accel_max, decel_max),
                minimum_path_length=minimum_path_length,
            )

    occupancy_map.remove([ego_agent.token])

    return unique_vehicles, unique_pedestrians, unique_static_objects, occupancy_map
