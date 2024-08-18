from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import CAP_STYLE
from shapely.ops import unary_union

from nuplan.common.actor_state.agent import Agent, PredictedTrajectory
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import ProgressStateSE2, StateSE2, StateVector2D
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.planning.simulation.path.interpolated_path import InterpolatedPath
from nuplan.planning.simulation.path.utils import trim_path, trim_path_up_to_progress
from nuplan.planning.simulation.observation.idm.idm_policy import IDMPolicy
from nuplan.planning.simulation.observation.idm.idm_states import IDMAgentState, IDMLeadAgentState
from nuplan.planning.simulation.observation.idm.utils import create_path_from_se2, path_to_linestring

from sledge.simulation.scenarios.sledge_scenario.sledge_scenario_utils import project_state_se2


def pose_distance(pose_a: StateSE2, pose_b: StateSE2) -> float:
    """
    Helper function to calculate distance between two poses.
    :return: distance
    """
    return np.linalg.norm(pose_a.array - pose_b.array)


@dataclass(frozen=True)
class SledgeIDMInitialState:
    """Initial state of SledgeIDMAgent."""

    metadata: SceneObjectMetadata
    tracked_object_type: TrackedObjectType
    box: OrientedBox
    velocity: StateVector2D
    path_progress: float
    predictions: Optional[List[PredictedTrajectory]]


class SledgeIDMAgent:
    """Vehicle state class for IMD agents in sledge."""

    def __init__(
        self,
        start_iteration: int,
        initial_state: SledgeIDMInitialState,
        route: List[LaneGraphEdgeMapObject],
        policy: IDMPolicy,
        minimum_path_length: float,
        max_route_len: int = 20,
    ):
        """
        Constructor for SledgeIDMAgent.
        :param start_iteration: scenario iteration where agent first appeared
        :param initial_state: agent initial state
        :param route: agent initial route plan
        :param policy: policy controlling the agent behavior
        :param minimum_path_length: [m] The minimum path length
        :param max_route_len: The max number of route elements to store
        """
        self._start_iteration = start_iteration  # scenario iteration where agent first appears
        self._initial_state = initial_state
        self._state = IDMAgentState(initial_state.path_progress, initial_state.velocity.x)
        self._route: Deque[LaneGraphEdgeMapObject] = deque(route, maxlen=max_route_len)
        self._path = self._convert_route_to_path()
        self._policy = policy
        self._minimum_path_length = minimum_path_length
        self._size = (initial_state.box.width, initial_state.box.length, initial_state.box.height)

        # This variable is used to trigger when the _full_agent_state needs to be recalculated
        self._requires_state_update: bool = True
        self._full_agent_state: Optional[Agent] = None

    def propagate(self, lead_agent: IDMLeadAgentState, tspan: float) -> None:
        """
        Propagate agent forward according to the IDM policy.
        :param lead_agent: the agent leading this agent
        :param tspan: the interval of time to propagate for
        """
        speed_limit = self.end_segment.speed_limit_mps
        if speed_limit is not None and speed_limit > 0.0:
            self._policy.target_velocity = speed_limit

        solution = self._policy.solve_forward_euler_idm_policy(
            IDMAgentState(0, self._state.velocity), lead_agent, tspan
        )
        self._state.progress += solution.progress
        self._state.velocity = max(solution.velocity, 0)

        # A caching flag to trigger re-computation of self.agent
        self._requires_state_update = True

    @property
    def agent(self) -> Agent:
        """:return: the agent as a Agent object"""
        return self._get_agent_at_progress(self._get_bounded_progress())

    @property
    def polygon(self) -> Polygon:
        """:return: the agent as a Agent object"""
        return self.agent.box.geometry

    def get_route(self) -> List[LaneGraphEdgeMapObject]:
        """:return: The route the IDM agent is following."""
        return list(self._route)

    @property
    def projected_footprint(self) -> Polygon:
        """
        Returns the agent's projected footprint along it's planned path. The extended length is proportional
        to it's current velocity
        :return: The agent's projected footprint as a Polygon.
        """
        start_progress = self._clamp_progress(self.progress - self.length / 2)
        end_progress = self._clamp_progress(self.progress + self.length / 2 + self.velocity * self._policy.headway_time)
        projected_path = path_to_linestring(trim_path(self._path, start_progress, end_progress))
        return unary_union([projected_path.buffer((self.width / 2), cap_style=CAP_STYLE.flat), self.polygon])

    @property
    def width(self) -> float:
        """:return: [m] agent's width"""
        return float(self._initial_state.box.width)

    @property
    def length(self) -> float:
        """:return: [m] agent's length"""
        return float(self._initial_state.box.length)

    @property
    def progress(self) -> float:
        """:return: [m] agent's progress"""
        return self._state.progress  # type: ignore

    @property
    def velocity(self) -> float:
        """:return: [m/s] agent's velocity along the path"""
        return self._state.velocity  # type: ignore

    @property
    def end_segment(self) -> LaneGraphEdgeMapObject:
        """
        Returns the last segment in the agent's route
        :return: End segment as a LaneGraphEdgeMapObject
        """
        return self._route[-1]

    def to_se2(self) -> StateSE2:
        """
        :return: the agent as a StateSE2 object
        """
        return self._get_agent_at_progress(self._get_bounded_progress()).box.center

    def is_active(self, ego_state: EgoState, radius: float) -> bool:
        """
        Return if the agent should be active at a simulation iteration
        :param iteration: the current simulation iteration
        :param radius: radius around ego to simulate
        :return: true if active, false otherwise
        """
        return pose_distance(self.to_se2(), ego_state.center) <= radius

    def has_valid_path(self) -> bool:
        """
        :return: true if agent has a valid path, false otherwise
        """
        return self._path is not None

    def _get_bounded_progress(self) -> float:
        """
        :return: [m] The agent's progress. The progress is clamped between the start and end progress of it's path
        """
        return self._clamp_progress(self._state.progress)

    def get_path_to_go(self) -> List[ProgressStateSE2]:
        """
        :return: The agent's path trimmed to start at the agent's current progress
        """
        return trim_path_up_to_progress(self._path, self._get_bounded_progress())  # type: ignore

    def get_progress_to_go(self) -> float:
        """
        return: [m] the progress left until the end of the path
        """
        return self._path.get_end_progress() - self.progress  # type: ignore

    def get_vehicle(self) -> Agent:
        """
        Samples the the agent's trajectory. The velocity is assumed to be constant over the sampled trajectory
        :param num_samples: number of elements to sample.
        :param sampling_time: [s] time interval of sequence to sample from.
        :return: the agent's trajectory as a list of Agent
        """
        return self._get_agent_at_progress(self._get_bounded_progress())

    def plan_route(self, traffic_light_status: Dict[TrafficLightStatusType, List[str]]) -> None:
        """
        The planning logic for the agent.
            - Prefers going straight. Selects edge with the lowest curvature.
            - Looks to add a segment to the route if:
                - the progress to go is less than the agent's desired velocity multiplied by the desired headway time
                  plus the minimum path length
                - the outgoing segment is active

        :param traffic_light_status: {traffic_light_status: lane_connector_ids} A dictionary containing traffic light information
        """

        max_number_expand = 10
        number_expand = 0
        while (
            self.get_progress_to_go()
            < self._minimum_path_length + self._policy.target_velocity * self._policy.headway_time
            and number_expand < max_number_expand
        ):
            outgoing_edges = self.end_segment.outgoing_edges

            green_outgoing_edges = []
            unknown_outgoing_edges = []
            red_outgoing_edges = []

            for edge in outgoing_edges:
                # Intersection handling
                if edge.id in traffic_light_status[TrafficLightStatusType.GREEN]:
                    green_outgoing_edges.append(edge)
                elif edge.id in traffic_light_status[TrafficLightStatusType.RED]:
                    red_outgoing_edges.append(edge)
                else:
                    unknown_outgoing_edges.append(edge)

            if len(green_outgoing_edges) > 0:
                selected_outgoing_edges = green_outgoing_edges
            elif len(unknown_outgoing_edges) > 0:
                selected_outgoing_edges = unknown_outgoing_edges
            elif len(red_outgoing_edges) > 0:
                selected_outgoing_edges = red_outgoing_edges
            else:
                break

            # Select edge with the lowest curvature (prefer going straight)
            curvatures = [abs(edge.baseline_path.get_curvature_at_arc_length(0.0)) for edge in selected_outgoing_edges]
            idx = np.argmin(curvatures)
            new_segment = selected_outgoing_edges[idx]

            self._route.append(new_segment)
            self._path = create_path_from_se2(self.get_path_to_go() + new_segment.baseline_path.discrete_path)
            self._state.progress = 0
            number_expand += 1

    def _get_agent_at_progress(
        self,
        progress: float,
    ) -> Agent:
        """
        Returns the agent as a box at a given progress
        :param progress: the arc length along the agent's path
        :return: the agent as a Agent object at the given progress
        """
        # Caching
        if not self._requires_state_update:
            return self._full_agent_state

        if self._path is not None:
            init_pose = self._path.get_state_at_progress(progress)
            box = OrientedBox.from_new_pose(
                self._initial_state.box, StateSE2(init_pose.x, init_pose.y, init_pose.heading)
            )
            self._full_agent_state = Agent(
                metadata=self._initial_state.metadata,
                oriented_box=box,
                velocity=self._velocity_to_global_frame(init_pose.heading),
                tracked_object_type=self._initial_state.tracked_object_type,
                predictions=None,
            )

        else:
            self._full_agent_state = Agent(
                metadata=self._initial_state.metadata,
                oriented_box=self._initial_state.box,
                velocity=self._initial_state.velocity,
                tracked_object_type=self._initial_state.tracked_object_type,
                predictions=self._initial_state.predictions,
            )
        self._requires_state_update = False
        return self._full_agent_state

    def _clamp_progress(self, progress: float) -> float:
        """
        Clamp the progress to be between the agent's path bounds
        :param progress: [m] the progress along the agent's path
        :return: [m] the progress clamped between the start and end progress of the agent's path
        """
        return max(self._path.get_start_progress(), (min(progress, self._path.get_end_progress())))  # type: ignore

    def _convert_route_to_path(self) -> InterpolatedPath:
        """
        Converts the route into an InterpolatedPath
        :return: InterpolatedPath from the agent's route
        """
        blp: List[StateSE2] = []
        for segment in self._route:
            blp.extend(segment.baseline_path.discrete_path)
        return create_path_from_se2(blp)

    def _velocity_to_global_frame(self, heading: float) -> StateVector2D:
        """
        Transform agent's velocity along the path to global frame
        :param heading: [rad] The heading defining the transform to global frame.
        :return: The velocity vector in global frame.
        """
        return StateVector2D(self.velocity * np.cos(heading), self.velocity * np.sin(heading))


class SledgePedestrianAgent:
    """Pedestrian state class for agents in sledge."""

    def __init__(self, pedestrian: Agent):
        """
        Constructor for pedestrian agent object.
        :param pedestrian: generic agent interface
        """
        assert pedestrian.tracked_object_type == TrackedObjectType.PEDESTRIAN
        self._initial_pedestrian = pedestrian
        self._current_pedestrian = pedestrian
        self._active_time_s = 0.0

    def propagate(self, delta_t: float) -> None:
        """
        Propagates pedestrian agents based one constant velocity and heading.
        :param delta_t: time span to project agent in seconds
        """

        self._active_time_s += delta_t
        propagated_pose = project_state_se2(
            self._initial_pedestrian.center, self._active_time_s, self._initial_pedestrian.velocity.magnitude()
        )
        oriented_box = OrientedBox(
            propagated_pose,
            length=self._initial_pedestrian.box.length,
            width=self._initial_pedestrian.box.width,
            height=self._initial_pedestrian.box.height,
        )
        self._current_pedestrian = Agent(
            TrackedObjectType.PEDESTRIAN,
            oriented_box,
            self._initial_pedestrian.velocity,
            self._initial_pedestrian.metadata,
        )

    def is_active(self, ego_state: EgoState, radius: float) -> bool:
        """
        Return if the agent should be active at a simulation iteration
        :param iteration: the current simulation iteration
        :param radius: radius around ego to simulate
        :return: true if active, false otherwise
        """
        return pose_distance(self._current_pedestrian.center, ego_state.center) <= radius

    def get_pedestrian(self) -> Agent:
        """
        :return: current agent instance of pedestrian
        """
        return self._current_pedestrian


class SledgeObjectAgent:
    """Static object state class for agents in sledge."""

    def __init__(self, static_object: StaticObject):
        self._initial_static_object = static_object

    def is_active(self, ego_state: EgoState, radius: float) -> bool:
        """
        Return if the agent should be active at a simulation iteration
        :param iteration: the current simulation iteration
        :param radius: radius around ego to simulate
        :return: true if active, false otherwise
        """
        return pose_distance(self._initial_static_object.center, ego_state.center) <= radius

    def get_object(self) -> StaticObject:
        """
        :return: current static object instance
        """
        return self._initial_static_object
