from collections import defaultdict
from typing import Dict, List, Optional, Type

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.common.actor_state.ego_state import EgoState

from sledge.simulation.observation.sledge_idm.sledge_idm_agent_manager import SledgeIDMAgentManager
from sledge.simulation.observation.sledge_idm.sledge_idm_agents_builder import build_idm_agents_on_map_rails


class SledgeIDMAgents(AbstractObservation):
    """Observation object for simulated agents in sledge."""

    def __init__(
        self,
        target_velocity: float,
        min_gap_to_lead_agent: float,
        headway_time: float,
        accel_max: float,
        decel_max: float,
        static_detections_types: List[str],
        scenario: AbstractScenario,
        minimum_path_length: float = 20,
        radius: float = 64,
    ):
        """
        Constructor for SledgeIDMAgents
        :param target_velocity: [m/s] Desired velocity in free traffic
        :param min_gap_to_lead_agent: [m] Minimum relative distance to lead vehicle
        :param headway_time: [s] Desired time headway. The minimum possible time to the vehicle in front
        :param accel_max: [m/s^2] maximum acceleration
        :param decel_max: [m/s^2] maximum deceleration (positive value)
        :param static_detections_types: object classes considered static.
        :param scenario: scenario interface during simulation
        :param minimum_path_length: [m] The minimum path length to maintain., defaults to 20
        :param radius: [m] Agents within this radius around the ego will be simulated, defaults to 64
        """
        self.current_iteration = 0

        self._target_velocity = target_velocity
        self._min_gap_to_lead_agent = min_gap_to_lead_agent
        self._headway_time = headway_time
        self._accel_max = accel_max
        self._decel_max = decel_max
        self._scenario = scenario
        self._static_detections_types: List[TrackedObjectType] = []

        self._minimum_path_length = minimum_path_length
        self._radius = radius

        # Prepare IDM agent manager
        self.current_ego_state: EgoState = scenario.initial_ego_state
        self._agent_manager: Optional[SledgeIDMAgentManager] = None
        self._initialize_static_detection_types(static_detections_types)

    def reset(self) -> None:
        """Inherited, see superclass."""
        self.current_iteration = 0
        self._agent_manager = None

    def _initialize_static_detection_types(self, static_detections_types: List[str]) -> None:
        """
        Converts the input static objects types to corresponding enums in nuPlan.
        :param static_detections_types: list of static object names as strings
        :raises ValueError: if string object type cannot be matched to TrackedObjectType enum
        """
        for _type in static_detections_types:
            try:
                self._static_detections_types.append(TrackedObjectType[_type])
            except KeyError:
                raise ValueError(f"The given detection type {_type} does not exist or is not supported!")

    def _get_agent_manager(self) -> SledgeIDMAgentManager:
        """
        Create sledge idm agent manager in case it does not already exists
        :return: SledgeIDMAgentManager
        """
        if not self._agent_manager:
            unique_vehicles, unique_pedestrians, unique_static_objects, occupancy_map = build_idm_agents_on_map_rails(
                self._target_velocity,
                self._min_gap_to_lead_agent,
                self._headway_time,
                self._accel_max,
                self._decel_max,
                self._minimum_path_length,
                self._scenario,
                self._static_detections_types,
            )
            self._agent_manager = SledgeIDMAgentManager(
                unique_vehicles,
                unique_pedestrians,
                unique_static_objects,
                occupancy_map,
                self._scenario.map_api,
                self._radius,
            )

        return self._agent_manager

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def initialize(self) -> None:
        """Inherited, see superclass."""
        pass

    def get_observation(self) -> DetectionsTracks:
        """Inherited, see superclass."""
        detections = self._get_agent_manager().get_active_agents(self.current_ego_state)
        return detections

    def update_observation(
        self,
        iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        history: SimulationHistoryBuffer,
    ) -> None:
        """Inherited, see superclass."""
        ego_state, _ = history.current_state

        self.current_iteration = next_iteration.index
        self.current_ego_state = ego_state

        tspan = next_iteration.time_s - iteration.time_s
        traffic_light_data = self._scenario.get_traffic_light_status_at_iteration(self.current_iteration)

        # Extract traffic light data into Dict[traffic_light_status, lane_connector_ids]
        traffic_light_status: Dict[TrafficLightStatusType, List[str]] = defaultdict(list)

        for data in traffic_light_data:
            traffic_light_status[data.status].append(str(data.lane_connector_id))

        self._get_agent_manager().propagate_agents(
            ego_state,
            tspan,
            traffic_light_status,
        )
