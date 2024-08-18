from __future__ import annotations

# import warnings
from pathlib import Path
from typing import Any, Generator, List, Optional, Set, Tuple, Type, cast

import numpy as np
from shapely.geometry import Point

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, SensorChannel, Sensors
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateVector2D, StateSE2, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import (
    TrafficLightStatusData,
    TrafficLightStatuses,
    TrafficLightStatusType,
    Transform,
)

from sledge.simulation.maps.sledge_map.sledge_map import SledgeMap
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVector
from sledge.simulation.scenarios.sledge_scenario.sledge_scenario_utils import (
    project_sledge_vector,
    sledge_vector_to_detection_tracks,
    sample_future_indices,
    get_route,
)


# TODO: add to some config
FUTURE_SAMPLING = TrajectorySampling(time_horizon=15, interval_length=0.1)
TRAFFIC_LIGHT_FLIP = 10.0  # [s]
ROUTE_LENGTH = 32  # [m]


class SledgeScenario(AbstractScenario):
    """Scenario implementation for sledge that is used for simulation in Lane & Agent mode."""

    def __init__(self, data_root: Path) -> None:
        """ """

        self._data_root = data_root
        self._initial_lidar_token = data_root.parent.name
        self._sledge_vector: SledgeVector = FeatureCachePickle().load_computed_feature_from_folder(
            data_root, SledgeVector
        )
        self._map_api = SledgeMap(self._sledge_vector)

        self._log_file = data_root
        self._log_name: str = data_root.parent.parent.parent.name
        self._scenario_type: str = data_root.parent.parent.name

        self._ego_vehicle_parameters = get_pacifica_parameters()

        # TODO: add to some config
        self._future_sampling = FUTURE_SAMPLING
        self._time_points = [
            TimePoint(int(time_s * 1e6))
            for time_s in np.arange(0, self._future_sampling.time_horizon, self._future_sampling.interval_length)
        ]
        self._number_of_iterations = len(self._time_points)

        self._route_roadblock_ids, self._route_path = get_route(self._map_api)

    def __reduce__(self) -> Tuple[Type[SledgeScenario], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return (self.__class__, (self._data_root,))

    @property
    def num_agents(self) -> int:
        detection_tracks = self.get_tracked_objects_at_iteration(0)
        detection_tracks.tracked_objects.get_agents()
        return len(detection_tracks.tracked_objects.get_agents())

    def _is_iteration_flipped(self, iteration: int) -> bool:
        flip_iteration_interval = int(TRAFFIC_LIGHT_FLIP / self._future_sampling.interval_length)
        segment = iteration // flip_iteration_interval
        return segment % 2 == 1

    @property
    def ego_vehicle_parameters(self) -> VehicleParameters:
        """Inherited, see superclass."""
        return self._ego_vehicle_parameters

    @property
    def token(self) -> str:
        """Inherited, see superclass."""
        return self._initial_lidar_token

    @property
    def log_name(self) -> str:
        """Inherited, see superclass."""
        return self._log_name

    @property
    def scenario_name(self) -> str:
        """Inherited, see superclass."""
        return self.token

    @property
    def scenario_type(self) -> str:
        """Inherited, see superclass."""
        return self._scenario_type

    @property
    def map_api(self) -> AbstractMap:
        """Inherited, see superclass."""
        return self._map_api

    @property
    def map_root(self) -> str:
        """Get the map root folder."""
        return self._map_root

    @property
    def map_version(self) -> str:
        """Get the map version."""
        return self._map_version

    @property
    def database_interval(self) -> float:
        """Inherited, see superclass."""
        return 0.05  # 20Hz

    def get_number_of_iterations(self) -> int:
        """Inherited, see superclass."""
        return self._future_sampling.num_poses

    def get_lidar_to_ego_transform(self) -> Transform:
        """Inherited, see superclass."""
        raise NotImplementedError

    def get_mission_goal(self) -> Optional[StateSE2]:
        """Inherited, see superclass."""
        last_iteration = self.get_number_of_iterations() - 1
        return self.get_ego_state_at_iteration(last_iteration).center

    def get_route_roadblock_ids(self) -> List[str]:
        """Inherited, see superclass."""

        roadblock_ids = self._route_roadblock_ids
        return cast(List[str], roadblock_ids)

    def get_expert_goal_state(self) -> StateSE2:
        """Inherited, see superclass."""
        last_iteration = self.get_number_of_iterations() - 1
        return self.get_ego_state_at_iteration(last_iteration).center

    def get_time_point(self, iteration: int) -> TimePoint:
        """Inherited, see superclass."""
        assert (
            0 <= iteration < self.get_number_of_iterations()
        ), f"Iteration {iteration} out of bound of {self.get_number_of_iterations()} iterations!"

        return self._time_points[iteration]

    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """Inherited, see superclass."""
        initial_distance = self._route_path.project(Point(0, 0))

        distance_per_iteration = ROUTE_LENGTH / self._future_sampling.num_poses

        center = self._route_path.interpolate([initial_distance + distance_per_iteration * iteration])[0]
        center_velocity_2d = StateVector2D(float(self._sledge_vector.ego.states), 0)
        center_acceleration_2d = StateVector2D(0, 0)

        # project ego with constant velocity and heading
        time_point = self.get_time_point(iteration)

        return EgoState.build_from_center(
            center=center,
            center_velocity_2d=center_velocity_2d,
            center_acceleration_2d=center_acceleration_2d,
            tire_steering_angle=0.0,
            time_point=time_point,
            vehicle_parameters=self._ego_vehicle_parameters,
        )

    def get_tracked_objects_at_iteration(
        self,
        iteration: int,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        """Inherited, see superclass."""
        assert 0 <= iteration < self.get_number_of_iterations(), f"Iteration is out of scenario: {iteration}!"

        # if not future_trajectory_sampling:
        #     warnings.warn("SledgeScenario: TrajectorySampling in get_tracked_objects_at_iteration() not supported.")

        time_point = self.get_time_point(iteration)
        projected_sledge_vector = project_sledge_vector(self._sledge_vector, time_point.time_s)
        detection_tracks = sledge_vector_to_detection_tracks(projected_sledge_vector, time_point.time_us)
        return detection_tracks

    def get_tracked_objects_within_time_window_at_iteration(
        self,
        iteration: int,
        past_time_horizon: float,
        future_time_horizon: float,
        filter_track_tokens: Optional[Set[str]] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        """Inherited, see superclass."""
        assert 0 <= iteration < self.get_number_of_iterations(), f"Iteration is out of scenario: {iteration}!"
        raise NotImplementedError

    def get_sensors_at_iteration(self, iteration: int, channels: Optional[List[SensorChannel]] = None) -> Sensors:
        """Inherited, see superclass."""
        raise NotImplementedError

    def get_future_timestamps(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TimePoint, None, None]:
        """Inherited, see superclass."""
        indices = sample_future_indices(self._future_sampling, iteration, time_horizon, num_samples)
        for idx in indices:
            yield self.get_time_point(idx)

    def get_past_timestamps(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TimePoint, None, None]:
        """Inherited, see superclass."""
        yield from []  # placeholder

    def get_ego_past_trajectory(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[EgoState, None, None]:
        """Inherited, see superclass."""
        # indices = sample_future_indices(self._future_sampling, iteration, time_horizon, num_samples)
        for idx in [0]:
            yield self.get_ego_state_at_iteration(idx)

    def get_ego_future_trajectory(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[EgoState, None, None]:
        """Inherited, see superclass."""
        indices = sample_future_indices(self._future_sampling, iteration, time_horizon, num_samples)
        for idx in indices:
            yield self.get_ego_state_at_iteration(idx)

    def get_past_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        """Inherited, see superclass."""
        # indices = sample_future_indices(self._future_sampling, iteration, time_horizon, num_samples)
        for idx in [0]:
            yield self.get_tracked_objects_at_iteration(idx)

    def get_future_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        """Inherited, see superclass."""

        # if not future_trajectory_sampling:
        #     warnings.warn("SledgeScenario: TrajectorySampling not available for get_future_tracked_objects")

        indices = sample_future_indices(self._future_sampling, iteration, time_horizon, num_samples)
        for idx in indices:
            yield self.get_tracked_objects_at_iteration(idx)

    def get_past_sensors(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        channels: Optional[List[SensorChannel]] = None,
    ) -> Generator[Sensors, None, None]:
        """Inherited, see superclass."""
        raise NotImplementedError

    def get_traffic_light_status_at_iteration(self, iteration: int) -> Generator[TrafficLightStatusData, None, None]:
        """Inherited, see superclass."""

        flip_traffic_lights = self._is_iteration_flipped(iteration)

        def _get_status_type(status_type: TrafficLightStatusType) -> TrafficLightStatusType:
            if flip_traffic_lights:
                if status_type == TrafficLightStatusType.RED:
                    return TrafficLightStatusType.GREEN
                else:
                    return TrafficLightStatusType.RED
            else:
                return status_type

        for lane_id, traffic_light in self._map_api.sledge_map_graph.traffic_light_dict.items():
            yield TrafficLightStatusData(_get_status_type(traffic_light.status), lane_id, self._time_points[iteration])

    def get_past_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TrafficLightStatuses, None, None]:
        """
        Gets past traffic light status.

        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
        :param time_horizon [s]: the desired horizon to the past.
        :param num_samples: number of entries in the future, if None it will be deduced from the DB.
        :return: Generator object for traffic light history to the past.
        """
        # FIXME: add traffic light stats
        yield from []  # placeholder

    def get_future_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TrafficLightStatuses, None, None]:
        """
        Gets future traffic light status.

        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
        :param time_horizon [s]: the desired horizon to the future.
        :param num_samples: number of entries in the future, if None it will be deduced from the DB.
        :return: Generator object for traffic light history to the future.
        """
        # FIXME: add traffic light stats
        yield from []  # placeholder

    def get_scenario_tokens(self) -> List[str]:
        """Return the list of lidarpc tokens from the DB that are contained in the scenario."""
        raise NotImplementedError
