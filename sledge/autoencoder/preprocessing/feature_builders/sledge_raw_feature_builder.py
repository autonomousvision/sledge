from typing import List, Type
from shapely.geometry import Polygon

from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType, SemanticMapLayer
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)

from sledge.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVectorRaw, SledgeConfig
from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_agent_feature import (
    compute_ego_features,
    compute_agent_features,
    compute_static_object_features,
)
from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_line_feature import (
    compute_line_features,
    compute_traffic_light_features,
)


class SledgeRawFeatureBuilder(AbstractFeatureBuilder):
    """Feature builder object for raw vector representation to train autoencoder in sledge"""

    def __init__(self, config: SledgeConfig):
        """
        Initializes the feature builder.
        :param config: configuration dataclass of autoencoder in sledge.
        """
        self._config = config

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "sledge_raw"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return SledgeVectorRaw

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> SledgeVectorRaw:
        """Inherited, see superclass."""

        history = current_input.history
        ego_state = history.ego_states[-1]
        detections = history.observations[-1]
        map_api = initialization.map_api
        traffic_light_data = current_input.traffic_light_data

        return self._compute_features(ego_state, map_api, detections, traffic_light_data)

    def get_features_from_scenario(self, scenario: AbstractScenario) -> SledgeVectorRaw:
        """Inherited, see superclass."""

        ego_state = scenario.initial_ego_state
        detections = scenario.initial_tracked_objects
        map_api = scenario.map_api
        traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))

        return self._compute_features(ego_state, map_api, detections, traffic_light_data)

    def _compute_features(
        self,
        ego_state: EgoState,
        map_api: AbstractMap,
        detections: DetectionsTracks,
        traffic_light_data: List[TrafficLightStatusData],
    ) -> SledgeVectorRaw:
        """
        Compute raw vector feature for autoencoder training.
        :param ego_state: object of ego vehicle state in nuPlan
        :param map_api: object of map in nuPlan
        :param detections: dataclass of detected objects in scenario
        :param traffic_light_data: dataclass of traffic lights in nuPlan
        :return: raw vector representation of lines, agents, objects, etc.
        """

        drivable_area_map = get_drivable_area_map(map_api, ego_state, self._config.radius)

        lines = compute_line_features(
            ego_state,
            map_api,
            self._config.radius,
            self._config.pose_interval,
        )
        vehicles = compute_agent_features(
            ego_state,
            detections,
            TrackedObjectType.VEHICLE,
            self._config.radius,
            drivable_area_map,
        )
        pedestrians = compute_agent_features(
            ego_state,
            detections,
            TrackedObjectType.PEDESTRIAN,
            self._config.radius,
        )
        static_objects = compute_static_object_features(
            ego_state,
            detections,
            self._config.radius,
            # drivable_area_map,
        )
        green_lights = compute_traffic_light_features(
            ego_state,
            map_api,
            traffic_light_data,
            TrafficLightStatusType.GREEN,
            self._config.radius,
            self._config.pose_interval,
        )
        red_lights = compute_traffic_light_features(
            ego_state,
            map_api,
            traffic_light_data,
            TrafficLightStatusType.RED,
            self._config.radius,
            self._config.pose_interval,
        )
        ego = compute_ego_features(ego_state)

        return SledgeVectorRaw(lines, vehicles, pedestrians, static_objects, green_lights, red_lights, ego)


def get_drivable_area_map(
    map_api: AbstractMap,
    ego_state: EgoState,
    map_radius: float,
    map_layers: List[SemanticMapLayer] = [
        SemanticMapLayer.ROADBLOCK,
        SemanticMapLayer.INTERSECTION,
    ],
) -> PDMOccupancyMap:
    """
    Helper function to create occupancy map of road polygons.
    :param map_api: object of map in nuPlan
    :param ego_state: object of ego vehicle state in nuPlan
    :param map_radius: radius around the ego vehicle to load polygons
    :param map_layers: layers to load, defaults to [ SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION, ]
    :return: occupancy map from PDM
    """
    # query all drivable map elements around ego position
    drivable_area = map_api.get_proximal_map_objects(ego_state.center.point, map_radius, map_layers)

    # collect lane polygons in list, save on-route indices
    drivable_polygons: List[Polygon] = []
    drivable_polygon_ids: List[str] = []

    for layer in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION]:
        for layer_object in drivable_area[layer]:
            drivable_polygons.append(layer_object.polygon)
            drivable_polygon_ids.append(layer_object.id)

    # create occupancy map with lane polygons
    drivable_area_map = PDMOccupancyMap(drivable_polygon_ids, drivable_polygons)

    return drivable_area_map
