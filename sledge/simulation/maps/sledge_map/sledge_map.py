from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import numpy.typing as npt

import shapely.geometry as geom
from shapely.geometry import Point

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap, MapObject
from nuplan.common.maps.abstract_map_objects import Lane, RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import RasterLayer, RasterMap, SemanticMapLayer, VectorLayer
from nuplan.common.maps.nuplan_map.utils import raster_layer_from_map_layer
from nuplan.database.maps_db.layer import MapLayer

from sledge.simulation.maps.sledge_map.sledge_map_graph import construct_sledge_map_graph, SledgeMapGraph
from sledge.simulation.maps.sledge_map.sledge_lane import SledgeLane
from sledge.simulation.maps.sledge_map.sledge_roadblock import SledgeRoadBlock
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVector


AVAILABLE_MAP_LAYERS = [SemanticMapLayer.LANE, SemanticMapLayer.ROADBLOCK]


class SledgeMap(AbstractMap):
    """Implementation of map api in sledge."""

    def __init__(self, sledge_vector: SledgeVector, map_name: Optional[str] = None) -> None:

        self._sledge_vector = sledge_vector
        self._sledge_map_graph: SledgeMapGraph = construct_sledge_map_graph(sledge_vector)

        self._vector_map: Dict[str, VectorLayer] = defaultdict(VectorLayer)
        self._raster_map: Dict[str, RasterLayer] = defaultdict(RasterLayer)
        self._map_objects: Dict[SemanticMapLayer, Dict[str, MapObject]] = defaultdict(dict)
        self._map_name = map_name if map_name else "undefined"

        self._map_object_getter: Dict[SemanticMapLayer, Callable[[str], MapObject]] = {
            SemanticMapLayer.LANE: self._get_lane,
            SemanticMapLayer.LANE_CONNECTOR: self._get_not_available,
            SemanticMapLayer.ROADBLOCK: self._get_roadblock,
            SemanticMapLayer.ROADBLOCK_CONNECTOR: self._get_not_available,
            SemanticMapLayer.STOP_LINE: self._get_not_available,
            SemanticMapLayer.CROSSWALK: self._get_not_available,
            SemanticMapLayer.INTERSECTION: self._get_not_available,
            SemanticMapLayer.WALKWAYS: self._get_not_available,
            SemanticMapLayer.CARPARK_AREA: self._get_not_available,
        }

        self._vector_layer_mapping = {
            SemanticMapLayer.LANE: "lanes_polygons",
            SemanticMapLayer.ROADBLOCK: "lane_groups_polygons",
            SemanticMapLayer.INTERSECTION: "intersections",
            SemanticMapLayer.STOP_LINE: "stop_polygons",
            SemanticMapLayer.CROSSWALK: "crosswalks",
            SemanticMapLayer.DRIVABLE_AREA: "drivable_area",
            SemanticMapLayer.LANE_CONNECTOR: "lane_connectors",
            SemanticMapLayer.ROADBLOCK_CONNECTOR: "lane_group_connectors",
            SemanticMapLayer.BASELINE_PATHS: "baseline_paths",
            SemanticMapLayer.BOUNDARIES: "boundaries",
            SemanticMapLayer.WALKWAYS: "walkways",
            SemanticMapLayer.CARPARK_AREA: "carpark_areas",
        }
        self._raster_layer_mapping = {
            SemanticMapLayer.DRIVABLE_AREA: "drivable_area",
        }

        # Special vector layer mapping for lane connector polygons.
        self._LANE_CONNECTOR_POLYGON_LAYER = "gen_lane_connectors_scaled_width_polygons"

    def __reduce__(self) -> Tuple[Type[SledgeMap], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        This object is reconstructed by pickle to avoid serializing potentially large state/caches.
        :return: Object type and constructor arguments to be used.
        """
        return self.__class__, (self._sledge_vector, self._map_name)

    @property
    def map_name(self) -> str:
        """Inherited, see superclass."""
        return self._map_name

    @property
    def sledge_map_graph(self) -> SledgeMapGraph:
        """Inherited, see superclass."""
        return self._sledge_map_graph

    def get_available_map_objects(self) -> List[SemanticMapLayer]:
        """Inherited, see superclass."""
        return list(self._map_object_getter.keys())

    def get_available_raster_layers(self) -> List[SemanticMapLayer]:
        """Inherited, see superclass."""
        return list(self._raster_layer_mapping.keys())

    def get_raster_map_layer(self, layer: SemanticMapLayer) -> RasterLayer:
        """Inherited, see superclass."""
        # FIXME: Update for Sledge
        layer_id = self._semantic_raster_layer_map(layer)
        return self._load_raster_layer(layer_id)

    def get_raster_map(self, layers: List[SemanticMapLayer]) -> RasterMap:
        """Inherited, see superclass."""
        # FIXME: Update for Sledge
        raster_map = RasterMap(layers=defaultdict(RasterLayer))
        for layer in layers:
            raster_map.layers[layer] = self.get_raster_map_layer(layer)
        return raster_map

    def is_in_layer(self, point: Point2D, layer: SemanticMapLayer) -> bool:
        """Inherited, see superclass."""

        # TODO add functionality for other map objects
        if layer in AVAILABLE_MAP_LAYERS:
            return len(self._sledge_map_graph.occupancy_map.query(Point(*point))) > 0

        return False

    def get_all_map_objects(self, point: Point2D, layer: SemanticMapLayer) -> List[MapObject]:
        """Inherited, see superclass."""

        if layer in AVAILABLE_MAP_LAYERS:
            contains_idx = self._sledge_map_graph.occupancy_map.query(Point(*point))
            contains_ids = [self._sledge_map_graph.occupancy_map.tokens[idx] for idx in contains_idx]
            return [self.get_map_object(object_id, layer) for object_id in contains_ids]

        return []

    def get_one_map_object(self, point: Point2D, layer: SemanticMapLayer) -> Optional[MapObject]:
        """Inherited, see superclass."""

        # NOTE: no idea what's the purpose of this function
        map_objects = self.get_all_map_objects(point, layer)
        if len(map_objects) == 0:
            return None

        return map_objects[0]

    def get_proximal_map_objects(
        self, point: Point2D, radius: float, layers: List[SemanticMapLayer]
    ) -> Dict[SemanticMapLayer, List[MapObject]]:
        """Inherited, see superclass."""
        x_min, x_max = point.x - radius, point.x + radius
        y_min, y_max = point.y - radius, point.y + radius
        patch = geom.box(x_min, y_min, x_max, y_max)

        # TODO: might remove these extra statements
        supported_layers = self.get_available_map_objects()
        unsupported_layers = [layer for layer in layers if layer not in supported_layers]
        assert len(unsupported_layers) == 0, f"Object representation for layer(s): {unsupported_layers} is unavailable"

        object_map: Dict[SemanticMapLayer, List[MapObject]] = defaultdict(list)

        for layer in layers:
            object_map[layer] = self._get_proximity_map_object(patch, layer)

        return object_map

    def get_map_object(self, object_id: str, layer: SemanticMapLayer) -> Optional[MapObject]:
        """Inherited, see superclass."""
        try:
            if object_id not in self._map_objects[layer]:
                map_object: MapObject = self._map_object_getter[layer](object_id)
                self._map_objects[layer][object_id] = map_object

            return self._map_objects[layer][object_id]
        except KeyError:
            raise ValueError(f"Object representation for layer: {layer.name} object: {object_id} is unavailable")

    def get_distance_to_nearest_map_object(
        self, point: Point2D, layer: SemanticMapLayer
    ) -> Tuple[Optional[str], Optional[float]]:
        """Inherited from superclass."""
        # TODO add functionality for other map objects
        if layer in AVAILABLE_MAP_LAYERS:
            (
                nearest_idcs,
                nearest_distances,
            ) = self._sledge_map_graph.occupancy_map._str_tree.query_nearest(Point(*point), return_distance=True)

            nearest_id = self._sledge_map_graph.occupancy_map.tokens[nearest_idcs[0]]
            nearest_distance = nearest_distances[0]
            return nearest_id, nearest_distance

        return None, None

    def get_distance_to_nearest_raster_layer(self, point: Point2D, layer: SemanticMapLayer) -> float:
        """Inherited from superclass"""
        raise NotImplementedError

    def get_distances_matrix_to_nearest_map_object(
        self, points: List[Point2D], layer: SemanticMapLayer
    ) -> Optional[npt.NDArray[np.float64]]:
        """
        Returns the distance matrix (in meters) between a list of points and their nearest desired surface.
            That distance is the L1 norm from the point to the closest location on the surface.
        :param points: [m] A list of x, y coordinates in global frame.
        :param layer: A semantic layer to query.
        :return: An array of shortest distance from each point to the nearest desired surface.
        """
        raise NotImplementedError

    def initialize_all_layers(self) -> None:
        """
        Load all layers to vector map
        :param: None
        :return: None
        """
        pass

    def _semantic_vector_layer_map(self, layer: SemanticMapLayer) -> str:
        """
        Mapping from SemanticMapLayer int to MapsDB internal representation of vector layers.
        :param layer: The querired semantic map layer.
        :return: A internal layer name as a string.
        @raise ValueError if the requested layer does not exist for MapsDBMap
        """
        try:
            return self._vector_layer_mapping[layer]
        except KeyError:
            raise ValueError("Unknown layer: {}".format(layer.name))

    def _semantic_raster_layer_map(self, layer: SemanticMapLayer) -> str:
        """
        Mapping from SemanticMapLayer int to MapsDB internal representation of raster layers.
        :param layer: The queried semantic map layer.
        :return: A internal layer name as a string.
        @raise ValueError if the requested layer does not exist for MapsDBMap
        """
        try:
            return self._raster_layer_mapping[layer]
        except KeyError:
            raise ValueError("Unknown layer: {}".format(layer.name))

    def _get_vector_map_layer(self, layer: SemanticMapLayer) -> VectorLayer:
        """Inherited, see superclass."""
        layer_id = self._semantic_vector_layer_map(layer)
        return self._load_vector_map_layer(layer_id)

    def _load_raster_layer(self, layer_name: str) -> RasterLayer:
        """
        Load and cache raster layers.
        :layer_name: the name of the vector layer to be loaded.
        :return: the loaded RasterLayer.
        """
        if layer_name not in self._raster_map:
            map_layer: MapLayer = self._maps_db.load_layer(self._map_name, layer_name)
            self._raster_map[layer_name] = raster_layer_from_map_layer(map_layer)

        return self._raster_map[layer_name]

    def _load_vector_map_layer(self, layer_name: str) -> VectorLayer:
        """
        Load and cache vector layers.
        :layer_name: the name of the vector layer to be loaded.
        :return: the loaded VectorLayer.
        """
        if layer_name not in self._vector_map:
            if layer_name == "drivable_area":
                self._initialize_drivable_area()
            else:
                self._vector_map[layer_name] = self._maps_db.load_vector_layer(self._map_name, layer_name)
        return self._vector_map[layer_name]

    def _get_proximity_map_object(self, patch: geom.Polygon, layer: SemanticMapLayer) -> List[MapObject]:
        """
        Gets nearby lanes within the given patch.
        :param patch: The area to be checked.
        :param layer: desired layer to check.
        :return: A list of map objects.
        """

        if layer in AVAILABLE_MAP_LAYERS:
            map_object_ids = self._sledge_map_graph.occupancy_map.intersects(patch)
            return [self.get_map_object(map_object_id, layer) for map_object_id in map_object_ids]
        return []

    def _get_not_available(self, object_id: str) -> MapObject:
        """
        Placeholder for map object getter that is not available in SledgeMap
        """
        return None

    def _get_lane(self, lane_id: str) -> Lane:
        """
        Gets the lane with the given lane id.
        :param lane_id: Desired unique id of a lane that should be extracted.
        :return: Lane object.
        """
        return SledgeLane(lane_id, self._sledge_map_graph)

    def _get_roadblock(self, roadblock_id: str) -> RoadBlockGraphEdgeMapObject:
        """
        Gets the roadblock with the given roadblock_id.
        :param roadblock_id: Desired unique id of a roadblock that should be extracted.
        :return: RoadBlock object.
        """
        return SledgeRoadBlock(roadblock_id, self._sledge_map_graph)
