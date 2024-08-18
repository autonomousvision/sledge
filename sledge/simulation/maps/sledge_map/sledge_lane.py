from functools import cached_property
from typing import List, Optional, Tuple

from shapely.geometry import Polygon, LineString

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map_objects import (
    Lane,
    LaneGraphEdgeMapObject,
    PolylineMapObject,
    RoadBlockGraphEdgeMapObject,
)

from sledge.simulation.maps.sledge_map.sledge_map_graph import SledgeMapGraph
from sledge.simulation.maps.sledge_map.sledge_polyline import SledgePolylineMapObject
import sledge.simulation.maps.sledge_map.sledge_roadblock as sledge_roadblock


class SledgeLane(Lane):
    """Lane implementation for sledge map."""

    def __init__(
        self,
        lane_id: str,
        sledge_map_graph: SledgeMapGraph,
        speed_limit_mps: float = 15.0,
    ):
        """
        Initializes sledge lane
        :param lane_id: unique identifies of lane
        :param sledge_map_graph: map graph interface in sledge
        :param speed_limit_mps: speed limit in [m/s], defaults to 15.0
        """
        super().__init__(lane_id)
        self._sledge_map_graph = sledge_map_graph
        self._speed_limit_mps = speed_limit_mps

    @cached_property
    def incoming_edges(self) -> List[LaneGraphEdgeMapObject]:
        """Inherited from superclass."""
        incoming_ids = list(self._sledge_map_graph.directed_lane_graph.predecessors(self.id))
        return [SledgeLane(incoming_id, self._sledge_map_graph) for incoming_id in incoming_ids]

    @cached_property
    def outgoing_edges(self) -> List[LaneGraphEdgeMapObject]:
        """Inherited from superclass."""
        outgoing_ids = list(self._sledge_map_graph.directed_lane_graph.successors(self.id))
        return [SledgeLane(outgoing_id, self._sledge_map_graph) for outgoing_id in outgoing_ids]

    @cached_property
    def parallel_edges(self) -> List[LaneGraphEdgeMapObject]:
        """Inherited from superclass"""
        raise NotImplementedError

    @cached_property
    def baseline_path(self) -> PolylineMapObject:
        """Inherited from superclass."""
        linestring = LineString(self._sledge_map_graph.baseline_paths_dict[self.id][:, :2])
        return SledgePolylineMapObject(self.id, linestring)

    @property
    def traffic_light_baseline_path(self) -> Optional[PolylineMapObject]:
        """Property indicating the section of the baseline path with traffic light signal."""
        if self.id in self._sledge_map_graph.traffic_light_dict.keys():
            poses = self._sledge_map_graph.traffic_light_dict[self.id].poses
            return SledgePolylineMapObject(self.id, LineString(poses[:, :2]))
        return None

    @property
    def traffic_light_polygon(self) -> Polygon:
        """Property of traffic light polygon."""
        if self.id in self._sledge_map_graph.traffic_light_dict.keys():
            polygon = self._sledge_map_graph.traffic_light_dict[self.id].polygon
            return polygon
        return None

    @cached_property
    def left_boundary(self) -> PolylineMapObject:
        """Inherited from superclass."""
        raise NotImplementedError

    @cached_property
    def right_boundary(self) -> PolylineMapObject:
        """Inherited from superclass."""
        raise NotImplementedError

    def get_roadblock_id(self) -> str:
        """Inherited from superclass."""
        return self.id

    @cached_property
    def parent(self) -> RoadBlockGraphEdgeMapObject:
        """Inherited from superclass"""
        return sledge_roadblock.SledgeRoadBlock(self.id, self._sledge_map_graph)

    @cached_property
    def speed_limit_mps(self) -> Optional[float]:
        """Inherited from superclass."""
        return self._speed_limit_mps

    @cached_property
    def polygon(self) -> Polygon:
        """Inherited from superclass."""
        return self._sledge_map_graph.polygon_dict[self.id]

    def is_left_of(self, other: Lane) -> bool:
        """Inherited from superclass."""
        raise NotImplementedError

    def is_right_of(self, other: Lane) -> bool:
        """Inherited from superclass."""
        raise NotImplementedError

    @cached_property
    def adjacent_edges(
        self,
    ) -> Tuple[Optional[LaneGraphEdgeMapObject], Optional[LaneGraphEdgeMapObject]]:
        """Inherited from superclass."""
        raise NotImplementedError

    def get_width_left_right(
        self, point: Point2D, include_outside: bool = False
    ) -> Tuple[Optional[float], Optional[float]]:
        """Inherited from superclass."""
        raise NotImplementedError

    def oriented_distance(self, point: Point2D) -> float:
        """Inherited from superclass"""
        raise NotImplementedError

    @cached_property
    def index(self) -> int:
        """Inherited from superclass"""
        raise NotImplementedError
