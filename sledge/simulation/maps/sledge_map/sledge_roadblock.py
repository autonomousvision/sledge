from functools import cached_property
from typing import List

from shapely.geometry import Polygon

from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject, StopLine

from sledge.simulation.maps.sledge_map.sledge_map_graph import SledgeMapGraph
import sledge.simulation.maps.sledge_map.sledge_lane as sledge_lane


class SledgeRoadBlock(RoadBlockGraphEdgeMapObject):
    """Implementation of Roadblock in sledge."""

    def __init__(self, roadblock_id: str, sledge_map_graph: SledgeMapGraph):
        """
        Initialize roadblock interface of sledge.
        NOTE: SledgeRoadBlock wrapper of a single lane with same id.
        :param roadblock_id: unique identifier of roadblock.
        :param sledge_map_graph: lane map graph interface in sledge.
        """
        super().__init__(roadblock_id)
        self._sledge_map_graph = sledge_map_graph

    @cached_property
    def incoming_edges(self) -> List[RoadBlockGraphEdgeMapObject]:
        """Inherited from superclass."""
        incoming_ids = list(self._sledge_map_graph.directed_lane_graph.predecessors(self.id))
        return [SledgeRoadBlock(incoming_id, self._sledge_map_graph) for incoming_id in incoming_ids]

    @cached_property
    def outgoing_edges(self) -> List[RoadBlockGraphEdgeMapObject]:
        """Inherited from superclass."""
        outgoing_ids = list(self._sledge_map_graph.directed_lane_graph.successors(self.id))
        return [SledgeRoadBlock(outgoing_id, self._sledge_map_graph) for outgoing_id in outgoing_ids]

    @cached_property
    def interior_edges(self) -> List[LaneGraphEdgeMapObject]:
        """Inherited from superclass."""
        # NOTE: Additional heuristic of grouping lanes in roadblock could be added.
        lane_ids = [self.id]
        return [sledge_lane.SledgeLane(lane_id, self._sledge_map_graph) for lane_id in lane_ids]

    @cached_property
    def polygon(self) -> Polygon:
        """Inherited from superclass."""
        return self._sledge_map_graph.polygon_dict[self.id]

    @cached_property
    def children_stop_lines(self) -> List[StopLine]:
        """Inherited from superclass."""
        raise NotImplementedError

    @cached_property
    def parallel_edges(self) -> List[RoadBlockGraphEdgeMapObject]:
        """Inherited from superclass."""
        raise NotImplementedError
