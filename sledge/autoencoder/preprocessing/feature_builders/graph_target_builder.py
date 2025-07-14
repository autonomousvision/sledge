# TODO: v0.2 add doctrings

from typing import Dict, Type

import networkx as nx
import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses

from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_line_feature import get_lane_graph
from sledge.autoencoder.preprocessing.features.graph_feature import Graph, GraphConfig
from sledge.simulation.planner.pdm_planner.utils.pdm_array_representation import array_to_states_se2
from sledge.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


class GraphFeatureBuilder(AbstractFeatureBuilder):
    def __init__(
        self,
        graph_config: GraphConfig,
    ) -> None:

        self._frame = graph_config.frame
        self._pose_interval = graph_config.pose_interval

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "graph"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return Graph  # type: ignore

    def get_features_from_simulation(self, current_input: PlannerInput, initialization: PlannerInitialization) -> Graph:
        """Inherited, see superclass."""
        ego_state = current_input.history.ego_states[-1]
        map_api = initialization.map_api
        return self._compute_feature(ego_state, map_api)

    def get_features_from_scenario(self, scenario: AbstractScenario) -> Graph:
        """Inherited, see superclass."""
        ego_state = scenario.initial_ego_state
        map_api = scenario.map_api
        return self._compute_feature(ego_state, map_api)

    def _compute_feature(self, ego_state: EgoState, map_api: AbstractMap) -> Graph:
        """Inherited, see superclass."""

        # G_edges, edge_path_dict = get_edge_graph(ego_state, map_api, radius)
        radius = (self._frame[0] ** 2 + self._frame[1] ** 2) ** 0.5
        G_lane, lane_dict = get_lane_graph(ego_state, map_api, radius)
        lane_poses_dict: Dict[str, npt.NDArray[np.float32]] = {}
        for lane_id, lane in lane_dict.items():

            local_discrete_path: npt.NDArray[np.float32] = convert_absolute_to_relative_poses(
                ego_state.center, lane.baseline_path.discrete_path
            )
            lane_path = PDMPath(list(array_to_states_se2(local_discrete_path)))
            limit = (
                lane_path.length + self._pose_interval
                if lane_path.length % self._pose_interval > self._pose_interval / 2
                else lane_path.length
            )
            distances = np.arange(0, limit, step=self._pose_interval)
            lane_poses = lane_path.interpolate(distances, as_array=True)

            if len(lane_poses) < 2:
                G_lane.remove_node(lane_id)
                continue

            lane_poses_dict[lane_id] = lane_poses

        num_lines = len(lane_poses_dict.values())
        max_poses = max([len(poses) for poses in lane_poses_dict.values()])

        lines = np.zeros((num_lines, max_poses, 3), dtype=np.float32)  # (line, points, 3)
        line_adjacency = nx.to_numpy_array(G_lane)  # (line, line)
        mask = np.zeros((num_lines, max_poses), dtype=bool)  # (line, points)

        for line_idx, lane_poses in enumerate(lane_poses_dict.values()):
            lines[line_idx, : len(lane_poses)] = lane_poses
            mask[line_idx, : len(lane_poses)] = True

        return Graph(lines, line_adjacency, mask)
