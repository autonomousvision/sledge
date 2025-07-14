from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from sledge.autoencoder.preprocessing.features.graph_feature import Graph, GraphConfig
from sledge.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle


@dataclass
class ReconstructionMetricConfig:
    radius: float = 50
    topo_subsampling: int = 10
    dist_thresh: float = 1.5
    angl_thresh: float = np.deg2rad(60)


def convert_to_digraph(graph: Graph) -> nx.DiGraph:

    lines, line_adjacency, mask = graph.lines, graph.line_adjacency, graph.mask

    G = nx.DiGraph()

    node_counter = 0
    line_start_node: Dict[int, int] = {}
    line_end_node: Dict[int, int] = {}

    for line_i, (line, line_mask) in enumerate(zip(lines, mask)):
        for pose_i, (pose, pose_mask) in enumerate(zip(line, line_mask)):
            if not pose_mask:
                break
            elif pose_i == 0:
                line_start_node[line_i] = node_counter
                G.add_node(node_counter, pose=pose, line_id=line_i)
            else:
                line_end_node[line_i] = node_counter  # overwrite
                G.add_node(node_counter, pose=pose, line_id=line_i)
                G.add_edge(node_counter - 1, node_counter)
            node_counter += 1

    num_lines = len(lines)
    for line_i in range(num_lines):
        for line_j in range(num_lines):
            if line_adjacency[line_i, line_j] > 0.0:
                G.add_edge(line_end_node[line_i], line_start_node[line_j])

    return G


def keep_nodes_in_frame(G: nx.DiGraph, graph_config: GraphConfig) -> nx.DiGraph:
    def _pose_in_frame(pose: npt.NDArray[np.float32]) -> bool:
        width, height = graph_config.frame

        within_width = -width / 2 <= pose[0] <= width / 2
        within_height = -height / 2 <= pose[1] <= height / 2

        return within_width and within_height

    nodes_to_remove: List[int] = []
    for node, attr in G.nodes(data=True):
        if not _pose_in_frame(attr["pose"]):
            nodes_to_remove.append(node)

    G.remove_nodes_from(nodes_to_remove)
    return G


def node_poses(G: nx.DiGraph, indices: Optional[List[int]] = None) -> npt.NDArray[np.float32]:

    if indices is None:
        return np.array([data["pose"] for node, data in G.nodes(data=True)])

    return np.array([G.nodes[idx]["pose"] for idx in indices])


def one_to_one_matching(
    G_a: nx.DiGraph, G_b: nx.DiGraph, metric_config: ReconstructionMetricConfig
) -> List[Tuple[int, int]]:

    matched_pairs = []
    a_nodes, b_nodes = list(G_a.nodes), list(G_b.nodes)

    if len(a_nodes) > 0 and len(b_nodes) > 0:

        a_idcs = [i for i in a_nodes]
        b_idcs = [j for j in b_nodes]

        poses_a, poses_b = node_poses(G_a, a_idcs), node_poses(G_b, b_idcs)

        dist_matrix = cdist(poses_a[..., :2], poses_b[..., :2])
        angl_matrix = np.abs(normalize_angle(poses_a[:, None, 2] - poses_b[None, :, 2]))
        invalid_match_mask = np.logical_or(
            dist_matrix > metric_config.dist_thresh, angl_matrix > metric_config.angl_thresh
        )

        dist_matrix[invalid_match_mask] = 1e9
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        # Filter out assignments with distances above the threshold
        matched_pairs = [(a_idcs[i], b_idcs[j]) for i, j in zip(row_ind, col_ind) if dist_matrix[i, j] < 1e9]

    return matched_pairs


def distance(node_a, node_b):
    sq_distance = (node_a[0] - node_b[0]) ** 2 + (node_a[1] - node_b[1]) ** 2
    return sq_distance**0.5


# Function to find all reachable nodes within 50m
def find_nodes_within_dist(G, start_node, radius: float = 50):
    reachable_nodes = set()
    visited = set()
    stack = [start_node]  # Stack for DFS, with (node, distance) tuples

    while stack:
        current_node = stack.pop()

        if current_node not in visited:
            visited.add(current_node)

            if distance(G.nodes[start_node]["pose"], G.nodes[current_node]["pose"]) <= radius:
                reachable_nodes.add(current_node)

                for neighbor in G.neighbors(current_node):
                    stack.append(neighbor)

    return reachable_nodes
