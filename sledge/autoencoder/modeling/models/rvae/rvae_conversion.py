from typing import Dict

import networkx as nx
import numpy as np
import numpy.typing as npt

from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig
from sledge.autoencoder.preprocessing.features.graph_feature import Graph, GraphConfig
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVector
from sledge.simulation.maps.sledge_map.sledge_map_graph import get_directed_lane_graph, interpolated_lines
from sledge.simulation.maps.sledge_map.sledge_path import SledgePath


def rvae_graph_conversion(sledge_vector: SledgeVector, config: RVAEConfig, graph_config: GraphConfig) -> Graph:
    """
    Converts output vector representation of the RVAE to uniform graph for metrics.
    :param sledge_vector: vector dataclass of RVAE
    :param config: config dataclass of RVAE
    :return: Graph representation for metric.
    """

    lines = sledge_vector.lines
    assert lines.mask.ndim == 1, "Wrong number of dimensions. Assumes un-batched input."

    # Extract and interpolate valid lines
    lane_path_list, lane_poses_list = interpolated_lines(
        sledge_vector.lines, graph_config.pose_interval, config.mask_threshold
    )

    # Convert to dictionary representation and add lane ids
    lane_path_dict: Dict[str, SledgePath] = {}
    lane_poses_dict: Dict[str, npt.NDArray[np.float64]] = {}
    for lane_id, (lane_path, lane_poses) in enumerate(zip(lane_path_list, lane_poses_list)):
        lane_path_dict[str(lane_id)] = lane_path
        lane_poses_dict[str(lane_id)] = lane_poses

    # Extract directed lane graph
    G_lane = get_directed_lane_graph(
        lane_path_dict,
        config.distance_threshold,
        config.angular_threshold,
    )

    # Create graph representation
    num_lines = len(lane_poses_dict.values())
    max_poses = max([len(poses) for poses in lane_poses_dict.values()])

    lines = np.zeros((num_lines, max_poses, 3), dtype=np.float32)  # (line, points, 3)
    line_adjacency = nx.to_numpy_array(G_lane)  # (line, line)
    mask = np.zeros((num_lines, max_poses), dtype=bool)  # (line, points)

    for line_idx, lane_poses in enumerate(lane_poses_dict.values()):
        lines[line_idx, : len(lane_poses)] = lane_poses
        mask[line_idx, : len(lane_poses)] = True

    return Graph(lines, line_adjacency, mask)
