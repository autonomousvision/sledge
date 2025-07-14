from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from sledge.autoencoder.preprocessing.features.graph_feature import Graph, GraphConfig
from sledge.metrics.reconstruction.utils import (
    ReconstructionMetricConfig,
    convert_to_digraph,
    find_nodes_within_dist,
    keep_nodes_in_frame,
    node_poses,
    one_to_one_matching,
)


def graph_reconstruction_metrics(
    target_graph: Graph, prediction_graph: Graph, graph_config: GraphConfig
) -> Dict[str, float]:

    # TODO: v0.2 remove
    metric_config = ReconstructionMetricConfig()

    # convert to digraph
    G_target = convert_to_digraph(target_graph)
    G_prediction = convert_to_digraph(prediction_graph)

    # remove invalid nodes
    G_target = keep_nodes_in_frame(G_target, graph_config)
    G_prediction = keep_nodes_in_frame(G_prediction, graph_config)

    # calculate metrics
    result: Dict[str, float] = {}
    matched_pairs = one_to_one_matching(G_target, G_prediction, metric_config)

    # 1. Geo Metrics
    geo_metric_functions = [
        geo_matching,
        geo_chamfer,
        geo_linear_displacement,
        geo_euclidean,
    ]
    for geo_metric_function in geo_metric_functions:
        geo_function_result = geo_metric_function(G_target, G_prediction, matched_pairs)
        result.update(geo_function_result)

    topo_results = topo_metrics(G_target, G_prediction, matched_pairs, metric_config)
    result.update(topo_results)

    return result


def geo_matching(G_a: nx.DiGraph, G_b: nx.DiGraph, matched_pairs: List[Tuple[int, int]]) -> Dict[str, float]:

    precision = len(matched_pairs) / len(G_a.nodes)
    recall = len(matched_pairs) / len(G_b.nodes)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0.0 else 0.0

    result = {"geo_precision": precision, "geo_recall": recall, "geo_f1": f1}

    return result


def geo_chamfer(G_a: nx.DiGraph, G_b: nx.DiGraph, matched_pairs: List[Tuple[int, int]]) -> Dict[str, float]:

    points_a, points_b = node_poses(G_a)[..., :2], node_poses(G_b)[..., :2]

    # Calculate the squared Euclidean distances between all pairs of points
    distances = ((points_a[:, None, :] - points_b[None, :, :]) ** 2).sum(axis=-1)

    # For each point in set1, find the squared distance to its nearest neighbor in set2
    min_distances_set1 = distances.min(axis=1)

    # For each point in set2, find the squared distance to its nearest neighbor in set1
    min_distances_set2 = distances.min(axis=0)

    # Calculate the Chamfer distance
    chamfer_distance = min_distances_set1.mean() + min_distances_set2.mean()

    return {"geo_chamfer": chamfer_distance}


def geo_euclidean(G_a: nx.DiGraph, G_b: nx.DiGraph, matched_pairs: List[Tuple[int, int]]) -> Dict[str, float]:

    a_indices = [i for i, j in matched_pairs]
    b_indices = [j for i, j in matched_pairs]

    points_a, points_b = node_poses(G_a, a_indices)[..., :2], node_poses(G_b, b_indices)[..., :2]
    euclidean_distance = (((points_a - points_b) ** 2).sum(-1) ** 0.5).mean()

    return {"geo_euclidean": euclidean_distance}


def geo_linear_displacement(G_a: nx.DiGraph, G_b: nx.DiGraph, matched_pairs: List[Tuple[int, int]]) -> Dict[str, float]:

    a_indices = [i for i, j in matched_pairs]
    b_indices = [j for i, j in matched_pairs]

    poses_a, poses_b = node_poses(G_a, a_indices), node_poses(G_b, b_indices)

    def linear_displacements(pose_a, pose_b):

        longitudinal, lateral = 0.0, 0.0
        if len(pose_a) > 0 and len(pose_b) > 0:
            # Convert heading to unit vector
            v = np.zeros((len(pose_a), 2), dtype=np.float64)

            v[..., 0] = np.cos(pose_a[..., 2])
            v[..., 1] = np.sin(pose_a[..., 2])

            # Vector from Pose A to Pose B
            ab = pose_b[..., :2] - pose_a[..., :2]

            longitudinal = np.einsum("bn, bn-> b", ab, v)

            projection_point = pose_a[..., :2] + longitudinal[..., None] * v
            lateral = np.linalg.norm(pose_b[..., :2] - projection_point, axis=-1)

        return longitudinal, lateral

    longitudinal, lateral = linear_displacements(poses_a, poses_b)
    return {"geo_linear_lat": np.mean(np.abs(lateral)), "geo_linear_long": np.mean(np.abs(longitudinal))}


def topo_metrics(
    G_a: nx.DiGraph, G_b: nx.DiGraph, matched_pairs: List[Tuple[int, int]], metric_config: ReconstructionMetricConfig
) -> Dict[str, float]:

    topo_precision = []
    topo_recall = []
    topo_chamfer = []
    topo_euclidean = []
    topo_linear_lat = []
    topo_linear_long = []

    geo_metrics_dict = {
        "geo_matching": geo_matching,
        "geo_chamfer": geo_chamfer,
        "geo_euclidean": geo_euclidean,
        "geo_linear": geo_linear_displacement,
    }
    for i, j in matched_pairs[:: metric_config.topo_subsampling]:
        S_a = G_a.subgraph(find_nodes_within_dist(G_a, i, radius=metric_config.radius))
        S_b = G_b.subgraph(find_nodes_within_dist(G_b, j, radius=metric_config.radius))

        S_matched_pairs = one_to_one_matching(S_a, S_b, metric_config)

        geo_results = {}
        for geo_name, geo_function in geo_metrics_dict.items():
            geo_function_result = geo_function(S_a, S_b, S_matched_pairs)
            geo_results.update(geo_function_result)

        topo_precision.append(geo_results["geo_precision"])
        topo_recall.append(geo_results["geo_recall"])
        topo_chamfer.append(geo_results["geo_chamfer"])
        topo_euclidean.append(geo_results["geo_euclidean"])
        topo_linear_long.append(geo_results["geo_linear_long"])
        topo_linear_lat.append(geo_results["geo_linear_lat"])

    topo_results = {}

    topo_results["topo_precision"] = np.mean(topo_precision)
    topo_results["topo_recall"] = np.mean(topo_recall)
    topo_results["topo_f1"] = (
        2
        * topo_results["topo_precision"]
        * topo_results["topo_recall"]
        / (topo_results["topo_precision"] + topo_results["topo_recall"])
        if topo_results["topo_precision"] + topo_results["topo_recall"] != 0.0
        else 0.0
    )

    topo_results["topo_chamfer"] = np.mean(topo_chamfer)
    topo_results["topo_euclidean"] = np.mean(topo_euclidean)
    topo_results["topo_linear_long"] = np.mean(topo_linear_long)
    topo_results["topo_linear_lat"] = np.mean(topo_linear_lat)

    return topo_results
