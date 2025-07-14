# flake8: noqa
# TODO: Fix this file and remove the noqa comment.
from typing import List

import networkx as nx
import numpy as np
import numpy.typing as npt
import torch
from nuplan.common.actor_state.state_representation import StateSE2
from shapely.geometry import LineString

from sledge.autoencoder.preprocessing.features.sledge_raster_feature import SledgeRaster
from sledge.metrics.training.preprocessing.mapping.map_raster_builder import paths_to_raster
from sledge.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle
from sledge.simulation.planner.pdm_planner.utils.pdm_path import PDMPath

NUM_LINES = 1000  # just a large number here
RASTER_CONFIG = MapRasterFeatureConfig(
    frame=[64, 64],
    pixel_size=0.25,
    num_lines=NUM_LINES,  # ignore
    num_vehicles=None,  # ignore
    num_pedestrians=None,  # ignore
    num_static_objects=None,  # ignore
    num_green_lights=None,  # ignore
    num_red_lights=None,  # ignore
    line_mode="edges",
)


def get_edges_graph(G: nx.DiGraph) -> npt.NDArray[np.float32]:
    def extract_edge(G, start_node, start_end_nodes):

        lines = []
        for neighbor_idx in range(len(list(G.successors(node)))):
            chain = [start_node]
            current_node = start_node
            next_node_idx = neighbor_idx

            while True:

                neighbors = list(G.successors(current_node))

                if len(neighbors) == 0:
                    break

                next_node = neighbors[next_node_idx]
                next_node_idx = 0

                if next_node in start_end_nodes:
                    chain.append(next_node)
                    break

                if next_node in chain:  # Avoid loops
                    break

                chain.append(next_node)
                current_node = next_node

            lines.append(chain)

        return lines

    start_end_nodes = [node for node, degree in G.degree() if degree != 2]

    G_lane = nx.DiGraph()
    edges = []
    for node in start_end_nodes:
        G_lane.add_node(node, pose=G.nodes()[node]["pose"])

    for node in start_end_nodes:
        node_edges = extract_edge(G, node, start_end_nodes)
        for node_edge in node_edges:
            if len(node_edge) > 1:
                G_lane.add_edge(node_edge[0], node_edge[-1])
                edges.append(node_edge)

    lanes = []
    for edge in edges:
        line = np.array([G.nodes()[node]["pose"] for node in edge], dtype=np.float32)
        lanes.append(line)

    return G_lane, lanes


def geometry_edges_lengths(edges: List[npt.NDArray[np.float32]]):
    lengths = []
    for edge in edges:
        shapely_line = LineString(edge[:, :2])
        lengths.append(shapely_line.length)
    return lengths


def geometry_edges_orientation(edges: List[npt.NDArray[np.float32]]):
    orientations = []
    for edge in edges:
        orientation = np.mean(edge[:, 2])
        orientations.append(orientation)
    return orientations


def urban_planning_reach_and_convenience(G_edges):
    reach = []
    convenience = []

    for source in G_edges.nodes():
        # Compute shortest path lengths from this source to all reachable nodes
        lengths_dict = dict(nx.single_source_shortest_path_length(G_edges, source))
        reach.append(len(lengths_dict) - 1)
        for target, length in lengths_dict.items():
            if source != target:
                convenience.append(length)

    return reach, convenience


def get_latents(
    edges: List[npt.NDArray[np.float32]],
    encoder: torch.nn.Module,
    device: torch.device,
    num_channels: int,
) -> npt.NDArray[np.float32]:

    pdm_paths: List[PDMPath] = []
    for edge in edges:
        discrete_path: List[StateSE2] = [StateSE2(x, y, heading) for (x, y, heading) in edge]
        pdm_paths.append(PDMPath(discrete_path))

    pixel_width, pixel_height = RASTER_CONFIG.pixel_frame
    line_raster = np.zeros((pixel_width, pixel_height, num_channels), dtype=np.float32)
    line_raster[..., :2] = paths_to_raster(pdm_paths, RASTER_CONFIG, NUM_LINES)

    raster = SledgeRaster(line_raster).to_feature_tensor()
    raster = SledgeRaster.collate([raster])
    raster = raster.to_device(device)

    with torch.no_grad():
        latent = encoder.forward(raster.data)

    if num_channels == 12:
        latent = latent[0][:, :32]

    latent = latent.flatten(1, -1).squeeze(dim=0)
    latent = latent.detach().cpu().numpy()

    return latent


def get_route_length(G: nx.DiGraph) -> float:

    # 1. find starting node
    nodes = np.array([node for node, _ in G.nodes(data=True)])
    poses = np.array([data["pose"] for _, data in G.nodes(data=True)])

    # first check filter by heading angle
    mask = np.abs(normalize_angle(poses[..., 2])) < np.deg2rad(60)
    poses_masked, nodes_masked = poses[mask], nodes[mask]

    if len(poses_masked) > 1:
        # find nearest pose with low heading
        distance = (poses_masked[..., :2] ** 2).sum(-1) ** 0.5
        nearest_node = nodes_masked[np.argmin(distance)]

    else:
        distance = (poses[..., :2] ** 2).sum(-1) ** 0.5
        nearest_node = nodes[np.argmin(distance)]

    # find longest path
    longest_length, longest_path = longest_path_dls(G, nearest_node, len(G))

    if longest_length <= 1:
        return 0.0

    # create polyline
    discrete_path = [StateSE2(*G.nodes[node]["pose"]) for node in longest_path]
    path = PDMPath(discrete_path)

    # measure length of polyline and return
    return path.length


def longest_path_dls(graph, start, max_depth):
    visited = set()  # To keep track of visited nodes in the current path
    return dfs(graph, start, max_depth, visited, 0, [])


def dfs(graph, node, depth, visited, current_length, path):
    if depth == 0 or node in visited:
        return current_length, path

    visited.add(node)
    max_length = current_length
    max_path = list(path)

    for neighbor in graph[node]:
        new_length, new_path = dfs(graph, neighbor, depth - 1, visited, current_length + 1, path + [neighbor])
        if new_length > max_length:
            max_length = new_length
            max_path = new_path

    visited.remove(node)
    return max_length, max_path
