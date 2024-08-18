# TODO: Refactor
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import networkx as nx

from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses

from sledge.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from sledge.simulation.planner.pdm_planner.utils.pdm_array_representation import array_to_states_se2
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVectorElement


def get_lane_objects(ego_state: EgoState, map_api: AbstractMap, radius: float) -> List[LaneGraphEdgeMapObject]:
    """
    Load all lanes in radius from map api.
    :param ego_state: object of ego vehicle state in nuPlan
    :param map_api: object of map in nuPlan
    :param radius: radius of lanes to load
    :return: list of lane objects
    """
    layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    nearby_map_objects = map_api.get_proximal_map_objects(
        layers=layers,
        point=ego_state.center.point,
        radius=radius,
    )
    nearby_lanes = []
    for layer in layers:
        nearby_lanes.extend(nearby_map_objects[layer])
    return nearby_lanes


def get_lane_graph(
    ego_state: EgoState, map_api: AbstractMap, radius: float
) -> Tuple[nx.DiGraph, Dict[str, LaneGraphEdgeMapObject]]:
    """
    Extract lane graph from map api as networkx directed graph
    :param ego_state: object of ego vehicle state in nuPlan
    :param map_api: object of map in nuPlan
    :param radius: radius of lanes to load
    :return: tuple of directed lane graph and dictionary of id's and lanes
    """

    lane_objects = get_lane_objects(ego_state, map_api, radius)
    lane_dict = {lane.id: lane for lane in lane_objects}

    G = nx.DiGraph()
    for lane_id, lane in lane_dict.items():
        G.add_node(lane_id)

    for lane_id, lane in lane_dict.items():
        for outgoing_lane in lane.outgoing_edges:
            outgoing_lane_id = outgoing_lane.id
            if outgoing_lane_id in lane_dict.keys():
                G.add_edge(lane_id, outgoing_lane_id)

    return G, lane_dict


def find_path_to_targets(graph: nx.DiGraph, start: str, targets: List[str]) -> List[str]:
    """
    Find path from start to any target node (without crossing another target node).
    :param graph: directed graph in networkx
    :param start: id of start node
    :param targets: list of id's as target nodes
    :return: path from start to closest target
    """

    def _find_path_to_target(graph: nx.DiGraph, start: str, target: str) -> List[str]:
        """Helper function to return path between start and target, if exists."""
        try:
            path = nx.shortest_path(graph, start, target)
            return path
        except nx.NetworkXNoPath:
            return []

    for target in targets:
        path_to_target = _find_path_to_target(graph, start, target)
        if path_to_target:
            # check if other target node was crossed)
            if len(set(path_to_target) & set(targets)) == 1:
                return path_to_target

    return []


def get_discrete_path(lane_dict: Dict[str, LaneGraphEdgeMapObject], lane_ids: List[str]) -> List[StateSE2]:
    """
    Helper function to convert lane path to discrete sequence of SE(2) samples.
    :param lane_dict: dictionary of lane id and lane objects
    :param lane_ids: list of lane id's in path
    :return: list of SE(2) objects
    """
    discrete_path = []
    for lane_id in lane_ids:
        discrete_path.extend(lane_dict[lane_id].baseline_path.discrete_path)
    return discrete_path


def get_edge_graph(ego_state: EgoState, map_api: AbstractMap, radius: float) -> List[PDMPath]:
    """
    Lane graph preprocessing method, as described in the supplementary material.
    Simplifies the formulation of a lane, by summarizing consecutive nodes,
    that don't have multiple ingoing our outgoing lanes.
    :param ego_state: object of ego vehicle state in nuPlan
    :param map_api: object of map in nuPlan
    :param radius: radius around ego for lanes to preprocess
    :return: list of summarized lanes, as interpolatable paths
    """

    G_lane, lane_dict = get_lane_graph(ego_state, map_api, radius)

    # collect start and end lanes of edges
    start_dict, stop_dict = {}, {}
    for lane_id, lane in lane_dict.items():
        in_degree = G_lane.in_degree(lane_id)
        out_degree = G_lane.out_degree(lane_id)

        if in_degree == 0 or in_degree > 1:
            start_dict[lane_id] = lane

        if out_degree == 0 or out_degree > 1:
            stop_dict[lane_id] = lane

    for lane_id, lane in lane_dict.items():
        is_predecessor_stop = len(set(G_lane.predecessors(lane_id)) & set(stop_dict.keys())) > 0
        if is_predecessor_stop:
            start_dict[lane_id] = lane

        is_successors_start = len(set(G_lane.successors(lane_id)) & set(start_dict.keys())) > 0
        if is_successors_start:
            stop_dict[lane_id] = lane

    paths_list: List[PDMPath] = []

    # add nodes to line graph
    for start_lane_id, start_lane in start_dict.items():
        path_to_targets = find_path_to_targets(G_lane, start_lane_id, list(stop_dict.keys()))
        if path_to_targets:
            discrete_path = get_discrete_path(lane_dict, path_to_targets)
            local_discrete_path: npt.NDArray[np.float32] = convert_absolute_to_relative_poses(
                ego_state.center, discrete_path
            )
            within_radius = np.linalg.norm(local_discrete_path[..., :2], axis=-1) <= radius
            local_discrete_path = local_discrete_path[within_radius]

            if len(local_discrete_path) < 2:
                continue

            paths_list.append(PDMPath(list(array_to_states_se2(local_discrete_path))))

    return paths_list


def get_traffic_light_discrete_paths(
    traffic_light_data: TrafficLightStatusData, traffic_light_status: TrafficLightStatusType, map_api: AbstractMap
) -> List[List[StateSE2]]:
    """
    Load traffic lights lines from map api.
    :param traffic_light_data: dataclass of current traffic lights
    :param traffic_light_status: status type to extract (ie. red or green)
    :param map_api: object of map in nuPlan
    :return: list of paths, as SE(2) sequence
    """
    traffic_light_lanes: List[LaneGraphEdgeMapObject] = []
    for data in traffic_light_data:
        if data.status == traffic_light_status:
            lane_connector = map_api.get_map_object(str(data.lane_connector_id), SemanticMapLayer.LANE_CONNECTOR)
            traffic_light_lanes.append(lane_connector)

    discrete_paths = []
    for lane in traffic_light_lanes:
        discrete_paths.append(lane.baseline_path.discrete_path)

    return discrete_paths


def compute_path_features(interpolatable_paths: List[PDMPath], pose_interval: float) -> SledgeVectorElement:
    """
    Compute raw vector element of lines.
    :param interpolatable_paths: lines as interpolatable SE(2) paths
    :param pose_interval: interval of poses to sample in meter
    :return: raw sledge vector element for feature caching.
    """
    line_poses_list: List[npt.NDArray[np.float32]] = []
    for edge_path in interpolatable_paths:
        limit = (
            edge_path.length + pose_interval
            if edge_path.length % pose_interval > pose_interval / 2
            else edge_path.length
        )
        distances = np.arange(0, limit, step=pose_interval)
        line_poses = edge_path.interpolate(distances, as_array=True)
        line_poses_list.append(line_poses)

    if len(line_poses_list) > 0:
        num_lines = len(line_poses_list)
        max_poses = max([len(poses) for poses in line_poses_list])

        lines = np.zeros((num_lines, max_poses, 3), dtype=np.float32)  # (line, points, 3)
        mask = np.zeros((num_lines, max_poses), dtype=bool)  # (line, points)

        for line_idx, line_poses in enumerate(line_poses_list):
            lines[line_idx, : len(line_poses)] = line_poses
            mask[line_idx, : len(line_poses)] = True
    else:

        lines = np.array([], dtype=np.float32)  # (line, points, 3)
        mask = np.array([], dtype=bool)  # (line, points)

    return SledgeVectorElement(lines, mask)


def compute_line_features(
    ego_state: EgoState, map_api: AbstractMap, radius: float, pose_interval: float
) -> SledgeVectorElement:
    """
    Compute raw vector element of lines (ie. lanes in map).
    :param ego_state: object of ego vehicle in nuPlan
    :param map_api: object of map in nuPlan
    :param radius: radius around ego for lanes to preprocess
    :param pose_interval: interval of poses to sample in meter
    :return: raw sledge vector element for feature caching.
    """
    interpolatable_paths = get_edge_graph(ego_state, map_api, radius)
    return compute_path_features(interpolatable_paths, pose_interval)


def compute_traffic_light_features(
    ego_state: EgoState,
    map_api: AbstractMap,
    traffic_light_data: List[TrafficLightStatusData],
    traffic_light_status: TrafficLightStatusType,
    radius: float,
    pose_interval: float,
) -> SledgeVectorElement:
    """
    Compute raw vector element of traffic lights, given status type.
    :param ego_state: object of ego vehicle in nuPlan
    :param map_api: object of map in nuPlan
    :param traffic_light_data: dataclass of traffic lights in nuPlan
    :param traffic_light_status: status type to preprocess
    :param radius: radius around ego for lanes to preprocess
    :param pose_interval: interval of poses to sample in meter
    :return: raw sledge vector element for feature caching
    """
    discrete_paths = get_traffic_light_discrete_paths(traffic_light_data, traffic_light_status, map_api)
    interpolatable_paths: List[PDMPath] = []
    for discrete_path in discrete_paths:
        local_discrete_path: npt.NDArray[np.float32] = convert_absolute_to_relative_poses(
            ego_state.center, discrete_path
        )
        within_radius = np.linalg.norm(local_discrete_path[..., :2], axis=-1) <= radius
        local_discrete_path = local_discrete_path[within_radius]
        if len(local_discrete_path) < 2:
            continue
        interpolatable_paths.append(PDMPath(list(array_to_states_se2(local_discrete_path))))

    return compute_path_features(interpolatable_paths, pose_interval)
