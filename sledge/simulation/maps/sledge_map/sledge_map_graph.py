from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt

import networkx as nx

from shapely.geometry import Polygon, LineString, Point
from shapely.geometry.base import CAP_STYLE

from nuplan.common.maps.maps_datatypes import TrafficLightStatusType

from sledge.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap
from sledge.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle
from sledge.simulation.maps.sledge_map.sledge_path import SledgePath
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeVector,
    SledgeVectorElement,
    SledgeVectorElementType,
)


@dataclass
class SledgeTrafficLight:
    """Traffic light object for sledge map graph interface."""

    poses: npt.NDArray[np.float64]
    status: TrafficLightStatusType
    polygon: Polygon


@dataclass
class SledgeMapGraph:
    """General map graph interface, used for sledge map."""

    baseline_paths_dict: Dict[str, npt.NDArray[np.float64]]
    polygon_dict: Dict[str, Polygon]
    directed_lane_graph: nx.DiGraph
    traffic_light_dict: Dict[str, SledgeTrafficLight]
    occupancy_map: PDMOccupancyMap = None

    def __post_init__(self):
        self.occupancy_map = PDMOccupancyMap(list(self.polygon_dict.keys()), list(self.polygon_dict.values()))


def construct_sledge_map_graph(
    sledge_vector: SledgeVector,
    baseline_path_interval: float = 0.5,
    mask_thresh: float = 0.3,
    distance_thresh: float = 1.5,
    angular_thresh: float = np.deg2rad(60),
    polygon_width: float = 3.0,
) -> SledgeMapGraph:
    """
    Constructs sledge map graph interface.
    TODO: Refactor & add parameters to config
    :param sledge_vector: dataclass of vector representation in sledge
    :param baseline_path_interval: interval of baseline path poses [m], defaults to 0.5
    :param mask_thresh: threshold for probability of existence, defaults to 0.3
    :param distance_thresh: distance threshold of lane node connection, defaults to 1.5
    :param angular_thresh: angular threshold of lane node connection, defaults to np.deg2rad(60)
    :param polygon_width: fixed width of lane polygons, defaults to 3.0
    :return: sledge map graph dataclass
    """

    # Create a directed graph objects
    lane_path_list, lane_poses_list = interpolated_lines(sledge_vector.lines, baseline_path_interval, mask_thresh)

    lane_path_dict: Dict[str, SledgePath] = {}
    baseline_paths_dict: Dict[str, npt.NDArray[np.float64]] = {}
    polygon_dict: Dict[str, Polygon] = {}

    for lane_id, (lane_path, lane_poses) in enumerate(zip(lane_path_list, lane_poses_list)):
        lane_path_dict[str(lane_id)] = lane_path
        baseline_paths_dict[str(lane_id)] = lane_poses
        polygon_dict[str(lane_id)] = LineString(lane_poses[:, :2]).buffer(polygon_width, cap_style=CAP_STYLE.square)

    directed_lane_graph = get_directed_lane_graph(lane_path_dict, distance_thresh, angular_thresh)
    traffic_light_dict = get_traffic_light_dict(sledge_vector, lane_path_dict, baseline_path_interval, mask_thresh)
    return SledgeMapGraph(baseline_paths_dict, polygon_dict, directed_lane_graph, traffic_light_dict)


def get_directed_lane_graph(
    lane_path_dict: Dict[str, SledgePath], distance_thresh: float = 1.5, angular_thresh: float = np.deg2rad(60)
) -> nx.DiGraph:
    """
    Extract lane graph from lane path dictionary.
    :param lane_path_dict: dictionary of id's and paths as keys and values
    :param distance_thresh: distance threshold of lane node connection, defaults to 1.5
    :param angular_thresh: angular threshold of lane node connection, defaults to np.deg2rad(60)
    :return: networkx directed lane graph
    """

    # 1. add lane id's as nodes
    directed_lane_graph = nx.DiGraph()
    for lane_id in lane_path_dict.keys():
        directed_lane_graph.add_node(lane_id)

    # 2. add lane connections based on thresholds
    start_poses = np.array([path.states_se2_array[0] for path in lane_path_dict.values()])
    end_poses = np.array([path.states_se2_array[-1] for path in lane_path_dict.values()])

    end_start_distance = np.linalg.norm(end_poses[:, None, :2] - start_poses[None, :, :2], axis=-1)
    end_start_angular_error = np.abs(normalize_angle(end_poses[:, None, 2] - start_poses[None, :, 2]))

    lane_ids = list(lane_path_dict.keys())
    connections = np.where(
        np.logical_and(
            end_start_distance < distance_thresh,
            end_start_angular_error < angular_thresh,
        )
    )

    for lane_i, lane_j in zip(*connections):
        lane_i_id, lane_j_id = lane_ids[lane_i], lane_ids[lane_j]
        if lane_path_dict[lane_i_id].length > distance_thresh or lane_path_dict[lane_j_id].length > distance_thresh:
            directed_lane_graph.add_edge(lane_i_id, lane_j_id)

    return directed_lane_graph


def get_traffic_light_dict(
    sledge_vector: SledgeVector,
    lane_path_dict: Dict[str, SledgePath],
    baseline_path_interval: float = 0.5,
    mask_thresh: float = 0.3,
    polygon_width: float = 3.0,
) -> Dict[str, SledgeTrafficLight]:
    """
    Extracts dictionary of traffic light dataclasses.
    :param sledge_vector: sledge vector dataclass of current scene
    :param lane_path_dict: dictionary of id's and paths as keys and values
    :param baseline_path_interval: interval of baseline path poses [m], defaults to 0.5
    :param mask_thresh: threshold for probability of existence, defaults to 0.3
    :param polygon_width: fixed width of lane polygons, defaults to 3.0
    :return: dictionary of lane id's and traffic light dataclass
    """

    traffic_light_dict: Dict[str, Tuple[TrafficLightStatusType, npt.NDArray[np.float64]]] = {}

    # extract green + red traffic lights
    _, green_lights = interpolated_lines(sledge_vector.green_lights, baseline_path_interval, mask_thresh)
    _, red_lights = interpolated_lines(sledge_vector.red_lights, baseline_path_interval, mask_thresh)
    status_types = len(green_lights) * [TrafficLightStatusType.GREEN] + len(red_lights) * [TrafficLightStatusType.RED]

    for traffic_light_line, traffic_light_type in zip(green_lights + red_lights, status_types):
        min_distance, min_lane_id = None, None
        for lane_id, lane_path in lane_path_dict.items():
            distances = []
            for pose in traffic_light_line:
                p = Point(pose[:2])
                distances.append(p.distance(lane_path.linestring))

            average_distance = np.mean(distances)
            if (min_distance is None) or (min_distance > average_distance):
                min_distance = average_distance
                min_lane_id = lane_id

        start_distance = lane_path_dict[min_lane_id].project(Point(traffic_light_line[0, :2]))
        end_distance = lane_path_dict[min_lane_id].project(Point(traffic_light_line[-1, :2]))

        start_distance = np.clip(start_distance, 0, lane_path_dict[min_lane_id].length)
        end_distance = np.clip(end_distance, 0, lane_path_dict[min_lane_id].length)
        num_samples = int((end_distance - start_distance) // baseline_path_interval)

        if num_samples > 1:
            distances = np.linspace(start_distance, end_distance, num=num_samples, endpoint=True)
            traffic_light_poses = lane_path_dict[min_lane_id].interpolate(distances)
            traffic_light_polygon = LineString(traffic_light_poses[:, :2]).buffer(
                polygon_width,
                cap_style=CAP_STYLE.square,
            )
            traffic_light_dict[min_lane_id] = SledgeTrafficLight(
                traffic_light_poses,
                traffic_light_type,
                traffic_light_polygon,
            )

    return traffic_light_dict


def interpolated_lines(
    sledge_vector_element: SledgeVectorElement,
    baseline_path_interval: float = 0.5,
    mask_thresh: float = 0.3,
) -> Tuple[List[SledgePath], List[npt.NDArray[np.float64]]]:
    """
    Extract interpolatable paths and poses from vector element dataclass
    :param sledge_vector_element: vector element dataclass
    :param baseline_path_interval: interval of baseline path poses [m], defaults to 0.5
    :param mask_thresh: threshold for probability of existence, defaults to 0.3
    :return: list of interpolatable paths and poses as numpy array
    """
    assert sledge_vector_element.get_element_type() == SledgeVectorElementType.LINE

    path_list: List[SledgePath] = []
    poses_list: List[npt.NDArray[np.float64]] = []

    for state, p in zip(sledge_vector_element.states, sledge_vector_element.mask):
        valid = p if (type(p) is np.bool_) else p > mask_thresh
        if valid:
            path = SledgePath(state)
            distances = np.arange(0, path.length + baseline_path_interval, step=baseline_path_interval)
            poses = path.interpolate(distances)
            if len(poses) > 1:
                path_list.append(path)
                poses_list.append(poses)

    return path_list, poses_list
