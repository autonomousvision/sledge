from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import networkx as nx

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObjects, TrackedObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType, AGENT_TYPES
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateVector2D
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from sledge.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from sledge.simulation.maps.sledge_map.sledge_map import SledgeMap
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVector, SledgeVectorElement
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeVectorElementType,
    AgentIndex,
    StaticObjectIndex,
    BoundingBoxIndex,
)

# TODO: add to some config
LABEL_THRESH = 0.3
OBJECT_HEIGHT = 1.0  # placeholder


def sledge_vector_to_detection_tracks(
    sledge_vector: SledgeVector,
    timestamp_us: int,
) -> DetectionsTracks:
    """
    Converts sledge vector dataclass into detection tracks dataclass for simulation pipeline.
    :param sledge_vector: sledge vector dataclass
    :param timestamp_us: time step as integer [μs]
    :return: detection track dataclass in nuPlan.
    """
    tracked_objects: List[TrackedObject] = []

    # 1. vehicles
    vehicles = sledge_element_to_tracked_objects(
        sledge_vector.vehicles,
        timestamp_us,
        TrackedObjectType.VEHICLE,
    )
    tracked_objects.extend(vehicles)

    # 2. pedestrians
    pedestrians = sledge_element_to_tracked_objects(
        sledge_vector.pedestrians,
        timestamp_us,
        TrackedObjectType.PEDESTRIAN,
    )
    tracked_objects.extend(pedestrians)

    # 3. static objects
    static_objects = sledge_element_to_tracked_objects(
        sledge_vector.static_objects,
        timestamp_us,
        TrackedObjectType.GENERIC_OBJECT,
    )
    tracked_objects.extend(static_objects)

    return DetectionsTracks(TrackedObjects(tracked_objects))


def sledge_element_to_tracked_objects(
    sledge_vector_element: SledgeVectorElement, timestamp_us: int, tracked_object_type: TrackedObjectType
) -> List[TrackedObject]:
    """
    Collects all valid bounding box entities from vector element dataclass.
    :param sledge_vector_element: vector element dataclass of bounding boxes.
    :param timestamp_us: time step as integer [μs]
    :param tracked_object_type: tracked object enum
    :return: list of valid tracked objects
    """
    assert sledge_vector_element.get_element_type() in [SledgeVectorElementType.AGENT, SledgeVectorElementType.STATIC]

    tracked_objects: List[TrackedObject] = []
    for element_idx, (state, mask) in enumerate(zip(sledge_vector_element.states, sledge_vector_element.mask)):
        invalid = not mask if type(mask) is np.bool_ else mask < LABEL_THRESH
        if invalid:
            continue
        tracked_object = state_array_to_tracked_object(state, timestamp_us, tracked_object_type, str(element_idx))
        tracked_objects.append(tracked_object)

    return tracked_objects


def state_array_to_tracked_object(
    state: npt.NDArray[np.float32], timestamp_us: int, tracked_object_type: TrackedObjectType, token: str
) -> TrackedObject:
    """
    Converts state array of bounding box element to tracked object.
    :param state: numpy array, containing information about the bounding box state
    :param timestamp_us: time step as integer [μs]
    :param tracked_object_type: tracked object enum
    :param token: enumerator token of object
    :return: tracked object of bounding box
    """
    assert state.ndim == 1, f"Expected state array to have one dimension, but got {state.ndim}"
    assert state.shape[-1] in [
        AgentIndex.size(),
        StaticObjectIndex.size(),
    ], f"Invalid state array size of {state.shape[-1]}"

    is_agent = tracked_object_type in AGENT_TYPES
    object_index: BoundingBoxIndex = AgentIndex if is_agent else StaticObjectIndex

    # 1. OrientedBox
    center = StateSE2(*state[object_index.STATE_SE2])
    oriented_box = OrientedBox(
        center,
        state[object_index.LENGTH],
        state[object_index.WIDTH],
        OBJECT_HEIGHT,
    )

    # 2. SceneObjectMetadata
    track_token = f"{tracked_object_type.value}_{token}"
    metadata = SceneObjectMetadata(
        timestamp_us,
        token=track_token,
        track_id=None,
        track_token=track_token,
    )  # NOTE: assume token is equal to track_token

    if is_agent:
        # 3. StateVector2D
        velocity = StateVector2D(*get_agent_dxy(state))
        return Agent(tracked_object_type, oriented_box, velocity, metadata)

    return StaticObject(tracked_object_type, oriented_box, metadata)


def interpolate_state_se2(
    state_se2: StateSE2, trajectory_sampling: TrajectorySampling, velocity: float
) -> List[StateSE2]:
    """
    Projects poses linearly, baseline on trajectory sampling and velocity.
    :param state_se2: pose to interpolate
    :param trajectory_sampling: dataclass for trajectory details
    :param velocity: absolute velocity [m/s]
    :return: list of interpolated poses
    """
    time_s = np.arange(0.0, trajectory_sampling.time_horizon, trajectory_sampling.interval_length)
    interpolated_states = [project_state_se2(state_se2, delta_t, velocity) for delta_t in time_s]
    return interpolated_states


def project_state_se2(state_se2: StateSE2, delta_t: float, velocity: float) -> StateSE2:
    """
    Linearly projects a pose, based on time interval and velocity.
    :param state_se2: pose object
    :param delta_t: time interval for projection
    :param velocity: absolute velocity [m/s]
    :return: projected poses along heading angle
    """
    x, y, heading = state_se2.serialize()
    new_x = x + velocity * delta_t * np.cos(heading)
    new_y = y + velocity * delta_t * np.sin(heading)
    return StateSE2(new_x, new_y, heading)


def get_agent_dxy(states: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Convert (x,y) velocity of agent bounding box, based on origin.
    :param states: numpy array, containing information about the bounding box state
    :return: numpy array of (x,y) velocity in [m/s]
    """
    assert states.shape[-1] == AgentIndex.size()
    headings = states[..., AgentIndex.HEADING]
    velocities = states[..., AgentIndex.VELOCITY]
    dxy = np.array(
        [
            np.cos(headings) * velocities,
            np.sin(headings) * velocities,
        ],
        dtype=np.float64,
    ).T
    return dxy


def project_sledge_vector(sledge_vector: SledgeVector, delta_t: float) -> SledgeVector:
    """
    Projects agents (vehicles, pedestrians) with constant velocities in vector dataclass.
    :param sledge_vector: sledge vector dataclass
    :param delta_t: time interval for projection in [m/s]
    :return: projected vector dataclass
    """
    if len(sledge_vector.vehicles.states) > 0:
        projected_vehicles = project_agents(sledge_vector.vehicles, delta_t)
    else:
        projected_vehicles = sledge_vector.vehicles

    if len(sledge_vector.pedestrians.states) > 0:
        projected_pedestrians = project_agents(sledge_vector.pedestrians, delta_t)
    else:
        projected_pedestrians = sledge_vector.pedestrians

    return SledgeVector(
        lines=sledge_vector.lines,
        vehicles=projected_vehicles,
        pedestrians=projected_pedestrians,
        static_objects=sledge_vector.static_objects,
        green_lights=sledge_vector.green_lights,
        red_lights=sledge_vector.red_lights,
        ego=sledge_vector.ego,
    )


def project_agents(sledge_vector_element: SledgeVectorElement, delta_t: float) -> SledgeVectorElement:
    """
    Project sledge vector element based on constant velocity and heading angle.
    :param sledge_vector_element: sledge vector dataclass of entity
    :param delta_t: time interval for projection in [m/s]
    :return: projected sledge vector element
    """
    assert sledge_vector_element.get_element_type() == SledgeVectorElementType.AGENT
    states, mask = sledge_vector_element.states, sledge_vector_element.mask
    projected_states = states.copy()
    projected_states[..., AgentIndex.POINT] += delta_t * get_agent_dxy(states)
    return SledgeVectorElement(projected_states, mask)


def sample_future_indices(
    future_sampling: TrajectorySampling,
    iteration: int,
    time_horizon: float,
    num_samples: Optional[int],
) -> List[int]:
    """
    Helper to sample iterations indices during simulation.
    :param future_sampling: future sampling dataclass.
    :param iteration: starting iteration index
    :param time_horizon: future time to sample indices for [s]
    :param num_samples: optional number of samples to return
    :raises ValueError: invalid input arguments
    :return: list of iteration indices
    """
    time_interval = future_sampling.interval_length
    if time_horizon <= 0.0 or time_interval <= 0.0 or time_horizon < time_interval:
        raise ValueError(
            f"Time horizon {time_horizon} must be greater or equal than target time interval {time_interval}"
            " and both must be positive."
        )

    num_samples = num_samples if num_samples else int(time_horizon / time_interval)
    num_intervals = int(time_horizon / time_interval) + 1
    step_size = num_intervals // num_samples
    time_idcs = np.arange(iteration, num_intervals, step_size)
    return list(time_idcs)


def get_route(map_api: SledgeMap) -> Tuple[List[str], PDMPath]:
    """
    Heuristic to find route to drive along during simulation.
    TODO: Refactor & Update
    :param map_api: map interface in sledge lane & agent simulation
    :return: tuple of roadblock ids and interpolatable path
    """

    # 1. find starting lane
    baseline_paths_dict = map_api.sledge_map_graph.baseline_paths_dict
    lane_distances = [np.linalg.norm(poses[..., :2], axis=-1).min(-1) for poses in baseline_paths_dict.values()]
    current_node = str(np.argmin(lane_distances))

    # 2. fine lane sinks (ending lanes) and check if path exists
    graph = map_api.sledge_map_graph.directed_lane_graph
    sink_nodes = [node for node in graph.nodes() if graph.out_degree(node) == 0]

    available_paths = []
    for sink_node in sink_nodes:
        try:
            path_to_sink = nx.shortest_path(graph, source=current_node, target=sink_node)
        except nx.NetworkXNoPath:
            continue
        available_paths.append(path_to_sink)

    if len(available_paths) == 0:
        available_paths = [current_node]

    lengths = [len(path) for path in available_paths]

    # TODO: add different path strategies
    route_roadblock_ids = available_paths[np.argmax(lengths)]
    discrete_route_path: List[StateSE2] = []

    # NOTE: Slight abuse, because roadblock and lanes have same id's in SledgeMap
    for route_roadblock_id in route_roadblock_ids:
        lane = map_api.get_map_object(route_roadblock_id, SemanticMapLayer.LANE)
        discrete_route_path.extend(lane.baseline_path.discrete_path)

    route_path = PDMPath(discrete_route_path)

    return route_roadblock_ids, route_path
