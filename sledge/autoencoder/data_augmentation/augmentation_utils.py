import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import StateSE2

from sledge.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeVector,
    SledgeVectorElement,
    LineIndex,
    AgentIndex,
    StaticObjectIndex,
)


def random_se2(scales: StateSE2) -> StateSE2:
    """
    Samples random SE(2) state space, given the scales from the input
    :param scales: scales for (x,y,θ)
    :return: random SE(2) state
    """

    noise_x = np.random.normal(loc=0, scale=scales.x)
    noise_y = np.random.normal(loc=0, scale=scales.y)
    noise_heading = np.random.normal(loc=0, scale=scales.heading)

    return StateSE2(noise_x, noise_y, noise_heading)


def transform_sledge_vector(sledge_vector: SledgeVector, origin: StateSE2) -> SledgeVector:
    """
    Transforms entire sledge vector dataclass into local coordinate from of origin
    TODO: move somewhere for general usage
    :param sledge_vector: Vector dataclass for sledge
    :param origin: SE(2) state as origin
    :return: transformed sledge vector dataclass
    """
    sledge_vector.lines.states[..., LineIndex.POINT] = transform_points_to_origin(
        sledge_vector.lines.states[..., LineIndex.POINT], origin
    )
    sledge_vector.vehicles.states[..., AgentIndex.STATE_SE2] = transform_se2_to_origin(
        sledge_vector.vehicles.states[..., AgentIndex.STATE_SE2], origin
    )
    sledge_vector.pedestrians.states[..., AgentIndex.STATE_SE2] = transform_se2_to_origin(
        sledge_vector.pedestrians.states[..., AgentIndex.STATE_SE2], origin
    )
    sledge_vector.static_objects.states[..., StaticObjectIndex.STATE_SE2] = transform_se2_to_origin(
        sledge_vector.static_objects.states[..., StaticObjectIndex.STATE_SE2], origin
    )
    sledge_vector.green_lights.states[..., LineIndex.POINT] = transform_points_to_origin(
        sledge_vector.green_lights.states[..., LineIndex.POINT], origin
    )
    sledge_vector.red_lights.states[..., LineIndex.POINT] = transform_points_to_origin(
        sledge_vector.red_lights.states[..., LineIndex.POINT], origin
    )
    return sledge_vector


def element_dropout(element: SledgeVectorElement, p: float) -> SledgeVectorElement:
    """
    Random dropout of entities in vector element dataclass
    :param element: sledge vector element dataclass (e.g. lines, vehicles)
    :param p: probability of removing entity from vector element collection
    :return: augmented vector element dataclass
    """

    num_entities = len(element.states)
    if num_entities == 0:
        return element

    dropout_mask = np.random.choice([True, False], size=num_entities, p=[1 - p, p])
    states = element.states[dropout_mask]
    mask = element.mask[dropout_mask]

    return SledgeVectorElement(states, mask)


def transform_points_to_origin(points: npt.NDArray[np.float64], origin: StateSE2) -> npt.NDArray[np.float64]:
    """
    Transforms (x,y)-array to origin frame
    TODO: move somewhere for general usage
    :param points: array with (x,y) in last axis
    :param origin: SE(2) state as origin
    :return: transformed points
    """

    if len(points) == 0:
        return points

    assert points.shape[-1] == 2
    theta = -origin.heading
    origin_array = np.array([[origin.x, origin.y, origin.heading]], dtype=np.float64)

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    points_rel = points - origin_array[..., :2]
    points_rel[..., :2] = points_rel[..., :2] @ R.T

    return points_rel


def transform_se2_to_origin(state_se2_array: npt.NDArray[np.float64], origin: StateSE2) -> npt.NDArray[np.float64]:
    """
    Transforms (x,y,θ)-array to origin frame
    TODO: move somewhere for general usage
    :param points: array with (x,y,θ) in last axis
    :param origin: SE(2) state as origin
    :return: transformed (x,y,θ)-array
    """
    if len(state_se2_array) == 0:
        return state_se2_array

    assert state_se2_array.shape[-1] == 3

    theta = -origin.heading
    origin_array = np.array([[origin.x, origin.y, origin.heading]], dtype=np.float64)

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    points_rel = state_se2_array - origin_array
    points_rel[..., :2] = points_rel[..., :2] @ R.T
    points_rel[:, 2] = normalize_angle(points_rel[:, 2])

    return points_rel
