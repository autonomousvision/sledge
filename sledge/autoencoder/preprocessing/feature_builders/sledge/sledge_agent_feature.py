from typing import List, Optional

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

from sledge.simulation.planner.pdm_planner.utils.pdm_geometry_utils import convert_absolute_to_relative_se2_array
from sledge.simulation.planner.pdm_planner.utils.pdm_array_representation import state_se2_to_array
from sledge.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap
from sledge.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
from sledge.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeVectorElement,
    StaticObjectIndex,
    AgentIndex,
    EgoIndex,
)


def compute_ego_features(ego_state: EgoState) -> SledgeVectorElement:
    """
    Compute raw sledge vector features for ego agents
    :param ego_state: object of ego vehicle state in nuPlan
    :return: sledge vector element of raw ego attributes.
    """

    state_array = ego_state_to_state_array(ego_state)
    ego_states = np.zeros((EgoIndex.size()), dtype=np.float64)
    ego_mask = np.ones((1), dtype=bool)  # dummy value
    ego_states[EgoIndex.VELOCITY_2D] = state_array[StateIndex.VELOCITY_2D]
    ego_states[EgoIndex.ACCELERATION_2D] = state_array[StateIndex.ACCELERATION_2D]

    return SledgeVectorElement(ego_states, ego_mask)


# TODO: Refactor
def compute_agent_features(
    ego_state: EgoState,
    detections: DetectionsTracks,
    agent_type: TrackedObjectType,
    radius: float,
    drivable_area_map: Optional[PDMOccupancyMap] = None,
) -> SledgeVectorElement:
    """
    Computes raw sledge vector features for agents (ie. vehicles, pedestrians)
    :param ego_state: object of ego vehicle state in nuPlan
    :param detections: dataclass for detected objects in nuPlan
    :param agent_type: enum of agent type (ie. vehicles, pedestrians)
    :param radius: radius around the ego vehicle to extract objects
    :param drivable_area_map: drivable area map for filtering if provided, defaults to None
    :return: raw sledge vector element of agent_type
    """

    tracked_objects = detections.tracked_objects
    agents_list: List[Agent] = tracked_objects.get_tracked_objects_of_type(agent_type)

    agents_states_list: List[npt.NDArray[np.float64]] = []
    for agent in agents_list:
        agent_states_ = np.zeros(AgentIndex.size(), dtype=np.float64)
        agent_states_[AgentIndex.STATE_SE2] = state_se2_to_array(agent.center)
        agent_states_[AgentIndex.WIDTH] = agent.box.width
        agent_states_[AgentIndex.LENGTH] = agent.box.length
        agent_states_[AgentIndex.VELOCITY] = agent.velocity.magnitude()
        agents_states_list.append(agent_states_)
    agents_states_all = np.array(agents_states_list)

    # convert to local coords and filter out of box
    if len(agents_states_all) > 0:
        if drivable_area_map is not None:
            in_drivable_area = drivable_area_map.points_in_polygons(agents_states_all[..., AgentIndex.POINT]).any(
                axis=0
            )
            agents_states_all = agents_states_all[in_drivable_area]

        # convert to local coordinates
        agents_states_all[..., AgentIndex.STATE_SE2] = convert_absolute_to_relative_se2_array(
            ego_state.center, agents_states_all[..., AgentIndex.STATE_SE2]
        )

        # filter detections
        within_radius = np.linalg.norm(agents_states_all[..., AgentIndex.POINT], axis=-1) <= radius
        agents_states_all = agents_states_all[within_radius]

    agent_states = np.array(agents_states_all, dtype=np.float32)
    agent_mask = np.zeros(len(agents_states_all), dtype=bool)

    return SledgeVectorElement(agent_states, agent_mask)


def compute_static_object_features(
    ego_state: EgoState,
    detections: DetectionsTracks,
    radius: float,
    drivable_area_map: Optional[PDMOccupancyMap] = None,
) -> SledgeVectorElement:
    """
    Computes raw sledge vector features for static objects (ie. barriers, generic)
    :param ego_state: object of ego vehicle state in nuPlan
    :param detections: dataclass for detected objects in nuPlan
    :param radius: radius around the ego vehicle to extract objects
    :param drivable_area_map: drivable area map for filtering if provided, defaults to None
    :return: raw sledge vector element of all static object classes
    """

    tracked_objects = detections.tracked_objects
    objects_list: List[Agent] = tracked_objects.get_static_objects()

    objects_states_list: List[npt.NDArray[np.float64]] = []
    for object in objects_list:
        object_states_ = np.zeros(StaticObjectIndex.size(), dtype=np.float64)
        object_states_[StaticObjectIndex.STATE_SE2] = state_se2_to_array(object.center)
        object_states_[StaticObjectIndex.WIDTH] = object.box.width
        object_states_[StaticObjectIndex.LENGTH] = object.box.length
        objects_states_list.append(object_states_)
    objects_states_all = np.array(objects_states_list)

    # convert to local coords and filter out of box
    if len(objects_states_all) > 0:
        if drivable_area_map is not None:
            in_drivable_area = drivable_area_map.points_in_polygons(
                objects_states_all[..., StaticObjectIndex.POINT]
            ).any(axis=0)
            objects_states_all = objects_states_all[in_drivable_area]

        # convert to local coordinates
        objects_states_all[..., StaticObjectIndex.STATE_SE2] = convert_absolute_to_relative_se2_array(
            ego_state.center, objects_states_all[..., StaticObjectIndex.STATE_SE2]
        )
        # filter detections
        within_radius = np.linalg.norm(objects_states_all[..., StaticObjectIndex.POINT], axis=-1) <= radius
        objects_states_all = objects_states_all[within_radius]

    object_states = np.array(objects_states_all, dtype=np.float32)
    object_mask = np.zeros(len(objects_states_all), dtype=bool)

    return SledgeVectorElement(object_states, object_mask)
