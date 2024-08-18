import copy
from typing import List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from sledge.simulation.planner.pdm_planner.utils.pdm_array_representation import array_to_states_se2, array_to_state_se2
from sledge.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig
from sledge.autoencoder.preprocessing.features.sledge_raster_feature import SledgeRaster, SledgeRasterIndex
from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_utils import (
    coords_in_frame,
    coords_to_pixel,
    pixel_in_frame,
    raster_mask_oriented_box,
)
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeConfig,
    SledgeVector,
    SledgeVectorRaw,
    SledgeVectorElement,
    StaticObjectIndex,
    AgentIndex,
    EgoIndex,
)


def sledge_raw_feature_processing(
    sledge_vector_raw: SledgeVectorRaw, config: SledgeConfig
) -> Tuple[SledgeVector, SledgeRaster]:
    """
    Computes sledge vector from raw format on-the-fly, ie. during augmentation.
    - processes the raw vector format to the frame and settings in the configuration.
    - rasterize the processed vector representation

    :param sledge_vector_raw: raw representation of a vector in sledge, see dataclass.
    :param config: configuration for sledge autoencoder, incl. frame and raster config
    :return: tuple for processed sledge vector and raster dataclasses
    """

    pixel_height, pixel_width = config.pixel_frame
    raster = np.zeros((pixel_height, pixel_width, SledgeRasterIndex.size()), dtype=np.float32)

    vector_lines, raster_lines = process_lines(
        sledge_vector_raw.lines,
        config,
        config.num_lines,
    )
    vector_vehicles, raster_vehicles = process_agents(
        sledge_vector_raw.vehicles,
        config,
        config.num_vehicles,
        config.vehicle_max_velocity,
        ego_element=sledge_vector_raw.ego,
    )
    vector_pedestrians, raster_pedestrians = process_agents(
        sledge_vector_raw.pedestrians,
        config,
        config.num_pedestrians,
        config.pedestrian_max_velocity,
    )
    vector_static, raster_static = process_static_objects(
        sledge_vector_raw.static_objects,
        config,
    )
    vector_green_lines, raster_green_lines = process_lines(
        sledge_vector_raw.green_lights,
        config,
        config.num_green_lights,
    )
    vector_red_lines, raster_red_lines = process_lines(
        sledge_vector_raw.red_lights,
        config,
        config.num_red_lights,
    )

    raster[..., SledgeRasterIndex.LINE] = raster_lines
    raster[..., SledgeRasterIndex.VEHICLE] = raster_vehicles
    raster[..., SledgeRasterIndex.PEDESTRIAN] = raster_pedestrians
    raster[..., SledgeRasterIndex.STATIC_OBJECT] = raster_static
    raster[..., SledgeRasterIndex.GREEN_LIGHT] = raster_green_lines
    raster[..., SledgeRasterIndex.RED_LIGHT] = raster_red_lines

    return (
        SledgeVector(
            vector_lines,
            vector_vehicles,
            vector_pedestrians,
            vector_static,
            vector_green_lines,
            vector_red_lines,
            copy.deepcopy(sledge_vector_raw.ego),
        ),
        SledgeRaster(raster),
    )


def process_lines(
    lines: SledgeVectorElement, config: SledgeConfig, num_lines: int
) -> Tuple[SledgeVectorElement, npt.NDArray[np.float32]]:
    """
    TODO: Refactor
    Processes the lines entities from raw vector format
    - sort and interpolate nearest lines for fixed sized array
    - rasterize lines in the two image channels

    :param lines: raw line vector element
    :param config: dataclass for sledge autoencoder
    :param num_lines: max number of lines in output representations
    :return: tuple of processed line elements and line channels
    """

    # 1. preprocess lines (e.g. check if in frame)
    lines_in_frame = []
    for line_states, line_mask in zip(lines.states, lines.mask):
        line_in_mask = line_states[line_mask]  # (n, 3)
        if len(line_in_mask) < 2:
            continue

        path = PDMPath(array_to_states_se2(line_in_mask))
        distances = np.arange(
            0,
            path.length + config.pixel_size,
            config.pixel_size,
        )
        line = path.interpolate(distances, as_array=True)
        frame_mask = coords_in_frame(line[..., :2], config.frame)
        indices_segments = find_consecutive_true_indices(frame_mask)

        for indices_segment in indices_segments:
            line_segment = line[indices_segment]
            if len(line_segment) < 3:
                continue
            lines_in_frame.append(line_segment)

    # sort out nearest num_lines elements
    lines_distances = [np.linalg.norm(line[..., :2], axis=-1).min() for line in lines_in_frame]
    lines_in_frame = [lines_in_frame[idx] for idx in np.argsort(lines_distances)[:num_lines]]

    # 2. rasterize preprocessed lines
    pixel_height, pixel_width = config.pixel_frame
    raster_lines = np.zeros((pixel_height, pixel_width, 2), dtype=np.float32)
    for line in lines_in_frame:

        # encode orientation as color value
        dxy = np.concatenate([np.cos(line[..., 2, None]), np.sin(line[..., 2, None])], axis=-1)
        values = 0.5 * (dxy + 1)
        pixel_coords = coords_to_pixel(line[..., :2], config.frame, config.pixel_size)
        pixel_mask = pixel_in_frame(pixel_coords, config.pixel_frame)

        pixel_coords, values = pixel_coords[pixel_mask], values[pixel_mask]
        raster_lines[pixel_coords[..., 0], pixel_coords[..., 1]] = values

        if config.line_dots_radius > 0:
            thickness = -1
            if len(values) > 1:

                # NOTE: OpenCV has origin on top-left corner
                cv2.circle(
                    raster_lines,
                    (pixel_coords[0, 1], pixel_coords[0, 0]),
                    radius=config.line_dots_radius,
                    color=values[0],
                    thickness=thickness,
                )
                cv2.circle(
                    raster_lines,
                    (pixel_coords[-1, 1], pixel_coords[-1, 0]),
                    radius=config.line_dots_radius,
                    color=values[-1],
                    thickness=thickness,
                )

    # 3. vectorized preprocessed lines
    vector_states = np.zeros((num_lines, config.num_line_poses, 2), dtype=np.float32)
    vector_labels = np.zeros((num_lines), dtype=bool)
    vector_labels[: len(lines_in_frame)] = True

    for line_idx, line in enumerate(lines_in_frame):
        path = PDMPath(array_to_states_se2(line))
        distances = np.linspace(0, path.length, num=config.num_line_poses, endpoint=True)
        vector_states[line_idx] = path.interpolate(distances, as_array=True)[..., :2]

    return SledgeVectorElement(vector_states, vector_labels), raster_lines


def process_agents(
    agents: SledgeVectorElement,
    config: SledgeConfig,
    num_agents: int,
    max_velocity: float,
    ego_element: Optional[SledgeVectorElement] = None,
) -> Tuple[SledgeVectorElement, npt.NDArray[np.float32]]:
    """
    TODO: Refactor
    Processes the agent entities from raw vector format
    - sort and interpolate nearest agents for fixed sized array
    - rasterize agents in the two image channels

    :param agents: raw sledge vector element of agent category
    :param config: config of sledge autoencoder
    :param num_agents: max number of agent type in output representations
    :param max_velocity: max velocity of agent type
    :param ego_element: optional ego vehicle feature (ie. rasterize in vehicle channel), defaults to None
    :return: tuple of processed agent elements and agent channels
    """

    # 1. vectorized raw agents (e.g. cap max number)
    agents_states_all = agents.states
    vector_states = np.zeros((num_agents, AgentIndex.size()), dtype=np.float32)
    vector_labels = np.zeros(num_agents, dtype=bool)
    if len(agents_states_all) > 0:
        frame_mask = coords_in_frame(agents_states_all[..., AgentIndex.POINT], config.frame)
        agents_states_frame = agents_states_all[frame_mask]
        distances = np.linalg.norm(agents_states_frame[..., AgentIndex.POINT], axis=-1)
        argsort = np.argsort(distances)[:num_agents]
        agents_states_nearest = agents_states_frame[argsort]
        vector_states[: len(agents_states_nearest)] = agents_states_nearest
        vector_labels[: len(agents_states_nearest)] = True
    vector_states[..., AgentIndex.VELOCITY] = np.minimum(vector_states[..., AgentIndex.VELOCITY], max_velocity)

    # 2. rasterize agent bounding boxes
    pixel_height, pixel_width = config.pixel_frame
    raster_agents = np.zeros((pixel_height, pixel_width, 2), dtype=np.float32)
    for agent_state in vector_states[vector_labels]:
        # Get the 2D coordinate of the detected agents.
        oriented_box = OrientedBox(
            array_to_state_se2(agent_state[AgentIndex.STATE_SE2]),
            agent_state[AgentIndex.LENGTH],
            agent_state[AgentIndex.WIDTH],
            1.0,  # NOTE: dummy height
        )
        raster_mask = raster_mask_oriented_box(oriented_box, config)

        # Calculate change in position
        dx = agent_state[AgentIndex.VELOCITY] * np.cos(agent_state[AgentIndex.HEADING])
        dy = agent_state[AgentIndex.VELOCITY] * np.sin(agent_state[AgentIndex.HEADING])

        raster_agents[raster_mask, 0] = 0.5 * (dx / max_velocity + 1)
        raster_agents[raster_mask, 1] = 0.5 * (dy / max_velocity + 1)

    # Optional 3. rasterize ego bounding box (ie. for vehicles channels)
    if ego_element:
        ego_car_footprint = get_pacifica_parameters()
        ego_center_array = np.zeros(3, dtype=np.float32)
        oriented_box = OrientedBox(
            array_to_state_se2(ego_center_array),
            ego_car_footprint.length,
            ego_car_footprint.width,
            1.0,  # NOTE: dummy height
        )

        raster_mask = raster_mask_oriented_box(oriented_box, config)
        velocity = np.linalg.norm(ego_element.states[EgoIndex.VELOCITY_2D], axis=-1)
        velocity = np.minimum(velocity, max_velocity)

        # Calculate change in position
        dx = velocity * np.cos(ego_center_array[AgentIndex.HEADING])
        dy = velocity * np.sin(ego_center_array[AgentIndex.HEADING])

        raster_agents[raster_mask, 0] = 0.5 * (dx / max_velocity + 1)
        raster_agents[raster_mask, 1] = 0.5 * (dy / max_velocity + 1)

    return SledgeVectorElement(vector_states, vector_labels), raster_agents


def process_static_objects(
    static_objects: SledgeVectorElement, config: RVAEConfig
) -> Tuple[SledgeVectorElement, npt.NDArray[np.float32]]:
    """
    TODO: Refactor
    Processes the static object entities from raw vector format
    - sort and interpolate nearest objects for fixed sized array
    - rasterize objects in the two image channels

    :param static_objects: raw sledge vector element of static bounding boxes
    :param config: config of sledge autoencoder
    :return: tuple of processed objects elements and object channels
    """

    # 1. vectorized raw static objects (e.g. cap max number)
    states_all = static_objects.states
    vector_states = np.zeros((config.num_static_objects, StaticObjectIndex.size()), dtype=np.float32)
    vector_labels = np.zeros(config.num_static_objects, dtype=bool)

    if len(states_all) > 0:
        frame_mask = coords_in_frame(states_all[..., StaticObjectIndex.POINT], config.frame)
        static_states_frame = states_all[frame_mask]

        distances = np.linalg.norm(static_states_frame[..., StaticObjectIndex.POINT], axis=-1)
        argsort = np.argsort(distances)[: config.num_static_objects]
        static_states_nearest = static_states_frame[argsort]

        vector_states[: len(static_states_nearest)] = static_states_nearest
        vector_labels[: len(static_states_nearest)] = True

    # 2. rasterize static object bounding boxes
    pixel_height, pixel_width = config.pixel_frame
    raster_objects = np.zeros((pixel_height, pixel_width, 2), dtype=np.float32)
    for agent_state in vector_states[vector_labels]:
        # Get the 2D coordinate of the detected objects.
        oriented_box = OrientedBox(
            array_to_state_se2(agent_state[StaticObjectIndex.STATE_SE2]),
            agent_state[StaticObjectIndex.LENGTH],
            agent_state[StaticObjectIndex.WIDTH],
            1.0,  # NOTE: dummy height
        )
        raster_mask = raster_mask_oriented_box(oriented_box, config)

        # Calculate change in position
        dx = np.cos(agent_state[StaticObjectIndex.HEADING])
        dy = np.sin(agent_state[StaticObjectIndex.HEADING])

        raster_objects[raster_mask, 0] = 0.5 * (dx + 1)
        raster_objects[raster_mask, 1] = 0.5 * (dy + 1)

    return SledgeVectorElement(vector_states, vector_labels), raster_objects


def find_consecutive_true_indices(mask: npt.NDArray[np.bool_]) -> List[npt.NDArray[np.int32]]:
    """
    Helper function for line preprocessing.
    For example, lines might exceed or return into frame.
    Find regions in mask where line is consecutively in frame (ie. to split line)

    :param mask: 1D numpy array of booleans
    :return: List of int32 arrays, where mask is consecutively true.
    """

    padded_mask = np.pad(np.asarray(mask), (1, 1), "constant", constant_values=False)

    changes = np.diff(padded_mask.astype(int))
    starts = np.where(changes == 1)[0]  # indices of False -> True
    ends = np.where(changes == -1)[0]  # indices of True -> False

    return [np.arange(start, end) for start, end in zip(starts, ends)]
