from typing import Optional, Tuple
import cv2

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.oriented_box import OrientedBox

from sledge.common.visualization.sledge_colors import Color, BLACK, WHITE, SLEDGE_ELEMENTS
from sledge.simulation.planner.pdm_planner.utils.pdm_array_representation import array_to_state_se2
from sledge.autoencoder.preprocessing.features.map_id_feature import MapID, MAP_ID_TO_ABBR
from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_utils import coords_to_pixel
from sledge.autoencoder.preprocessing.features.sledge_raster_feature import SledgeRaster
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeConfig,
    SledgeVector,
    SledgeVectorElement,
    SledgeVectorElementType,
    BoundingBoxIndex,
)


def add_border_to_raster(
    image: npt.NDArray[np.uint8], border_size: int = 1, border_color: Color = BLACK
) -> npt.NDArray[np.uint8]:
    """
    Add boarder to numpy array / image.
    :param image: image as numpy array
    :param border_size: size of border in pixels, defaults to 1
    :param border_color: color of border, defaults to BLACK
    :return: image with border.
    """
    bordered_image = cv2.copyMakeBorder(
        image,
        border_size,
        border_size,
        border_size,
        border_size,
        cv2.BORDER_CONSTANT,
        value=border_color.rgb,
    )
    return bordered_image


def get_sledge_raster(
    raster: SledgeRaster, pixel_frame: Tuple[int, int], threshold: float = 0.0, add_border: bool = True
) -> npt.NDArray[np.uint8]:
    """
    Convert sledge raster dataclass to numpy RGB image.
    :param raster: sledge raster dataclass
    :param pixel_frame: sizel of pixel in meter
    :param threshold: threshold for color channels, defaults to 0.0
    :param add_border: whether to add border, defaults to True
    :return: numpy RGB image
    """
    # FIXME: method only works reliably for ground-trough rasters
    pixel_width, pixel_height = pixel_frame
    image: npt.NDArray[np.uint8] = np.full((pixel_width, pixel_height, 3), WHITE.rgb, dtype=np.uint8)
    image[raster.lines_layer[0].mean(0) > threshold] = SLEDGE_ELEMENTS["lines"].rgb
    image[raster.vehicles_layer[0].mean(0) > threshold] = SLEDGE_ELEMENTS["vehicles"].rgb
    image[raster.pedestrians_layer[0].mean(0) > threshold] = SLEDGE_ELEMENTS["pedestrians"].rgb
    image[raster.static_objects_layer[0].mean(0) > threshold] = SLEDGE_ELEMENTS["static_objects"].rgb
    image[raster.green_lights_layer[0].mean(0) > threshold] = SLEDGE_ELEMENTS["green_lights"].rgb
    image[raster.red_lights_layer[0].mean(0) > threshold] = SLEDGE_ELEMENTS["red_lights"].rgb
    image = image[::-1, ::-1]

    if add_border:
        image = add_border_to_raster(image)

    return image


def get_sledge_vector_as_raster(
    sledge_vector: SledgeVector, config: SledgeConfig, map_id: Optional[MapID] = None
) -> npt.NDArray[np.uint8]:
    """
    Convert sledge vector into RGB numpy array for visualization.
    :param sledge_vector: dataclass of vector representation
    :param config: config dataclass of sledge autoencoder
    :param map_id: map identifier to draw if provided, defaults to None
    :return: numpy RGB image
    """

    pixel_width, pixel_height = config.pixel_frame
    image: npt.NDArray[np.uint8] = np.full((pixel_width, pixel_height, 3), WHITE.rgb, dtype=np.uint8)
    draw_dict = {
        "L": {"elem": sledge_vector.lines, "color": SLEDGE_ELEMENTS["lines"], "count": 0},
        "V": {"elem": sledge_vector.vehicles, "color": SLEDGE_ELEMENTS["vehicles"], "count": 0},
        "P": {"elem": sledge_vector.pedestrians, "color": SLEDGE_ELEMENTS["pedestrians"], "count": 0},
        "S": {"elem": sledge_vector.static_objects, "color": SLEDGE_ELEMENTS["static_objects"], "count": 0},
        "G": {"elem": sledge_vector.green_lights, "color": SLEDGE_ELEMENTS["green_lights"], "count": 0},
        "R": {"elem": sledge_vector.red_lights, "color": SLEDGE_ELEMENTS["red_lights"], "count": 0},
    }

    for key, elem_dict in draw_dict.items():
        image, counter = draw_sledge_vector_element(image, elem_dict["elem"], config, elem_dict["color"])
        draw_dict[key]["count"] = counter

    # TODO: adapt to autoencoder config
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    line_height = 15
    width_offset = pixel_width - 39
    height_offset = pixel_height - 99

    for i, (key, elem_dict) in enumerate(draw_dict.items()):
        count = elem_dict["count"]
        cv2.putText(
            image,
            f"{key}={count}",
            (width_offset, height_offset + (i + 1) * line_height),
            font,
            font_scale,
            elem_dict["color"].rgb,
            thickness,
            cv2.LINE_AA,
        )

        draw_dict[key]["count"] = counter

    if map_id:
        map_abbreviation = MAP_ID_TO_ABBR[int(map_id.id[0])]
        cv2.putText(
            image,
            f"{map_abbreviation}",
            (width_offset, height_offset),
            font,
            font_scale,
            BLACK.rgb,
            thickness,
            cv2.LINE_AA,
        )

    image = add_border_to_raster(image)
    return image


def draw_sledge_vector_element(
    image: npt.NDArray[np.uint8], sledge_vector_element: SledgeVectorElement, config: SledgeConfig, color: Color
) -> Tuple[npt.NDArray[np.uint8], int]:
    """
    Draws vector element on numpy RGB image.
    :param image: numpy RGB image
    :param sledge_vector_element: vector element to draw
    :param config: dataclass config of autoencoder in sledge
    :param color: color helper class
    :return: tuple of numpy RGB image and element count
    """

    element_counter = 0
    element_type = sledge_vector_element.get_element_type()
    element_index = sledge_vector_element.get_element_index()

    for states, p in zip(sledge_vector_element.states, sledge_vector_element.mask):
        draw_element = False
        if type(p) is np.bool_:
            draw_element = p
        else:
            draw_element = p > config.threshold
        if not draw_element:
            continue

        if element_type == SledgeVectorElementType.LINE:
            image = draw_line_element(image, states, config, color)
        else:
            image = draw_bounding_box_element(image, states, config, color, element_index)
        element_counter += 1

    return image, element_counter


def draw_line_element(
    image: npt.NDArray[np.uint8], state: npt.NDArray[np.float32], config: SledgeConfig, color: Color
) -> npt.NDArray[np.uint8]:
    """
    Draws a line state (eg. of lane or traffic light) onto numpy RGB image.
    :param image: numpy RGB image
    :param state: coordinate array of line
    :param config: dataclass config of autoencoder in sledge
    :param color: color helper class
    :return: numpy RGB image
    """
    assert state.shape[-1] == 2
    line_mask = np.zeros(config.pixel_frame, dtype=np.float32)
    indices = coords_to_pixel(state, config.frame, config.pixel_size)
    coords_x, coords_y = indices[..., 0], indices[..., 1]

    # NOTE: OpenCV has origin on top-left corner
    for point_1, point_2 in zip(zip(coords_x[:-1], coords_y[:-1]), zip(coords_x[1:], coords_y[1:])):
        cv2.line(line_mask, point_1, point_2, color=1.0, thickness=1)

    cv2.circle(line_mask, (coords_x[0], coords_y[0]), radius=3, color=1.0, thickness=-1)
    cv2.circle(line_mask, (coords_x[-1], coords_y[-1]), radius=3, color=1.0, thickness=-1)
    line_mask = np.rot90(line_mask)[:, ::-1]

    image[line_mask > 0] = color.rgb
    return image


def draw_bounding_box_element(
    image: npt.NDArray[np.uint8],
    state: npt.NDArray[np.float32],
    config: SledgeConfig,
    color: Color,
    object_indexing: BoundingBoxIndex,
) -> npt.NDArray[np.uint8]:
    """
    Draws a bounding box (eg. of vehicle) onto numpy RGB image.
    :param image: numpy RGB image
    :param state: state array of bounding box
    :param config: dataclass config of autoencoder in sledge
    :param color: color helper class
    :param object_indexing: index enum of state array
    :return: numpy RGB image
    """

    # Get the 2D coordinate of the detected agents.
    raster_oriented_box = OrientedBox(
        array_to_state_se2(state[object_indexing.STATE_SE2]),
        state[object_indexing.LENGTH],
        state[object_indexing.WIDTH],
        1.0,  # NOTE: dummy height
    )
    box_bottom_corners = raster_oriented_box.all_corners()
    corners = np.asarray([[corner.x, corner.y] for corner in box_bottom_corners])  # type: ignore
    corner_indices = coords_to_pixel(corners, config.frame, config.pixel_size)

    bounding_box_mask = np.zeros(config.pixel_frame, dtype=np.float32)
    cv2.fillPoly(bounding_box_mask, [corner_indices], color=1.0, lineType=cv2.LINE_AA)

    # NOTE: OpenCV has origin on top-left corner
    bounding_box_mask = np.rot90(bounding_box_mask)[:, ::-1]

    image[bounding_box_mask > 0] = color.rgb
    return image
