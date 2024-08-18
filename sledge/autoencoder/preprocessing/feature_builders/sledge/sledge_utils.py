# TODO: Move these functions for general use.
# eg. sledge.common.visualization

from typing import Tuple

import cv2
import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.oriented_box import OrientedBox
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeConfig


def raster_mask_oriented_box(oriented_box: OrientedBox, config: SledgeConfig) -> npt.NDArray[np.bool_]:
    """
    Create raster mask if a oriented bounding box in BEV
    :param oriented_box: class of a bounding box with heading
    :param config: config dataclass of a sledge autoencoder
    :return: creates raster mask of oriented bounding box
    """

    pixel_width, pixel_height = config.pixel_frame
    corners = np.asarray([[corner.x, corner.y] for corner in oriented_box.all_corners()])
    corner_idcs = coords_to_pixel(corners, config.frame, config.pixel_size)

    # TODO: check if float32 is really necessary here.
    raster_mask = np.zeros((pixel_width, pixel_height), dtype=np.float32)
    cv2.fillPoly(raster_mask, [corner_idcs], color=1.0, lineType=cv2.LINE_AA)

    # NOTE: OpenCV has origin on top-left corner
    raster_mask = np.rot90(raster_mask)[::-1]
    return raster_mask > 0


def coords_in_frame(coords: npt.NDArray[np.float32], frame: Tuple[float, float]) -> npt.NDArray[np.bool_]:
    """
    Checks which coordinates are within the given 2D frame extend.
    :param coords: coordinate array in numpy (x,y) in last axis
    :param frame: tuple of frame extend in meter
    :return: numpy array of boolean's
    """
    assert coords.shape[-1] == 2, "Coordinate array must have last dim size of 2 (ie. x,y)"
    width, height = frame

    within_width = np.logical_and(-width / 2 <= coords[..., 0], coords[..., 0] <= width / 2)
    within_height = np.logical_and(-height / 2 <= coords[..., 1], coords[..., 1] <= height / 2)

    return np.logical_and(within_width, within_height)


def pixel_in_frame(pixel: npt.NDArray[np.int32], pixel_frame: Tuple[int, int]) -> npt.NDArray[np.bool_]:
    """
    Checks if pixels indices are within the image.
    :param pixel: pixel indices as numpy array
    :param pixel_frame: tuple of raster width and height
    :return: numpy array of boolean's
    """
    assert pixel.shape[-1] == 2, "Coordinate array must have last dim size of 2 (ie. x,y)"
    pixel_width, pixel_height = pixel_frame

    within_width = np.logical_and(0 <= pixel[..., 0], pixel[..., 0] < pixel_width)
    within_height = np.logical_and(0 <= pixel[..., 1], pixel[..., 1] < pixel_height)

    return np.logical_and(within_width, within_height)


def coords_to_pixel(
    coords: npt.NDArray[np.float32], frame: Tuple[float, float], pixel_size: float
) -> npt.NDArray[np.int32]:
    """
    Converts ego-centric coordinates into pixel coordinates (ie. indices)
    :param coords: coordinate array in numpy (x,y) in last axis
    :param frame: tuple of frame extend in meter
    :param pixel_size: size of a pixel
    :return: indices of pixel coordinates
    """
    assert coords.shape[-1] == 2

    width, height = frame
    pixel_width, pixel_height = int(width / pixel_size), int(height / pixel_size)
    pixel_center = np.array([[pixel_width / 2.0, pixel_height / 2.0]])
    coords_idcs = (coords / pixel_size) + pixel_center

    return coords_idcs.astype(np.int32)
