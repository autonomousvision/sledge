# TODO: remove class and merge with PDMPath in sledge.common

from typing import Any, List, Union

import numpy as np
import numpy.typing as npt

from scipy.interpolate import interp1d
from shapely.creation import linestrings
from shapely.geometry import LineString

from sledge.simulation.planner.pdm_planner.utils.pdm_enums import SE2Index
from sledge.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle


def compute_headings(coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Compute the heading angles, based on the 2D coordinates of a line.
    :param coords: numpy array of (x,y) coordinates
    :return: 1D numpy arrays of heading angles
    """
    assert coords.ndim == 2
    assert coords.shape[-1] == 2

    A = coords[:-1]
    B = coords[1:]
    y = B[:, 1] - A[:, 1]
    x = B[:, 0] - A[:, 0]
    headings = np.arctan2(y, x)

    return np.append(headings, headings[-1])


def calculate_progress(states_se2_array: npt.NDArray[np.float64]) -> List[float]:
    """
    Calculate the cumulative progress of a given path.
    :param states_se2_array: array of (x,y,θ) in last dim
    :return: a cumulative array of progress
    """
    x_diff = np.diff(states_se2_array[:, 0])
    y_diff = np.diff(states_se2_array[:, 1])
    points_diff: npt.NDArray[np.float64] = np.concatenate(([x_diff], [y_diff]), axis=0, dtype=np.float64)
    progress_diff = np.append(0.0, np.linalg.norm(points_diff, axis=0))

    return np.cumsum(progress_diff, dtype=np.float64)


class SledgePath:
    """Interpolatable path from 2D coordinates."""

    def __init__(self, coords: npt.NDArray[np.float64]):
        """
        Initializes interpolatable path.
        NOTE: object will be removed in future (repetitive with PDMPath).
        :param coords:  numpy array of (x,y) coordinates
        """

        if coords.shape[-1] == 2:
            coords = np.concatenate([coords, compute_headings(coords)[..., None]], axis=-1)

        self._states_se2_array = np.copy(coords)
        self._states_se2_array[:, SE2Index.HEADING] = np.unwrap(self._states_se2_array[:, SE2Index.HEADING], axis=0)
        self._progress = calculate_progress(self._states_se2_array)

        self._linestring = linestrings(self._states_se2_array[:, : SE2Index.HEADING])
        self._interpolator = interp1d(self._progress, self._states_se2_array, axis=0)

    @property
    def length(self):
        """Getter for length of path."""
        return self._progress[-1]

    @property
    def states_se2_array(self):
        """Getter for state se2 array."""
        return self._states_se2_array

    @property
    def linestring(self) -> LineString:
        """Getter for shapely's linestring of path."""
        return self._linestring

    def project(self, points: Any) -> Any:
        """Projects point on linestring, to retrieve distance along path."""
        return self._linestring.project(points)

    def interpolate(
        self,
        distances: Union[List[float], npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        """
        Calculates (x,y,θ) for a given distance along the path.
        :param distances: list of array of distance values
        :param as_array: whether to return in array representation, defaults to False
        :return: array of StateSE2 class or (x,y,θ) values
        """
        clipped_distances = np.clip(distances, 1e-5, self.length)
        interpolated_se2_array = self._interpolator(clipped_distances)
        interpolated_se2_array[..., 2] = normalize_angle(interpolated_se2_array[..., 2])
        interpolated_se2_array[np.isnan(interpolated_se2_array)] = 0.0

        return interpolated_se2_array
