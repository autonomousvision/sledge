from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional

import torch
import torchvision
from torch import Tensor

import numpy as np
from numpy import ndarray

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType


@dataclass
class SledgeRaster(AbstractModelFeature):
    """Feature class of raster in sledge."""

    data: FeatureDataType

    @property
    def num_batches(self) -> Optional[int]:
        """Number of batches in the feature."""
        return None if len(self.data.shape) < 4 else self.data.shape[0]

    def to_feature_tensor(self) -> AbstractModelFeature:
        """Implemented. See interface."""
        to_tensor_torchvision = torchvision.transforms.ToTensor()
        data = to_tensor_torchvision(np.asarray(self.data))
        return SledgeRaster(data=data)

    def to_device(self, device: torch.device) -> SledgeRaster:
        """Implemented. See interface."""
        validate_type(self.data, torch.Tensor)
        return SledgeRaster(data=self.data.to(device=device))

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> SledgeRaster:
        """Implemented. See interface."""
        return SledgeRaster(data=data["data"])

    def unpack(self) -> List[SledgeRaster]:
        """Implemented. See interface."""
        return [SledgeRaster(data[None]) for data in self.data]

    @staticmethod
    def from_feature_tensor(tensor: torch.Tensor) -> SledgeRaster:
        """Implemented. See interface."""
        array = tensor.numpy()

        # So can assume that the torch tensor will always be channels first
        # and the numpy array will always be channels last.
        # So this moves the channels to last when reading the torch tensor
        if len(array.shape) == 4:
            array = array.transpose(0, 2, 3, 1)
        else:
            array = array.transpose(1, 2, 0)

        return SledgeRaster(array)

    @property
    def width(self) -> int:
        """
        :return: the width of a raster
        """
        return self.data.shape[-2] if self._is_channels_last() else self.data.shape[-1]  # type: ignore

    @property
    def height(self) -> int:
        """
        :return: the height of a raster
        """
        return self.data.shape[-3] if self._is_channels_last() else self.data.shape[-2]  # type: ignore

    def num_channels(self) -> int:
        """
        Number of raster channels.
        """
        return self.data.shape[-1] if self._is_channels_last() else self.data.shape[-3]  # type: ignore

    @property
    def lines_layer(self) -> FeatureDataType:
        return self._get_data_channel(SledgeRasterIndex.LINE)

    @property
    def vehicles_layer(self) -> FeatureDataType:
        return self._get_data_channel(SledgeRasterIndex.VEHICLE)

    @property
    def pedestrians_layer(self) -> FeatureDataType:
        return self._get_data_channel(SledgeRasterIndex.PEDESTRIAN)

    @property
    def static_objects_layer(self) -> FeatureDataType:
        return self._get_data_channel(SledgeRasterIndex.STATIC_OBJECT)

    @property
    def green_lights_layer(self) -> FeatureDataType:
        return self._get_data_channel(SledgeRasterIndex.GREEN_LIGHT)

    @property
    def red_lights_layer(self) -> FeatureDataType:
        return self._get_data_channel(SledgeRasterIndex.RED_LIGHT)

    def _is_channels_last(self) -> bool:
        """
        Check location of channel dimension
        :return True if position [-1] is the number of channels
        """
        # For tensor, channel is put right before the spatial dimention.
        if isinstance(self.data, Tensor):
            return False

        # The default numpy array data format is channel last.
        elif isinstance(self.data, ndarray):
            return True
        else:
            raise RuntimeError(
                f"The data needs to be either numpy array or torch Tensor, but got type(data): {type(self.data)}"
            )

    def _get_data_channel(self, index: Union[int, slice]) -> FeatureDataType:
        """
        Extract channel data
        :param index: of layer
        :return: data corresponding to layer
        """
        if self._is_channels_last():
            return self.data[..., index]
        else:
            return self.data[..., index, :, :]


class SledgeRasterIndex:
    """Index Enum of raster channel in SledgeRaster."""

    _LINE_X = 0
    _LINE_Y = 1
    _VEHICLE_X = 2
    _VEHICLE_Y = 3
    _PEDESTRIAN_X = 4
    _PEDESTRIAN_Y = 5
    _STATIC_OBJECT_X = 6
    _STATIC_OBJECT_Y = 7
    _GREEN_LIGHT_X = 8
    _GREEN_LIGHT_Y = 9
    _RED_LIGHT_X = 10
    _RED_LIGHT_Y = 11

    @classmethod
    def size(cls) -> int:
        """
        :return: number of channels
        """
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def LINE_X(cls):
        return cls._LINE_X

    @classmethod
    @property
    def LINE_Y(cls):
        return cls._LINE_Y

    @classmethod
    @property
    def LINE(cls):
        return slice(cls._LINE_X, cls._LINE_Y + 1)

    @classmethod
    @property
    def VEHICLE_X(cls):
        return cls._VEHICLE_X

    @classmethod
    @property
    def VEHICLE_Y(cls):
        return cls._VEHICLE_Y

    @classmethod
    @property
    def VEHICLE(cls):
        return slice(cls._VEHICLE_X, cls._VEHICLE_Y + 1)

    @classmethod
    @property
    def PEDESTRIAN_X(cls):
        return cls._PEDESTRIAN_X

    @classmethod
    @property
    def PEDESTRIAN_Y(cls):
        return cls._PEDESTRIAN_Y

    @classmethod
    @property
    def PEDESTRIAN(cls):
        return slice(cls._PEDESTRIAN_X, cls._PEDESTRIAN_Y + 1)

    @classmethod
    @property
    def STATIC_OBJECT_X(cls):
        return cls._STATIC_OBJECT_X

    @classmethod
    @property
    def STATIC_OBJECT_Y(cls):
        return cls._STATIC_OBJECT_Y

    @classmethod
    @property
    def STATIC_OBJECT(cls):
        return slice(cls._STATIC_OBJECT_X, cls._STATIC_OBJECT_Y + 1)

    @classmethod
    @property
    def GREEN_LIGHT_X(cls):
        return cls._GREEN_LIGHT_X

    @classmethod
    @property
    def GREEN_LIGHT_Y(cls):
        return cls._GREEN_LIGHT_Y

    @classmethod
    @property
    def GREEN_LIGHT(cls):
        return slice(cls._GREEN_LIGHT_X, cls._GREEN_LIGHT_Y + 1)

    @classmethod
    @property
    def RED_LIGHT_X(cls):
        return cls._RED_LIGHT_X

    @classmethod
    @property
    def RED_LIGHT_Y(cls):
        return cls._RED_LIGHT_Y

    @classmethod
    @property
    def RED_LIGHT(cls):
        return slice(cls._RED_LIGHT_X, cls._RED_LIGHT_Y + 1)
