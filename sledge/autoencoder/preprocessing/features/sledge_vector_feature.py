from __future__ import annotations

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union
from enum import Enum

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType, to_tensor
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature


@dataclass
class SledgeConfig:
    """General autoencoder config, for SledgeVector(Raw) features."""

    # 1. features raw
    radius: int = 100
    pose_interval: int = 1.0

    # 2. features raster & vector
    frame: Tuple[int, int] = (64, 64)

    num_lines: int = 50
    num_vehicles: int = 50
    num_pedestrians: int = 20
    num_static_objects: int = 30
    num_green_lights: int = 20
    num_red_lights: int = 20

    num_line_poses: int = 20
    vehicle_max_velocity: float = 15
    pedestrian_max_velocity: float = 2

    pixel_size: float = 0.25
    line_dots_radius: int = 0

    # output
    threshold: float = 0.3

    def __post_init__(self):
        assert 0 <= self.threshold <= 1, "Config threshold must be in [0,1]"

    @property
    def pixel_frame(self) -> Tuple[int, int]:
        frame_width, frame_height = self.frame
        return int(frame_width / self.pixel_size), int(frame_height / self.pixel_size)


@dataclass
class SledgeVector(AbstractModelFeature):
    """Feature class of complete vector representation in sledge."""

    lines: SledgeVectorElement
    vehicles: SledgeVectorElement
    pedestrians: SledgeVectorElement
    static_objects: SledgeVectorElement
    green_lights: SledgeVectorElement
    red_lights: SledgeVectorElement
    ego: SledgeVectorElement

    def to_device(self, device: torch.device) -> SledgeVector:
        """Implemented. See interface."""
        return SledgeVector(
            lines=self.lines.to_device(device=device),
            vehicles=self.vehicles.to_device(device=device),
            pedestrians=self.pedestrians.to_device(device=device),
            static_objects=self.static_objects.to_device(device=device),
            green_lights=self.green_lights.to_device(device=device),
            red_lights=self.red_lights.to_device(device=device),
            ego=self.ego.to_device(device=device),
        )

    def to_feature_tensor(self) -> SledgeVector:
        """Inherited, see superclass."""
        return SledgeVector(
            lines=self.lines.to_feature_tensor(),
            vehicles=self.vehicles.to_feature_tensor(),
            pedestrians=self.pedestrians.to_feature_tensor(),
            static_objects=self.static_objects.to_feature_tensor(),
            green_lights=self.green_lights.to_feature_tensor(),
            red_lights=self.red_lights.to_feature_tensor(),
            ego=self.ego.to_feature_tensor(),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> SledgeVector:
        """Implemented. See interface."""
        return SledgeVector(
            lines=SledgeVectorElement.deserialize(data["lines"]),
            vehicles=SledgeVectorElement.deserialize(data["vehicles"]),
            pedestrians=SledgeVectorElement.deserialize(data["pedestrians"]),
            static_objects=SledgeVectorElement.deserialize(data["static_objects"]),
            green_lights=SledgeVectorElement.deserialize(data["green_lights"]),
            red_lights=SledgeVectorElement.deserialize(data["red_lights"]),
            ego=SledgeVectorElement.deserialize(data["ego"]),
        )

    def unpack(self) -> List[SledgeVector]:
        """Implemented. See interface."""
        return [
            SledgeVector(lines, vehicles, pedestrians, static_objects, green_lights, red_lights, ego)
            for lines, vehicles, pedestrians, static_objects, green_lights, red_lights, ego in zip(
                self.lines.unpack(),
                self.vehicles.unpack(),
                self.pedestrians.unpack(),
                self.static_objects.unpack(),
                self.green_lights.unpack(),
                self.red_lights.unpack(),
                self.ego.unpack(),
            )
        ]

    @classmethod
    def collate(cls, batch: List[SledgeVector]) -> SledgeVector:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        return SledgeVector(
            lines=SledgeVectorElement.collate([item.lines for item in batch]),
            vehicles=SledgeVectorElement.collate([item.vehicles for item in batch]),
            pedestrians=SledgeVectorElement.collate([item.pedestrians for item in batch]),
            static_objects=SledgeVectorElement.collate([item.static_objects for item in batch]),
            green_lights=SledgeVectorElement.collate([item.green_lights for item in batch]),
            red_lights=SledgeVectorElement.collate([item.red_lights for item in batch]),
            ego=SledgeVectorElement.collate([item.ego for item in batch]),
        )

    def torch_to_numpy(self, apply_sigmoid: bool = True) -> SledgeVector:
        """Helper method to convert feature from torch tensor to numpy array."""
        return SledgeVector(
            lines=self.lines.torch_to_numpy(apply_sigmoid),
            vehicles=self.vehicles.torch_to_numpy(apply_sigmoid),
            pedestrians=self.pedestrians.torch_to_numpy(apply_sigmoid),
            static_objects=self.static_objects.torch_to_numpy(apply_sigmoid),
            green_lights=self.green_lights.torch_to_numpy(apply_sigmoid),
            red_lights=self.red_lights.torch_to_numpy(apply_sigmoid),
            ego=self.ego.torch_to_numpy(apply_sigmoid),
        )


@dataclass
class SledgeVectorRaw(SledgeVector):
    """Feature class of raw vector representation, for feature caching."""

    # NOTE: Placeholder class for type hints
    pass


@dataclass
class SledgeVectorElement(AbstractModelFeature):
    """Feature class individual vector element, eg. line, vehicle, etc."""

    states: FeatureDataType
    mask: FeatureDataType

    def to_device(self, device: torch.device) -> SledgeVectorElement:
        """Implemented. See interface."""
        validate_type(self.states, torch.Tensor)
        validate_type(self.mask, torch.Tensor)
        return SledgeVectorElement(states=self.states.to(device=device), mask=self.mask.to(device=device))

    def to_feature_tensor(self) -> SledgeVectorElement:
        """Inherited, see superclass."""
        return SledgeVectorElement(states=to_tensor(self.states), mask=to_tensor(self.mask))

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> SledgeVectorElement:
        """Implemented. See interface."""
        return SledgeVectorElement(states=data["states"], mask=data["mask"])

    def unpack(self) -> List[SledgeVectorElement]:
        """Implemented. See interface."""
        return [SledgeVectorElement(states[None], mask[None]) for states, mask in zip(self.states, self.mask)]

    def torch_to_numpy(self, apply_sigmoid: bool = True) -> SledgeVectorElement:
        """Helper method to convert feature from torch tensor to numpy array."""
        _mask = self.mask.sigmoid() if apply_sigmoid else self.mask
        return SledgeVectorElement(
            states=self.states.squeeze(dim=0).detach().cpu().numpy(),
            mask=_mask.squeeze(dim=0).detach().cpu().numpy(),
        )

    def get_element_type(self) -> SledgeVectorElementType:
        """Helper method to get type of vector element."""
        # NOTE: assumes types have different state sizes.
        n_dim = self.states.shape[-1]
        if n_dim == LineIndex.size():
            return SledgeVectorElementType.LINE
        elif n_dim == AgentIndex.size():
            return SledgeVectorElementType.AGENT
        elif n_dim == StaticObjectIndex.size():
            return SledgeVectorElementType.STATIC
        elif n_dim == EgoIndex.size():
            return SledgeVectorElementType.EGO
        else:
            raise ValueError("SledgeVectorElement cannot be matched to types in SledgeVectorElementType!")

    def get_element_index(self) -> Union[LineIndex, AgentIndex, StaticObjectIndex, EgoIndex]:
        """Helper method to get index enum of vector element."""
        element_type = self.get_element_type()
        if element_type == SledgeVectorElementType.LINE:
            return LineIndex
        elif element_type == SledgeVectorElementType.AGENT:
            return AgentIndex
        elif element_type == SledgeVectorElementType.STATIC:
            return StaticObjectIndex
        elif element_type == SledgeVectorElementType.EGO:
            return EgoIndex
        else:
            raise ValueError("SledgeVectorElement cannot be matched to types in SledgeVectorElementType!")


class SledgeVectorElementType(Enum):
    """Enum of vector element types."""

    LINE = 0
    AGENT = 1
    STATIC = 2
    EGO = 3


class LineIndex:
    """Index Enum of line states in SledgeVectorElement."""

    _X = 0
    _Y = 1

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
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def POINT(cls):
        # NOTE: assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)


class AgentIndex:
    """Index Enum of vehicles and pedestrian states in SledgeVectorElement."""

    _X = 0
    _Y = 1
    _HEADING = 2
    _WIDTH = 3
    _LENGTH = 4
    _VELOCITY = 5

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
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def VELOCITY(cls):
        return cls._VELOCITY

    @classmethod
    @property
    def POINT(cls):
        # NOTE: assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # NOTE: assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)


class StaticObjectIndex:
    """Index Enum of static object states in SledgeVectorElement."""

    _X = 0
    _Y = 1
    _HEADING = 2
    _WIDTH = 3
    _LENGTH = 4

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
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def POINT(cls):
        # NOTE: assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # NOTE: assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)


class EgoIndex:
    """Index Enum of ego state in SledgeVectorElement."""

    _VELOCITY_X = 0
    _VELOCITY_Y = 1
    _ACCELERATION_X = 2
    _ACCELERATION_Y = 3

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
    def VELOCITY_X(cls):
        return cls._VELOCITY_X

    @classmethod
    @property
    def VELOCITY_Y(cls):
        return cls._VELOCITY_Y

    @classmethod
    @property
    def ACCELERATION_X(cls):
        return cls._ACCELERATION_X

    @classmethod
    @property
    def ACCELERATION_Y(cls):
        return cls._ACCELERATION_Y

    @classmethod
    @property
    def VELOCITY_2D(cls):
        # assumes velocity X, Y have subsequent indices
        return slice(cls._VELOCITY_X, cls._VELOCITY_Y + 1)

    @classmethod
    @property
    def ACCELERATION_2D(cls):
        # assumes acceleration X, Y have subsequent indices
        return slice(cls._ACCELERATION_X, cls._ACCELERATION_Y + 1)


BoundingBoxIndex = Union[AgentIndex, StaticObjectIndex]
