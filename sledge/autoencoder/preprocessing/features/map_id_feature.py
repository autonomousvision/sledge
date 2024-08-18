from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType, to_tensor

MAP_NAME_ID_ABBR = [
    (0, "us-nv-las-vegas-strip", "LAV"),
    (1, "us-pa-pittsburgh-hazelwood", "PGH"),
    (2, "sg-one-north", "SGP"),
    (3, "us-ma-boston", "BOS"),
]

MAP_NAME_TO_ID = {name: id for id, name, abbr in MAP_NAME_ID_ABBR}
MAP_ID_TO_NAME = {id: name for id, name, abbr in MAP_NAME_ID_ABBR}
MAP_ID_TO_ABBR = {id: abbr for id, name, abbr in MAP_NAME_ID_ABBR}


@dataclass
class MapID(AbstractModelFeature):
    """Feature class of to store map id."""

    id: FeatureDataType

    def to_device(self, device: torch.device) -> MapID:
        """Implemented. See interface."""
        validate_type(self.id, torch.Tensor)
        return MapID(id=self.id.to(device=device))

    def to_feature_tensor(self) -> MapID:
        """Inherited, see superclass."""
        return MapID(id=to_tensor(self.id))

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> MapID:
        """Implemented. See interface."""
        return MapID(id=data["id"])

    def unpack(self) -> List[MapID]:
        """Implemented. See interface."""
        return [MapID(id) for id in zip(self.id)]

    def torch_to_numpy(self) -> MapID:
        """Helper method to convert feature from torch tensor to numpy array."""
        return MapID(id=self.id.detach().cpu().numpy())

    def squeeze(self) -> MapID:
        """Helper method to apply .squeeze() on features."""
        return MapID(id=self.squeeze(0))
