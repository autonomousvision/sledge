from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import torch

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType, to_tensor


@dataclass
class RVAEMatchingFeature(AbstractModelFeature):
    """Feature class to score matched entities during RVAE training."""

    indices: FeatureDataType

    def to_device(self, device: torch.device) -> RVAEMatchingFeature:
        """Implemented. See interface."""
        validate_type(self.indices, torch.Tensor)
        return RVAEMatchingFeature(indices=self.indices.to(device=device))

    def to_feature_tensor(self) -> RVAEMatchingFeature:
        """Inherited, see superclass."""
        return RVAEMatchingFeature(indices=to_tensor(self.indices))

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> RVAEMatchingFeature:
        """Implemented. See interface."""
        return RVAEMatchingFeature(indices=data["indices"])

    def unpack(self) -> List[RVAEMatchingFeature]:
        """Implemented. See interface."""
        return [RVAEMatchingFeature(indices) for indices in zip(self.indices)]
