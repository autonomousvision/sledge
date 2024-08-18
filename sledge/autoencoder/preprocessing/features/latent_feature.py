from __future__ import annotations

from typing import Any, Dict, List
from dataclasses import dataclass
import torch

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType, to_tensor


@dataclass
class Latent(AbstractModelFeature):
    """Feature class of latent variable."""

    mu: FeatureDataType
    log_var: FeatureDataType

    def to_device(self, device: torch.device) -> Latent:
        """Implemented. See interface."""
        validate_type(self.mu, torch.Tensor)
        validate_type(self.log_var, torch.Tensor)
        return Latent(mu=self.mu.to(device=device), log_var=self.log_var.to(device=device))

    def to_feature_tensor(self) -> Latent:
        """Inherited, see superclass."""
        return Latent(mu=to_tensor(self.mu), log_var=to_tensor(self.log_var))

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Latent:
        """Implemented. See interface."""
        return Latent(mu=data["mu"], log_var=data["log_var"])

    def unpack(self) -> List[Latent]:
        """Implemented. See interface."""
        return [Latent(mu, log_var) for mu, log_var in zip(self.mu, self.log_var)]

    def torch_to_numpy(self) -> Latent:
        """Helper method to convert feature from torch tensor to numpy array."""
        return Latent(mu=self.mu.detach().cpu().numpy(), log_var=self.log_var.detach().cpu().numpy())

    def squeeze(self) -> Latent:
        """Helper method to apply .squeeze() on features."""
        return Latent(mu=self.mu.squeeze(0), log_var=self.log_var.squeeze(0))
