from abc import ABC, abstractmethod
import torch

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType


class AbstractMatching(ABC):
    """Matching interface used in for hungarian matching losses during training."""

    @abstractmethod
    @torch.no_grad()
    def compute(self, predictions: FeaturesType, targets: TargetsType) -> TargetsType:
        """
        Run matching between model predictions and targets for loss computation.
        :param predictions: Predicted feature tensors for matching.
        :param targets: Target feature tensors for matching
        :return: Matching formulation between prediction and target
        """
        pass
