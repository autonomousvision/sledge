from abc import ABC, abstractmethod
from typing import Dict

import torch

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType, ScenarioListType


class AbstractCustomMetric(ABC):
    """Custom abstract class for metric definition. Allows to multiple metric in one objects."""

    @abstractmethod
    def compute(
        self,
        predictions: FeaturesType,
        targets: TargetsType,
        matchings: TargetsType,
        scenarios: ScenarioListType,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the metric given the ground truth targets and the model's predictions.
        :param predictions: dictionary of model's predictions
        :param targets: dictionary of ground-truth targets from the dataset
        :param matchings: dictionary of matchings between targets and predictions
        :param scenarios: list if scenario types (for type-specific weighting)
        :return: dictionary of metric name and scalar.
        """
        pass
