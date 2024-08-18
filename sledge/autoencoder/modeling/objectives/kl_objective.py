from typing import Dict

import torch

from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType

from sledge.autoencoder.preprocessing.features.latent_feature import Latent
from sledge.autoencoder.modeling.objectives.abstract_custom_objective import AbstractCustomObjective


class KLObjective(AbstractCustomObjective):
    """Kullback-Leibler divergence objective for VAEs."""

    def __init__(self, weight: float, scenario_type_loss_weighting: Dict[str, float]):
        """
        Initialize KL objective.
        :param weight: scalar for loss weighting (aka. β)
        :param scenario_type_loss_weighting: scenario-type specific loss weights (ignored).
        """
        self._weight = weight
        self._scenario_type_loss_weighting = scenario_type_loss_weighting

    def compute(
        self, predictions: FeaturesType, targets: TargetsType, matchings: TargetsType, scenarios: ScenarioListType
    ) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        pred_latent: Latent = predictions["latent"]
        mu, log_var = pred_latent.mu, pred_latent.log_var
        kl_loss = -0.5 * torch.mean(1 + log_var - mu**2 - log_var.exp())

        return {"kl_latent": self._weight * kl_loss}
