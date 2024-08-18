from typing import Dict

import torch

from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType

from sledge.autoencoder.preprocessing.features.latent_feature import Latent
from sledge.autoencoder.modeling.metrics.abstract_custom_metric import AbstractCustomMetric


class KLMetric(AbstractCustomMetric):
    def __init__(self, scenario_type_loss_weighting: Dict[str, float]):
        """
        Initializes the class
        :param scenario_type_loss_weighting: loss weight per scenario (ignored)
        """
        self._scenario_type_loss_weighting = scenario_type_loss_weighting

    def compute(
        self, predictions: FeaturesType, targets: TargetsType, matchings: TargetsType, scenarios: ScenarioListType
    ) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        pred_latent: Latent = predictions["latent"]
        mu, log_var = pred_latent.mu, pred_latent.log_var
        kl_metric = -0.5 * torch.mean(1 + log_var - mu**2 - log_var.exp())

        return {"kl_metric": kl_metric}
