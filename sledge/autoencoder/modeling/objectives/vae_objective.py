from typing import Dict

import torch
import torch.nn.functional as F

from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType

from sledge.autoencoder.preprocessing.features.sledge_raster_feature import SledgeRaster
from sledge.autoencoder.modeling.objectives.abstract_custom_objective import AbstractCustomObjective


class VAEL1Objective(AbstractCustomObjective):
    """Object for image reconstruction loss (ie. l1)."""

    def __init__(self, weight: float, scenario_type_loss_weighting: Dict[str, float]):
        """
        Initialize l1 objective for raster reconstruction.
        :param weight: scalar for loss weighting
        :param scenario_type_loss_weighting: scenario-type specific loss weights (ignored)
        """
        self._weight = weight
        self._scenario_type_loss_weighting = scenario_type_loss_weighting

    def compute(
        self, predictions: FeaturesType, targets: TargetsType, matchings: TargetsType, scenarios: ScenarioListType
    ) -> Dict[str, torch.Tensor]:

        gt_raster: SledgeRaster = targets["sledge_raster"]
        pred_raster: SledgeRaster = predictions["sledge_raster"]

        l1_loss = F.l1_loss(gt_raster.data, pred_raster.data)

        return {"l1_loss": self._weight * l1_loss}


class VAEBCEObjective(AbstractCustomObjective):
    """Object for image reconstruction loss (ie. binary cross-entropy)."""

    def __init__(self, weight: float, scenario_type_loss_weighting: Dict[str, float]):
        """
        Initialize binary cross-entropy objective for raster reconstruction.
        :param weight: scalar for loss weighting
        :param scenario_type_loss_weighting: scenario-type specific loss weights (ignored)
        """
        self._weight = weight
        self._scenario_type_loss_weighting = scenario_type_loss_weighting

    def compute(
        self, predictions: FeaturesType, targets: TargetsType, matchings: TargetsType, scenarios: ScenarioListType
    ) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        pred_raster: SledgeRaster = predictions["sledge_raster"]
        gt_raster: SledgeRaster = targets["sledge_raster"]
        bce_loss = F.binary_cross_entropy_with_logits(pred_raster.data, gt_raster.data)

        return {"bce_loss": self._weight * bce_loss}
