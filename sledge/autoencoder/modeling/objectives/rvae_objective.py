from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType

from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig
from sledge.autoencoder.modeling.objectives.abstract_custom_objective import AbstractCustomObjective
from sledge.autoencoder.preprocessing.features.rvae_matching_feature import RVAEMatchingFeature
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVectorElement, SledgeVectorElementType


class RVAEHungarianObjective(AbstractCustomObjective):
    """Object for hungarian loss (ie. lw + bce) for RVAE model."""

    def __init__(self, key: str, config: RVAEConfig, scenario_type_loss_weighting: Dict[str, float]):
        """
        Initialize hungarian loss object.
        :param key: string identifier if sledge vector dataclass
        :param config: config dataclass of RVAE
        :param scenario_type_loss_weighting: scenario-type specific loss weights (ignored)
        """
        self._key = key
        self._config = config
        self._scenario_type_loss_weighting = scenario_type_loss_weighting

    def compute(
        self, predictions: FeaturesType, targets: TargetsType, matchings: TargetsType, scenarios: ScenarioListType
    ) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        # Retrieve all relevant predictions, targets, matchings
        pred_vector_element: SledgeVectorElement = getattr(predictions["sledge_vector"], self._key)
        gt_vector_element: SledgeVectorElement = getattr(targets["sledge_vector"], self._key)

        element_type = pred_vector_element.get_element_type()
        assert element_type == gt_vector_element.get_element_type()
        assert element_type in [
            SledgeVectorElementType.LINE,
            SledgeVectorElementType.AGENT,
            SledgeVectorElementType.STATIC,
        ]

        matching: RVAEMatchingFeature = matchings[f"{self._key}_matching"]

        # Unpack predictions and targets
        gt_states, gt_mask = gt_vector_element.states, gt_vector_element.mask
        pred_states, pred_logits = pred_vector_element.states, pred_vector_element.mask

        # Arrange predictions and targets according to matching
        indices, permutation_indices = matching.indices, _get_src_permutation_idx(matching.indices)

        pred_states_idx = pred_states[permutation_indices]
        gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

        pred_logits_idx = pred_logits[permutation_indices]
        gt_mask_idx = torch.cat([t[i] for t, (_, i) in zip(gt_mask, indices)], dim=0).float()

        # calculate CE and L1 Loss
        l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
        if element_type == SledgeVectorElementType.LINE:
            l1_loss = l1_loss.sum(-1).mean(-1) * gt_mask_idx
            ce_weight, reconstruction_weight = self._config.line_ce_weight, self._config.line_reconstruction_weight
        else:
            l1_loss = l1_loss.sum(-1) * gt_mask_idx
            ce_weight, reconstruction_weight = self._config.box_ce_weight, self._config.box_reconstruction_weight

        ce_loss = F.binary_cross_entropy_with_logits(pred_logits_idx, gt_mask_idx, reduction="none")

        # Whether to average by batch size or entity count
        bs = gt_mask.shape[0]
        if self._config.norm_by_count:
            num_gt_instances = gt_mask.float().sum(-1)
            num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1
            l1_loss = l1_loss.view(bs, -1).sum() / num_gt_instances
            ce_loss = ce_loss.view(bs, -1).sum() / num_gt_instances
        else:
            l1_loss = l1_loss.view(bs, -1).mean()
            ce_loss = ce_loss.view(bs, -1).mean()

        return {f"l1_{self._key}": reconstruction_weight * l1_loss, f"ce_{self._key}": ce_weight * ce_loss}


class RVAEEgoObjective(AbstractCustomObjective):
    """Simple regression loss of ego attributes (ie. lw + bce)."""

    def __init__(self, weight: float, scenario_type_loss_weighting: Dict[str, float]):
        """
        Initialize ego objective.
        :param weight: scalar for loss weighting
        :param scenario_type_loss_weighting: scenario-type specific loss weights (ignored)
        """
        self._weight = weight
        self._scenario_type_loss_weighting = scenario_type_loss_weighting

    def compute(
        self, predictions: FeaturesType, targets: TargetsType, matchings: TargetsType, scenarios: ScenarioListType
    ) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        pred_ego_element: SledgeVectorElement = predictions["sledge_vector"].ego
        gt_ego_element: SledgeVectorElement = targets["sledge_vector"].ego
        l1_loss = F.l1_loss(pred_ego_element.states, gt_ego_element.states[..., 0])

        return {"l1_ego": self._weight * l1_loss}


def _get_src_permutation_idx(indices) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper function for permutation of matched indices."""

    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])

    return batch_idx, src_idx
