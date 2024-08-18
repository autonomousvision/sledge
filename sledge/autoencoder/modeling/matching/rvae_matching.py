import torch
from scipy.optimize import linear_sum_assignment

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType

from sledge.autoencoder.modeling.matching.abstract_matching import AbstractMatching
from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig
from sledge.autoencoder.preprocessing.features.rvae_matching_feature import RVAEMatchingFeature
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeVectorElement,
    SledgeVectorElementType,
    BoundingBoxIndex,
)


class RVAEHungarianMatching(AbstractMatching):
    """Object for hungarian matching of RVAE model"""

    def __init__(self, key: str, config: RVAEConfig):
        """
        Initialize matching object of RVAE
        :param key: string identifier if sledge vector dataclass
        :param config: config dataclass of RVAE
        """

        self._key = key
        self._config = config

    @torch.no_grad()
    def compute(self, predictions: FeaturesType, targets: TargetsType) -> TargetsType:
        """Inherited from superclass."""

        pred_vector_element: SledgeVectorElement = getattr(predictions["sledge_vector"], self._key)
        gt_vector_element: SledgeVectorElement = getattr(targets["sledge_vector"], self._key)

        element_type = pred_vector_element.get_element_type()
        assert element_type == gt_vector_element.get_element_type()
        assert element_type in [
            SledgeVectorElementType.LINE,
            SledgeVectorElementType.AGENT,
            SledgeVectorElementType.STATIC,
        ]

        gt_states, gt_mask = gt_vector_element.states, gt_vector_element.mask
        pred_states, pred_logits = pred_vector_element.states, pred_vector_element.mask

        ce_cost = _get_ce_cost(gt_mask, pred_logits)

        if element_type == SledgeVectorElementType.LINE:
            l1_cost = _get_line_l1_cost(gt_states, pred_states, gt_mask)
            ce_weight, reconstruction_weight = self._config.line_ce_weight, self._config.line_reconstruction_weight
        else:
            l1_cost = _get_box_l1_cost(gt_states, pred_states, gt_mask, pred_vector_element.get_element_index())
            ce_weight, reconstruction_weight = self._config.box_ce_weight, self._config.box_reconstruction_weight

        cost = ce_weight * ce_cost + reconstruction_weight * l1_cost
        cost = cost.cpu()  # NOTE: This unfortunately is the runtime bottleneck

        indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
        matching = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        return {f"{self._key}_matching": RVAEMatchingFeature(matching)}


@torch.no_grad()
def _get_ce_cost(gt_mask: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Calculated cross-entropy matching cost based on numerically stable PyTorch version, see:
    https://github.com/pytorch/pytorch/blob/c64e006fc399d528bb812ae589789d0365f3daf4/aten/src/ATen/native/Loss.cpp#L214
    :param gt_mask: ground-truth binary existence labels, shape: (batch, num_gt)
    :param pred_logits: predicted (normalized) logits of existence, shape: (batch, num_pred)
    :return: cross-entropy cost tensor of shape (batch, num_pred, num_gt)
    """

    gt_mask_expanded = gt_mask[:, :, None].detach().float()  # (b, ng, 1)
    pred_logits_expanded = pred_logits[:, None, :].detach()  # (b, 1, np)

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val))
    ce_cost = (1 - gt_mask_expanded) * pred_logits_expanded + helper_term  # (b, ng, np)
    ce_cost = ce_cost.permute(0, 2, 1)  # (b, np, ng)

    return ce_cost


@torch.no_grad()
def _get_line_l1_cost(gt_states: torch.Tensor, pred_states: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """
    Calculates the L1 matching cost for line state tensors.
    :param gt_states: ground-truth line tensor, shape: (batch, num_gt, state_size)
    :param pred_states: predicted line tensor, shape: (batch, num_pred, state_size)
    :param gt_mask: ground-truth binary existence labels for masking, shape: (batch, num_gt)
    :return: L1 cost tensor of shape (batch, num_pred, num_gt)
    """

    gt_states_expanded = gt_states[:, :, None].detach()  # (b, ng, 1, *s)
    pred_states_expanded = pred_states[:, None].detach()  # (b, 1, np, *s)
    l1_cost = gt_mask[..., None] * (gt_states_expanded - pred_states_expanded).abs().sum(dim=-1).mean(dim=-1)
    l1_cost = l1_cost.permute(0, 2, 1)  # (b, np, ng)

    return l1_cost


@torch.no_grad()
def _get_box_l1_cost(
    gt_states: torch.Tensor, pred_states: torch.Tensor, gt_mask: torch.Tensor, object_indexing: BoundingBoxIndex
) -> torch.Tensor:
    """
    Calculates the L1 matching cost for bounding box state tensors, based on the (x,y) position.
    :param gt_states: ground-truth box tensor, shape: (batch, num_gt, state_size)
    :param pred_states: predicted box tensor, shape: (batch, num_pred, state_size)
    :param gt_mask: ground-truth binary existence labels for masking, shape: (batch, num_gt)
    :param object_indexing: index enum of object type.
    :return: L1 cost tensor of shape (batch, num_pred, num_gt)
    """

    # NOTE: Bounding Box L1 matching only considers position, ignoring irrelevant attr. (e.g. box extent)
    gt_states_expanded = gt_states[:, :, None, object_indexing.POINT].detach()  # (b, ng, 1, 2)
    pred_states_expanded = pred_states[:, None, :, object_indexing.POINT].detach()  # (b, 1, np, 2)
    l1_cost = gt_mask[..., None] * (gt_states_expanded - pred_states_expanded).abs().sum(dim=-1)  # (b, ng, np)
    l1_cost = l1_cost.permute(0, 2, 1)  # (b, np, ng)

    return l1_cost
