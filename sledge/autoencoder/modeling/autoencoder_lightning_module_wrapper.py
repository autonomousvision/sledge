import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import pytorch_lightning as pl

from nuplan.planning.script.builders.lr_scheduler_builder import build_lr_scheduler
from nuplan.planning.training.modeling.objectives.abstract_objective import aggregate_objectives
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType

from sledge.autoencoder.modeling.autoencoder_torch_module_wrapper import AutoencoderTorchModuleWrapper
from sledge.autoencoder.modeling.matching.abstract_matching import AbstractMatching
from sledge.autoencoder.modeling.metrics.abstract_custom_metric import AbstractCustomMetric
from sledge.autoencoder.modeling.objectives.abstract_custom_objective import AbstractCustomObjective

logger = logging.getLogger(__name__)


class AutoencoderLightningModuleWrapper(pl.LightningModule):
    """
    Custom lightning module that wraps the training/validation/testing procedure and handles the objective/metric computation.
    """

    def __init__(
        self,
        model: AutoencoderTorchModuleWrapper,
        objectives: List[AbstractCustomObjective],
        metrics: Optional[List[AbstractCustomMetric]],
        matchings: Optional[List[AbstractMatching]],
        optimizer: Optional[DictConfig] = None,
        lr_scheduler: Optional[DictConfig] = None,
        warm_up_lr_scheduler: Optional[DictConfig] = None,
        objective_aggregate_mode: str = "sum",
    ) -> None:
        """
        Initialize lightning autoencoder wrapper.
        :param model: autoencoder torch module wrapper.
        :param objectives: list of autoencoder objectives computed at each step
        :param metrics: optional list of metrics to track
        :param matchings: optional list of matching objects (e.g. for hungarian objectives)
        :param optimizer: config for instantiating optimizer. Can be 'None' for older models
        :param lr_scheduler: config for instantiating lr_scheduler. Can be 'None' for older models and when an lr_scheduler is not being used.
        :param warm_up_lr_scheduler: _description_, defaults to None
        :param objective_aggregate_mode: how should different objectives be combined, can be 'sum', 'mean', and 'max'.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.objectives = objectives
        self.metrics = metrics
        self.matchings = matchings
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.warm_up_lr_scheduler = warm_up_lr_scheduler
        self.objective_aggregate_mode = objective_aggregate_mode

    def _step(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str) -> torch.Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        """
        features, targets, scenarios = batch

        predictions = self.forward(features)
        matchings = self._compute_matchings(predictions, targets)
        objectives = self._compute_objectives(predictions, targets, matchings, scenarios)
        metrics = self._compute_metrics(predictions, targets, matchings, scenarios)
        loss = aggregate_objectives(objectives, agg_mode=self.objective_aggregate_mode)

        self._log_step(loss, objectives, metrics, prefix)

        return loss

    def _compute_objectives(
        self, predictions: FeaturesType, targets: TargetsType, matchings: TargetsType, scenarios: ScenarioListType
    ) -> Dict[str, torch.Tensor]:
        """
        Computes a set of learning objectives used for supervision given the model's predictions and targets.

        :param predictions: dictionary of predicted dataclasses.
        :param targets: dictionary of target dataclasses.
        :param matchings: dictionary of prediction-target matching.
        :param scenarios: list of scenario types (for adaptive weighting)
        :return: dictionary of objective names and values
        """
        objectives_dict: Dict[str, torch.Tensor] = {}
        for objective in self.objectives:
            objectives_dict.update(objective.compute(predictions, targets, matchings, scenarios))
        return objectives_dict

    def _compute_metrics(
        self, predictions: FeaturesType, targets: TargetsType, matchings: TargetsType, scenarios: ScenarioListType
    ) -> Dict[str, torch.Tensor]:
        """
        Computes a set of metrics used for logging.

        :param predictions: dictionary of predicted dataclasses.
        :param targets: dictionary of target dataclasses.
        :param matchings: dictionary of prediction-target matching.
        :param scenarios: list of scenario types (for adaptive weighting)
        :return: dictionary of metrics names and values
        """
        metrics_dict: Dict[str, torch.Tensor] = {}
        if self.metrics:
            for metric in self.metrics:
                metrics_dict.update(metric.compute(predictions, targets, matchings, scenarios))
        return metrics_dict

    def _compute_matchings(self, predictions: FeaturesType, targets: TargetsType) -> FeaturesType:
        """
        Computes a the matchings (e.g. for hungarian loss) between prediction and targets.

        :param predictions: dictionary of predicted dataclasses.
        :param targets: dictionary of target dataclasses.
        :return: dictionary of matching names and matching dataclasses
        """
        matchings_dict: Dict[str, torch.Tensor] = {}
        if self.matchings:
            for matching in self.matchings:
                matchings_dict.update(matching.compute(predictions, targets))
        return matchings_dict

    def _log_step(
        self,
        loss: torch.Tensor,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = "loss",
    ) -> None:
        """
        Logs the artifacts from a training/validation/test step.

        :param loss: scalar loss value
        :type objectives: [type]
        :param metrics: dictionary of metrics names and values
        :param prefix: prefix prepended at each artifact's name
        :param loss_name: name given to the loss for logging
        """
        self.log(f"loss/{prefix}_{loss_name}", loss)

        for key, value in objectives.items():
            self.log(f"objectives/{prefix}_{key}", value)

        if self.metrics:
            for key, value in metrics.items():
                self.log(f"metrics/{prefix}_{key}", value)

    def training_step(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "train")

    def validation_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "val")

    def test_step(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "test")

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        return self.model(features)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        if self.optimizer is None:
            raise RuntimeError("To train, optimizer must not be None.")

        # Get optimizer
        optimizer: Optimizer = instantiate(
            config=self.optimizer,
            params=self.parameters(),
            lr=self.optimizer.lr,  # Use lr found from lr finder; otherwise use optimizer config
        )
        # Log the optimizer used
        logger.info(f"Using optimizer: {self.optimizer._target_}")

        # Get lr_scheduler
        lr_scheduler_params: Dict[str, Union[_LRScheduler, str, int]] = build_lr_scheduler(
            optimizer=optimizer,
            lr=self.optimizer.lr,
            warm_up_lr_scheduler_cfg=self.warm_up_lr_scheduler,
            lr_scheduler_cfg=self.lr_scheduler,
        )
        lr_scheduler_params["interval"] = "step"
        lr_scheduler_params["frequency"] = 1

        optimizer_dict: Dict[str, Any] = {}
        optimizer_dict["optimizer"] = optimizer
        if lr_scheduler_params:
            logger.info(f"Using lr_schedulers {lr_scheduler_params}")
            optimizer_dict["lr_scheduler"] = lr_scheduler_params

        return optimizer_dict if "lr_scheduler" in optimizer_dict else optimizer_dict["optimizer"]
