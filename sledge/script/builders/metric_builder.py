import logging
from typing import List

from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.planning.script.builders.utils.utils_type import validate_type

from sledge.autoencoder.modeling.metrics.abstract_custom_metric import AbstractCustomMetric
from sledge.autoencoder.modeling.objectives.abstract_custom_objective import AbstractCustomObjective

logger = logging.getLogger(__name__)


def build_custom_training_metrics(cfg: DictConfig) -> List[AbstractCustomMetric]:
    """
    Build objectives based on config
    :param cfg: config
    :return list of objectives.
    """
    instantiated_metrics = []

    scenario_type_loss_weighting = (
        cfg.scenario_type_weights.scenario_type_loss_weights
        if ("scenario_type_weights" in cfg and "scenario_type_loss_weights" in cfg.scenario_type_weights)
        else {}
    )
    for metric_name, metric_type in cfg.training_metric.items():
        new_metric: AbstractCustomMetric = instantiate(
            metric_type, scenario_type_loss_weighting=scenario_type_loss_weighting
        )
        validate_type(new_metric, AbstractCustomMetric)
        instantiated_metrics.append(new_metric)
    return instantiated_metrics


def build_custom_objectives(cfg: DictConfig) -> List[AbstractCustomObjective]:
    """
    Build objectives based on config
    :param cfg: config
    :return list of objectives.
    """
    instantiated_objectives = []

    scenario_type_loss_weighting = (
        cfg.scenario_type_weights.scenario_type_loss_weights
        if ("scenario_type_weights" in cfg and "scenario_type_loss_weights" in cfg.scenario_type_weights)
        else {}
    )
    for objective_name, objective_type in cfg.objective.items():
        new_objective: AbstractCustomObjective = instantiate(
            objective_type, scenario_type_loss_weighting=scenario_type_loss_weighting
        )
        validate_type(new_objective, AbstractCustomObjective)
        instantiated_objectives.append(new_objective)
    return instantiated_objectives
