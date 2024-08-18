import logging
from typing import List, cast

import pytorch_lightning as pl

from omegaconf import DictConfig
from hydra.utils import instantiate

from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor
from nuplan.planning.script.builders.data_augmentation_builder import build_agent_augmentor
from nuplan.planning.script.builders.scenario_builder import build_scenarios
from nuplan.planning.script.builders.splitter_builder import build_splitter
from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool

from sledge.autoencoder.modeling.autoencoder_torch_module_wrapper import AutoencoderTorchModuleWrapper
from sledge.autoencoder.modeling.autoencoder_lightning_module_wrapper import AutoencoderLightningModuleWrapper
from sledge.autoencoder.data_loader.autoencoder_datamodule import AutoencoderDataModule
from sledge.script.builders.metric_builder import build_custom_training_metrics, build_custom_objectives
from sledge.script.builders.matching_builder import build_matching

logger = logging.getLogger(__name__)


def build_autoencoder_lightning_datamodule(
    cfg: DictConfig, worker: WorkerPool, model: AutoencoderTorchModuleWrapper
) -> pl.LightningDataModule:
    """
    Build the autoencoder lightning datamodule for SLEDGE from the config.
    :param cfg: Omegaconf dictionary.
    :param model: NN model used for training.
    :param worker: Worker to submit tasks which can be executed in parallel.
    :return: Instantiated datamodule object.
    """
    # Build features and targets
    feature_builders = model.get_list_of_required_feature()
    target_builders = model.get_list_of_computed_target()

    # Build splitter
    splitter = build_splitter(cfg.splitter)

    # Create feature preprocessor
    feature_preprocessor = FeaturePreprocessor(
        cache_path=cfg.cache.autoencoder_cache_path,
        force_feature_computation=cfg.cache.force_feature_computation,
        feature_builders=feature_builders,
        target_builders=target_builders,
    )

    # Create data augmentation
    augmentors = build_agent_augmentor(cfg.data_augmentation) if "data_augmentation" in cfg else None

    # Build dataset scenarios
    scenarios = build_scenarios(cfg, worker, model)

    # Create custom datamodule (always applies data augmentation for pre-processing)
    datamodule: pl.LightningDataModule = AutoencoderDataModule(
        feature_preprocessor=feature_preprocessor,
        splitter=splitter,
        all_scenarios=scenarios,
        dataloader_params=cfg.data_loader.params,
        augmentors=augmentors,
        worker=worker,
        scenario_type_sampling_weights=cfg.scenario_type_weights.scenario_type_sampling_weights,
        **cfg.data_loader.datamodule,
    )

    return datamodule


def build_autoencoder_lightning_module(
    cfg: DictConfig, torch_module_wrapper: AutoencoderTorchModuleWrapper
) -> pl.LightningModule:
    """
    Builds the lightning module from the config.
    :param cfg: omegaconf dictionary
    :param torch_module_wrapper: NN model used for training
    :return: built object.
    """
    # Build loss
    objectives = build_custom_objectives(cfg)

    # Build metrics to evaluate the performance of predictions
    metrics = build_custom_training_metrics(cfg) if "training_metric" in cfg else None

    # Build matcher, e.g. used for DETR-style autoencoder in SLEDGE
    matchings = build_matching(cfg.matching) if "matching" in cfg else None

    # Create the complete Module
    model = AutoencoderLightningModuleWrapper(
        model=torch_module_wrapper,
        objectives=objectives,
        metrics=metrics,
        matchings=matchings,
        optimizer=cfg.optimizer,
        lr_scheduler=cfg.lr_scheduler if "lr_scheduler" in cfg else None,
        warm_up_lr_scheduler=cfg.warm_up_lr_scheduler if "warm_up_lr_scheduler" in cfg else None,
        objective_aggregate_mode=cfg.objective_aggregate_mode,
    )

    return cast(pl.LightningModule, model)


def build_callbacks(cfg: DictConfig) -> List[pl.Callback]:
    """
    Build callbacks based on config.
    :param cfg: Dict config.
    :return List of callbacks.
    """
    logger.info("Building callbacks...")

    instantiated_callbacks = []
    for callback_type in cfg.callbacks.values():
        callback: pl.Callback = instantiate(callback_type)
        validate_type(callback, pl.Callback)
        instantiated_callbacks.append(callback)

    logger.info("Building callbacks...DONE!")

    return instantiated_callbacks


def build_autoencoder_trainer(cfg: DictConfig) -> pl.Trainer:
    """
    Builds the lightning trainer from the config.
    :param cfg: omegaconf dictionary
    :return: built object.
    """
    params = cfg.lightning.trainer.params

    callbacks = build_callbacks(cfg)

    loggers = [
        pl.loggers.TensorBoardLogger(
            save_dir=cfg.group,
            name=cfg.experiment,
            log_graph=False,
            version="",
            prefix="",
        ),
    ]

    # TODO: remove this workaround
    del cfg.lightning.trainer.params.gpus

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        **params,
    )

    return trainer
