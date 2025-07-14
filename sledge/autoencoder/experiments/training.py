import logging
import os
from typing import cast

import pytorch_lightning as pl
import torch
from nuplan.planning.script.builders.utils.utils_type import is_target_type
from nuplan.planning.training.experiments.training import TrainingEngine
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from omegaconf import DictConfig
from torch.optim.lr_scheduler import OneCycleLR

from sledge.script.builders.autoencoder_builder import (
    build_autoencoder_lightning_datamodule,
    build_autoencoder_lightning_module,
    build_autoencoder_trainer,
)
from sledge.script.builders.model_builder import build_autoencoder_torch_module_wrapper

logger = logging.getLogger(__name__)


def build_training_engine(cfg: DictConfig, worker: WorkerPool) -> TrainingEngine:
    """
    Build the three core lightning modules: LightningDataModule, LightningModule and Trainer
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    :return: TrainingEngine
    """
    logger.info("Building training engine...")

    # Create model
    torch_module_wrapper = build_autoencoder_torch_module_wrapper(cfg)

    # Build the datamodule
    datamodule = build_autoencoder_lightning_datamodule(cfg, worker, torch_module_wrapper)

    cfg = scale_cfg_for_distributed_training(cfg, datamodule=datamodule)

    # Build lightning module
    model = build_autoencoder_lightning_module(cfg, torch_module_wrapper)

    # Build trainer
    trainer = build_autoencoder_trainer(cfg)

    engine = TrainingEngine(trainer=trainer, datamodule=datamodule, model=model)

    return engine


def scale_cfg_for_distributed_training(cfg: DictConfig, datamodule: pl.LightningDataModule) -> DictConfig:
    """
    Adjusts parameters in cfg for ddp.
    :param cfg: Config with parameters for instantiation.
    :param datamodule: Datamodule which will be used for updating the lr_scheduler parameters.
    :return cfg: Updated config.
    """
    number_devices = get_num_devices(cfg)

    # Setup learning rate and momentum schedulers
    if is_target_type(cfg.lr_scheduler, OneCycleLR):
        datamodule.setup("fit")
        steps_per_epoch = len(datamodule.train_dataloader()) // number_devices
        cfg.lr_scheduler.steps_per_epoch = steps_per_epoch

    return cfg


def get_num_devices(cfg: DictConfig) -> int:
    num_devices = os.getenv("WORLD_SIZE", -1)

    if num_devices == -1:  # if environment variable WORLD_SIZE is not set, find from trainer
        logger.info("WORLD_SIZE was not set.")
        trainer_num_devices = cfg.lightning.trainer.params.devices

        if isinstance(trainer_num_devices, str):
            raise RuntimeError("Error, please specify gpus as integer. Received string.")
        trainer_num_devices = cast(int, trainer_num_devices)

        if trainer_num_devices == -1:  # if trainer gpus = -1, all gpus are used, so find all available devices
            logger.info(
                "PytorchLightning Trainer devices was set to -1, finding number of GPUs used from torch.cuda.device_count()."
            )
            cuda_num_devices = torch.cuda.device_count() * int(os.getenv("NUM_NODES", 1))
            num_devices = cuda_num_devices

        else:  # if trainer devices is not -1
            logger.info(f"Trainer devices was set to {trainer_num_devices}, using this as the number of gpus.")
            num_devices = trainer_num_devices

    num_devices = int(num_devices)
    logger.info(f"Number of devices found to be in use: {num_devices}")
    return num_devices
