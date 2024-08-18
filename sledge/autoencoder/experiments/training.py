import logging
from omegaconf import DictConfig

from torch.optim.lr_scheduler import OneCycleLR
import pytorch_lightning as pl

from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.training.experiments.training import TrainingEngine
from nuplan.planning.script.builders.utils.utils_config import get_num_gpus_used
from nuplan.planning.script.builders.utils.utils_type import is_target_type

from sledge.script.builders.model_builder import build_autoencoder_torch_module_wrapper
from sledge.script.builders.autoencoder_builder import (
    build_autoencoder_lightning_datamodule,
    build_autoencoder_lightning_module,
    build_autoencoder_trainer,
)

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

    cfg = scale_cfg_for_distributed_training(cfg, datamodule=datamodule, worker=worker)

    # Build lightning module
    model = build_autoencoder_lightning_module(cfg, torch_module_wrapper)

    # Build trainer
    trainer = build_autoencoder_trainer(cfg)

    engine = TrainingEngine(trainer=trainer, datamodule=datamodule, model=model)

    return engine


def scale_cfg_for_distributed_training(
    cfg: DictConfig, datamodule: pl.LightningDataModule, worker: WorkerPool
) -> DictConfig:
    """
    Adjusts parameters in cfg for ddp.
    :param cfg: Config with parameters for instantiation.
    :param datamodule: Datamodule which will be used for updating the lr_scheduler parameters.
    :return cfg: Updated config.
    """
    number_gpus = get_num_gpus_used(cfg)

    # Setup learning rate and momentum schedulers
    if is_target_type(cfg.lr_scheduler, OneCycleLR):
        num_train_samples = int(
            len(datamodule._splitter.get_train_samples(datamodule._all_samples, worker)) * datamodule._train_fraction
        )

        cfg.lr_scheduler.steps_per_epoch = (num_train_samples // cfg.data_loader.params.batch_size) // number_gpus

    return cfg
