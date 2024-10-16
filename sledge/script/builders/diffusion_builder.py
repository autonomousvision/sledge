import logging
import os
from shutil import rmtree
from pathlib import Path
from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
import torch.nn as nn

import accelerate
import diffusers
from datasets import load_dataset

from nuplan.planning.script.builders.utils.utils_type import validate_type

from sledge.diffusion.modelling.ldm_pipeline import LDMPipeline
from sledge.script.builders.model_builder import build_autoencoder_torch_module_wrapper


logger = logging.getLogger(__name__)


def build_accelerator(cfg: DictConfig) -> accelerate.Accelerator:
    """
    Build accelerator from config file.
    :param cfg: Omegaconf dictionary
    :return: accelerator
    """
    kwargs_handlers = []
    logger.info("Building Accelerator...")
    accelerator: accelerate.Accelerator = instantiate(config=cfg.accelerator, kwargs_handlers=kwargs_handlers)
    validate_type(accelerator, accelerate.Accelerator)
    logger.info("Building Accelerator...DONE!")
    return accelerator


def build_diffusion_model(cfg: DictConfig) -> diffusers.DiTTransformer2DModel:
    """
    Build diffusion model from config file.
    :param cfg: Omegaconf dictionary
    :return: diffusion model
    """
    logger.info("Building Diffusion Model...")
    diffusion_model: diffusers.DiTTransformer2DModel = instantiate(cfg.diffusion_model)
    validate_type(diffusion_model, diffusers.DiTTransformer2DModel)
    logger.info("Building Diffusion Model...DONE!")
    return diffusion_model


def build_noise_scheduler(cfg: DictConfig) -> diffusers.SchedulerMixin:
    """
    Build noise scheduler for diffusion training from config file.
    :param cfg: Omegaconf dictionary
    :return: noise scheduler
    """
    logger.info("Building Noise Scheduler...")
    noise_scheduler: diffusers.SchedulerMixin = instantiate(cfg.noise_scheduler)
    validate_type(noise_scheduler, diffusers.SchedulerMixin)
    logger.info("Building Noise Scheduler...DONE!")
    return noise_scheduler


def build_optimizer(cfg: DictConfig, diffusion_model: nn.Module) -> torch.optim.Optimizer:
    """
    Build torch optimizer from config file.
    :param cfg: Omegaconf dictionary
    :return: torch optimizer
    """
    logger.info("Building Optimizer...")
    optimizer: torch.optim.Optimizer = instantiate(config=cfg.optimizer, params=diffusion_model.parameters())
    validate_type(optimizer, torch.optim.Optimizer)
    logger.info("Building Optimizer...DONE!")
    return optimizer


def build_dataset(cfg: DictConfig) -> torch.utils.data.Dataset:
    """
    Build torch dataset for diffusion training from config file.
    :param cfg: Omegaconf dictionary
    :return: torch dataset
    """

    logger.info("Building Dataset...")
    if cfg.cache.cleanup_diffusion_cache and Path(cfg.cache.diffusion_cache_path).exists():
        logger.info("Deleting existing dataset...")
        rmtree(cfg.cache.diffusion_cache_path)
        logger.info("Deleting existing dataset...DONE!")

    dataset_file_path = Path(os.getenv("SLEDGE_DEVKIT_ROOT")) / "sledge/diffusion/dataset/rvae_latent_dataset.py"
    dataset = load_dataset(
        path=str(dataset_file_path),
        data_dir=cfg.cache.autoencoder_cache_path,
        cache_dir=cfg.cache.diffusion_cache_path,
        split="train",
        trust_remote_code=True,
    )

    def transform_images(examples):
        images = [torch.as_tensor(image, dtype=torch.float32) for image in examples["features"]]
        labels = examples["label"]
        return {"input": images, "label": labels}

    dataset.set_transform(transform_images)
    logger.info("Building Dataset...DONE!")
    return dataset


def build_dataloader(cfg: DictConfig, dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
    """
    Build torch dataloader for diffusion training from config file.
    :param cfg: Omegaconf dictionary
    :return: torch dataloader
    """

    # TODO: remove
    logger.info("Building Dataloader...")
    dataloader = torch.utils.data.DataLoader(dataset, **cfg.data_loader.params)
    logger.info("Building Dataloader...DONE!")
    return dataloader


def build_lr_scheduler(
    cfg: DictConfig, optimizer: torch.optim.Optimizer, num_training_steps: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Build learning rate scheduler for diffusion training from config file.
    :param cfg: Omegaconf dictionary
    :return: torch learning rate scheduler
    """

    logger.info("Building LR Scheduler...DONE!")
    lr_scheduler = diffusers.optimization.get_scheduler(
        cfg.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_scheduler.num_warmup_steps * cfg.accelerator.gradient_accumulation_steps,
        num_training_steps=num_training_steps,
    )
    logger.info("Building LR Scheduler...DONE!")
    return lr_scheduler


def build_decoder(cfg: DictConfig) -> nn.Module:
    """
    Build decoder from autoencoder for diffusion training from config file.
    :param cfg: Omegaconf dictionary
    :return: decoder as torch module
    """

    logger.info("Building Decoder...DONE!")
    assert cfg.autoencoder_checkpoint is not None, "cfg.autoencoder_checkpoint is not specified!"
    torch_module_wrapper = build_autoencoder_torch_module_wrapper(cfg)
    decoder = torch_module_wrapper.get_decoder()
    logger.info("Building Decoder...DONE!")

    return decoder


def build_pipeline_from_checkpoint(cfg: DictConfig) -> LDMPipeline:
    """
    Build latent diffusion pipeline from config file.
    :param cfg: Omegaconf dictionary
    :return: latent diffusion pipeline
    """

    logger.info("Building LDMPipeline...")
    assert cfg.diffusion_checkpoint is not None, "cfg.diffusion_checkpoint is not specified!"
    pipeline = LDMPipeline.from_pretrained(cfg.diffusion_checkpoint, use_safetensors=True)
    logger.info(f"Load from checkpoint {cfg.autoencoder_checkpoint}...DONE!")
    logger.info("Building LDMPipeline...DONE!")

    return pipeline
