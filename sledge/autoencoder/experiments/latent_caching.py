import logging
from pathlib import Path
from omegaconf import DictConfig

import torch
from tqdm import tqdm

from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle

from sledge.script.builders.model_builder import build_autoencoder_torch_module_wrapper
from sledge.script.builders.autoencoder_builder import build_autoencoder_lightning_datamodule
from sledge.autoencoder.preprocessing.features.latent_feature import Latent
from sledge.autoencoder.modeling.autoencoder_lightning_module_wrapper import AutoencoderLightningModuleWrapper

logger = logging.getLogger(__name__)


def cache_latent(cfg: DictConfig, worker: WorkerPool) -> None:
    """
    Build the lightning datamodule and cache the latent of all training samples.
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    """

    assert cfg.autoencoder_checkpoint is not None, "cfg.autoencoder_checkpoint is not specified for latent caching!"

    # Create model
    logger.info("Building Autoencoder Module...")
    torch_module_wrapper = build_autoencoder_torch_module_wrapper(cfg)
    torch_module_wrapper = AutoencoderLightningModuleWrapper.load_from_checkpoint(
        cfg.autoencoder_checkpoint, model=torch_module_wrapper
    ).model
    logger.info("Building Autoencoder Module...DONE!")

    # Build the datamodule
    logger.info("Building Datamodule Module...")
    datamodule = build_autoencoder_lightning_datamodule(cfg, worker, torch_module_wrapper)
    datamodule.setup("fit")
    dataloader = datamodule.train_dataloader()
    logger.info("Building Datamodule Module...DONE!")

    autoencoder_cache_path = Path(cfg.cache.autoencoder_cache_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    storing_mechanism = FeatureCachePickle()
    torch_module_wrapper = torch_module_wrapper.to(device)

    # Perform inference
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="Cache Latents (batch-wise)"):
            # Assuming batch is a tuple of (inputs, labels, indices) where indices track sample order
            features, targets, scenarios = datamodule.transfer_batch_to_device(batch, device, 0)
            predictions = torch_module_wrapper.forward(features, encode_only=True)
            assert "latent" in predictions

            latents: Latent = predictions["latent"]
            latents = latents.torch_to_numpy()

            for latent, scenario in zip(latents.unpack(), scenarios):
                file_name = (
                    autoencoder_cache_path
                    / scenario.log_name
                    / scenario.scenario_type
                    / scenario.token
                    / cfg.cache.latent_name
                )
                storing_mechanism.store_computed_feature_to_folder(file_name, latent)

    return None
