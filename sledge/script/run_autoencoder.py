import logging
import os
from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from nuplan.planning.script.builders.folder_builder import build_training_experiment_folder
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.profiler_context_manager import ProfilerContextManager
from nuplan.planning.script.utils import set_default_path
from nuplan.planning.training.experiments.training import TrainingEngine

from sledge.autoencoder.experiments.training import build_training_engine
from sledge.autoencoder.experiments.latent_caching import cache_latent
from sledge.autoencoder.experiments.feature_caching import cache_feature
from sledge.script.builders.utils.utils_config import update_config_for_autoencoder_training

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = os.getenv("NUPLAN_HYDRA_CONFIG_PATH", "config/autoencoder")

if os.environ.get("NUPLAN_HYDRA_CONFIG_PATH") is not None:
    CONFIG_PATH = os.path.join("../../../../", CONFIG_PATH)

if os.path.basename(CONFIG_PATH) != "autoencoder":
    CONFIG_PATH = os.path.join(CONFIG_PATH, "autoencoder")
CONFIG_NAME = "default_autoencoder"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> Optional[TrainingEngine]:
    """
    Main entrypoint for autoencoder experiments.
    :param cfg: omegaconf dictionary
    """
    # Fix random seed
    pl.seed_everything(cfg.seed, workers=True)

    # Configure logger
    build_logger(cfg)

    # Override configs based on setup, and print config
    update_config_for_autoencoder_training(cfg)

    # Create output storage folder
    build_training_experiment_folder(cfg=cfg)

    # Build worker
    worker = build_worker(cfg)

    if cfg.py_func == "training":
        # Build training engine
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "build_training_engine"):
            engine = build_training_engine(cfg, worker)

        # Run training
        logger.info("Starting training...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "training"):
            engine.trainer.fit(model=engine.model, datamodule=engine.datamodule)
        return engine
    elif cfg.py_func == "feature_caching":
        # Precompute and cache all features
        logger.info("Starting feature caching...")
        if cfg.worker == "ray_distributed" and cfg.worker.use_distributed:
            raise AssertionError("ray in distributed mode will not work with this job")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "caching"):
            cache_feature(cfg=cfg, worker=worker)
        return None
    elif cfg.py_func == "latent_caching":
        # Precompute and cache latents of the autoencoder
        logger.info("Starting latent caching...")
        if cfg.worker == "ray_distributed" and cfg.worker.use_distributed:
            raise AssertionError("ray in distributed mode will not work with this job")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "caching"):
            cache_latent(cfg=cfg, worker=worker)
        return None
    else:
        raise NameError(f"Function {cfg.py_func} does not exist")


if __name__ == "__main__":
    main()
