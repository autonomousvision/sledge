import logging
from pathlib import Path
from shutil import rmtree

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# TODO: maybe remove this function
def update_config_for_autoencoder_training(cfg: DictConfig) -> None:
    """
    Updates the config based on some conditions.
    :param cfg: omegaconf dictionary that is used to run the experiment.
    """
    # Make the configuration editable.
    OmegaConf.set_struct(cfg, False)

    if cfg.cache.autoencoder_cache_path is None:
        logger.warning("Parameter autoencoder_cache_path is not set, caching is disabled")
    else:
        if cfg.cache.cleanup_autoencoder_cache and Path(cfg.cache.autoencoder_cache_path).exists():
            rmtree(cfg.cache.autoencoder_cache_path)

        Path(cfg.cache.autoencoder_cache_path).mkdir(parents=True, exist_ok=True)

    cfg.cache.cache_path = cfg.cache.autoencoder_cache_path  # TODO: remove this workaround
    cfg.lightning.trainer.params.gpus = -1  # TODO: remove this workaround

    # Save all interpolations and remove keys that were only used for interpolation and have no further use.
    OmegaConf.resolve(cfg)

    # Finalize the configuration and make it non-editable.
    OmegaConf.set_struct(cfg, True)

    # Log the final configuration after all overrides, interpolations and updates.
    if cfg.log_config:
        logger.info(f"Creating experiment name [{cfg.experiment}] in group [{cfg.group}] with config...")
        logger.info("\n" + OmegaConf.to_yaml(cfg))
