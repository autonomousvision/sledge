import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.planning.script.builders.utils.utils_type import validate_type

from sledge.autoencoder.modeling.autoencoder_torch_module_wrapper import AutoencoderTorchModuleWrapper
from sledge.autoencoder.modeling.autoencoder_lightning_module_wrapper import AutoencoderLightningModuleWrapper

logger = logging.getLogger(__name__)


def build_autoencoder_torch_module_wrapper(cfg: DictConfig) -> AutoencoderTorchModuleWrapper:
    """
    Builds the autoencoder module.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of AutoencoderTorchModuleWrapper.
    """
    logger.info("Building AutoencoderTorchModuleWrapper...")
    model = instantiate(cfg.autoencoder_model)
    validate_type(model, AutoencoderTorchModuleWrapper)
    if cfg.autoencoder_checkpoint:
        model = AutoencoderLightningModuleWrapper.load_from_checkpoint(cfg.autoencoder_checkpoint, model=model).model
        logger.info(f"Load from checkpoint {cfg.autoencoder_checkpoint}...DONE!")
    logger.info("Building AutoencoderTorchModuleWrapper...DONE!")

    return model
