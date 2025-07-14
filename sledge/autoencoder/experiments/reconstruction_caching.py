import logging
from pathlib import Path

import torch
from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from omegaconf import DictConfig
from tqdm import tqdm

from sledge.autoencoder.modeling.autoencoder_lightning_module_wrapper import AutoencoderLightningModuleWrapper
from sledge.autoencoder.preprocessing.features.graph_feature import GraphConfig
from sledge.script.builders.autoencoder_builder import build_autoencoder_lightning_datamodule
from sledge.script.builders.model_builder import build_autoencoder_torch_module_wrapper

logger = logging.getLogger(__name__)


def cache_reconstruction(cfg: DictConfig, worker: WorkerPool) -> None:
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
    datamodule.setup("test")
    dataloader = datamodule.test_dataloader()

    logger.info("Building Datamodule Module...DONE!")

    reconstruction_cache_path = Path(cfg.cache.reconstruction_cache_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    storing_mechanism = FeatureCachePickle()
    torch_module_wrapper = torch_module_wrapper.to(device)

    # TODO: v0.2 move to different folder
    graph_config = GraphConfig(frame=[64, 64], pose_interval=1.5)

    # Perform inference
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="Cache Reconstructions (batch-wise)"):
            # Assuming batch is a tuple of (inputs, labels, indices) where indices track sample order
            features, _, scenarios = datamodule.transfer_batch_to_device(batch, device, 0)
            predictions = torch_module_wrapper.forward(features)
            graphs = torch_module_wrapper.predictions_to_graphs(predictions, graph_config)

            assert len(graphs) == len(scenarios)
            for graph, scenario in zip(graphs, scenarios):
                file_name = (
                    reconstruction_cache_path
                    / scenario.log_name
                    / scenario.scenario_type
                    / scenario.token
                    / cfg.cache.reconstruction_name
                )
                file_name.parent.mkdir(parents=True, exist_ok=True)
                storing_mechanism.store_computed_feature_to_folder(file_name, graph)

    return None
