from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig
from accelerate.logging import get_logger

from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVector
from sledge.autoencoder.preprocessing.features.map_id_feature import MAP_ID_TO_NAME
from sledge.script.builders.diffusion_builder import build_pipeline_from_checkpoint

logger = get_logger(__name__, log_level="INFO")


def run_scenario_caching(cfg: DictConfig) -> None:
    """
    Applies the diffusion model generate and cache scenarios.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    """

    logger.info("Building pipeline from checkpoint...")
    pipeline = build_pipeline_from_checkpoint(cfg)
    pipeline.to("cuda")
    logger.info("Building pipeline from checkpoint...DONE!")

    logger.info("Scenario caching...")
    storing_mechanism = FeatureCachePickle()
    current_cache_size: int = 0
    class_labels = list(range(cfg.num_classes)) * (cfg.inference_batch_size // cfg.num_classes)
    num_total_batches = (cfg.cache.scenario_cache_size // cfg.inference_batch_size) + 1
    for _ in tqdm(range(num_total_batches), desc="Load cache files..."):
        sledge_vector_list = pipeline(
            class_labels=class_labels,
            num_inference_timesteps=cfg.num_inference_timesteps,
            guidance_scale=cfg.guidance_scale,
            num_classes=cfg.num_classes,
        )
        for sledge_vector, map_id in zip(sledge_vector_list, class_labels):
            sledge_vector_numpy: SledgeVector = sledge_vector.torch_to_numpy()
            file_name = (
                Path(cfg.cache.scenario_cache_path)
                / "log"
                / MAP_ID_TO_NAME[map_id]
                / str(current_cache_size)
                / "sledge_vector"
            )
            file_name.parent.mkdir(parents=True, exist_ok=True)
            storing_mechanism.store_computed_feature_to_folder(file_name, sledge_vector_numpy)
            current_cache_size += 1
            if current_cache_size >= cfg.cache.scenario_cache_size:
                break
    logger.info("Scenario caching...DONE!")
    return None
