import gc
import itertools
import logging
import os
import uuid
from typing import Dict, List, Optional, Union

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.training.experiments.cache_metadata_entry import CacheMetadataEntry, CacheResult
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig

from sledge.autoencoder.experiments.feature_caching import build_scenarios_from_config
from sledge.autoencoder.preprocessing.feature_builders.graph_target_builder import GraphFeatureBuilder
from sledge.autoencoder.preprocessing.features.graph_feature import GraphConfig

logger = logging.getLogger(__name__)


def metric_caching(cfg: DictConfig, worker: WorkerPool) -> None:
    """
    Build the lightning datamodule and cache all samples.
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    """
    assert (
        cfg.cache.metric_cache_path is not None
    ), f"Cache path cannot be None when caching, got {cfg.cache.metric_cache_path}"

    scenario_builder = build_scenario_builder(cfg)
    scenarios = build_scenarios_from_config(cfg, scenario_builder, worker)

    data_points = [{"scenario": scenario, "cfg": cfg} for scenario in scenarios]
    logger.info("Starting dataset caching of %s files...", str(len(data_points)))

    _ = worker_map(worker, cache_scenarios, data_points)


def cache_scenarios(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[CacheResult]:
    """
    Performs the caching of scenario DB files in parallel.
    :param args: A list of dicts containing the following items:
        "scenario": the scenario as built by scenario_builder
        "cfg": the DictConfig to use to process the file.
    :return: A dict with the statistics of the job. Contains the following keys:
        "successes": The number of successfully processed scenarios.
        "failures": The number of scenarios that couldn't be processed.
    """

    # Define a wrapper method to help with memory garbage collection.
    # This way, everything will go out of scope, allowing the python GC to clean up after the function.
    #
    # This is necessary to save memory when running on large datasets.
    def cache_scenarios_internal(args: List[Dict[str, Union[List[AbstractScenario], DictConfig]]]) -> List[CacheResult]:
        node_id = int(os.environ.get("NODE_RANK", 0))
        thread_id = str(uuid.uuid4())

        scenarios: List[AbstractScenario] = [a["scenario"] for a in args]
        cfg: DictConfig = args[0]["cfg"]

        # TODO: v0.2 add load parameters from config
        feature_builders = [GraphFeatureBuilder(GraphConfig(frame=[64, 64], pose_interval=1.5))]

        # Create feature preprocessor
        assert (
            cfg.cache.metric_cache_path is not None
        ), f"Cache path cannot be None when caching, got {cfg.cache.metric_cache_path}"
        preprocessor = FeaturePreprocessor(
            cache_path=cfg.cache.metric_cache_path,
            force_feature_computation=cfg.cache.force_feature_computation,
            feature_builders=feature_builders,
            target_builders=[],
        )

        logger.info("Extracted %s scenarios for thread_id=%s, node_id=%s.", str(len(scenarios)), thread_id, node_id)
        num_failures = 0
        num_successes = 0
        all_file_cache_metadata: List[Optional[CacheMetadataEntry]] = []
        for idx, scenario in enumerate(scenarios):
            logger.info(
                "Processing scenario %s / %s in thread_id=%s, node_id=%s",
                idx + 1,
                len(scenarios),
                thread_id,
                node_id,
            )

            features, targets, file_cache_metadata = preprocessor.compute_features(scenario)

            scenario_num_failures = sum(
                0 if feature.is_valid else 1 for feature in itertools.chain(features.values(), targets.values())
            )
            scenario_num_successes = len(features.values()) + len(targets.values()) - scenario_num_failures
            num_failures += scenario_num_failures
            num_successes += scenario_num_successes
            all_file_cache_metadata += file_cache_metadata

        logger.info("Finished processing scenarios for thread_id=%s, node_id=%s", thread_id, node_id)
        return [CacheResult(failures=num_failures, successes=num_successes, cache_metadata=all_file_cache_metadata)]

    result = cache_scenarios_internal(args)

    # Force a garbage collection to clean up any unused resources
    gc.collect()

    return result
