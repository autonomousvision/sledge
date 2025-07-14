import gc
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig

from sledge.autoencoder.preprocessing.features.graph_feature import Graph, GraphConfig
from sledge.common.helper.cache_helper import find_feature_paths
from sledge.metrics.reconstruction.run_reconstruction_metric import graph_reconstruction_metrics

logger = logging.getLogger(__name__)


def reconstruction_metrics(cfg: DictConfig, worker: WorkerPool) -> None:
    """
    Build the lightning datamodule and cache all samples.
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    """
    assert (
        cfg.cache.metric_cache_path is not None
    ), f"Cache path cannot be None when caching, got {cfg.cache.metric_cache_path}"

    assert (
        cfg.cache.reconstruction_cache_path is not None
    ), f"Cache path cannot be None when caching, got {cfg.cache.reconstruction_cache_path}"

    metric_cache_path = Path(cfg.cache.metric_cache_path)
    reconstruction_cache_path = Path(cfg.cache.reconstruction_cache_path)

    # Load ground-truth and predicted feature paths
    target_paths = find_feature_paths(metric_cache_path, "graph")
    prediction_paths = find_feature_paths(reconstruction_cache_path, cfg.cache.reconstruction_name)

    # Convert to scenario token dictionary
    target_path_dict = {path.parent.name: path for path in target_paths}
    prediction_path_dict = {path.parent.name: path for path in prediction_paths}

    # Sort dictionaries
    target_path_dict = dict(sorted(target_path_dict.items()))
    prediction_path_dict = dict(sorted(prediction_path_dict.items()))

    # Run the metrics
    assert target_path_dict.keys() == prediction_path_dict.keys()
    data_points = [
        {"target_path": target_path, "prediction_path": prediction_path, "cfg": cfg}
        for target_path, prediction_path in zip(target_path_dict.values(), prediction_path_dict.values())
    ]
    results = worker_map(worker, run_reconstruction_metric_samples, data_points)

    reconstruction_metric_df = pd.DataFrame(results)
    mean_row = reconstruction_metric_df.drop(columns=["name"]).mean(skipna=True)
    mean_row["name"] = "mean"
    reconstruction_metric_df.loc[len(reconstruction_metric_df)] = mean_row

    output_csv = Path(cfg.output_dir) / "reconstruction_metric.csv"
    reconstruction_metric_df.to_csv(output_csv, index=False)

    return None


def run_reconstruction_metric_samples(args: List[Dict[str, Union[Path, DictConfig]]]) -> List[Dict[str, float]]:

    # Define a wrapper method to help with memory garbage collection.
    # This way, everything will go out of scope, allowing the python GC to clean up after the function.
    #
    # This is necessary to save memory when running on large datasets.
    def run_internal(args: List[Dict[str, Union[Path, DictConfig]]]) -> List[Dict[str, float]]:

        thread_id = str(uuid.uuid4())

        # cfg: DictConfig = args[0]["cfg"]
        target_paths: List[Path] = [a["target_path"] for a in args]
        prediction_paths: List[Path] = [a["prediction_path"] for a in args]

        assert len(target_paths) == len(prediction_paths)
        logger.info(f"Extracted {len(target_paths)} scenarios for thread_id={thread_id}")

        storing_mechanism = FeatureCachePickle()

        graph_config = GraphConfig(frame=[64, 64], pose_interval=1.5)

        results: List[Dict[str, float]] = []
        for idx, (target_path, prediction_path) in enumerate(zip(target_paths, prediction_paths)):
            logger.info(f"Processing scenario {idx+1} / {len(target_paths)} in thread_id={thread_id}")

            token: str = target_path.parent.name
            scenario_type: str = target_path.parent.parent.name
            log_name: str = target_path.parent.parent.parent.name

            assert token == prediction_path.parent.name
            assert scenario_type == prediction_path.parent.parent.name
            assert log_name == prediction_path.parent.parent.parent.name

            name = f"{log_name}/{scenario_type}/{token}"
            result = {"name": name}

            target_graph = storing_mechanism.load_computed_feature_from_folder(target_path, Graph)
            prediction_graph = storing_mechanism.load_computed_feature_from_folder(prediction_path, Graph)

            reconstruction_result = graph_reconstruction_metrics(target_graph, prediction_graph, graph_config)

            result.update(reconstruction_result)
            results.append(result)

        logger.info(f"Finished processing scenarios for thread_id={thread_id}")
        return results

    result = run_internal(args)

    # Force a garbage collection to clean up any unused resources
    gc.collect()

    return result
