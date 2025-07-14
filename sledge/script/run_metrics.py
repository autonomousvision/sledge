import logging
import os
from typing import Optional

import hydra
import pytorch_lightning as pl
from nuplan.planning.script.builders.folder_builder import build_training_experiment_folder
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.utils import set_default_path
from nuplan.planning.training.experiments.training import TrainingEngine
from omegaconf import DictConfig

from sledge.metrics.experiments.metric_caching import metric_caching
from sledge.metrics.experiments.reconstruction_metrics import reconstruction_metrics

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = os.getenv("NUPLAN_HYDRA_CONFIG_PATH", "config/metrics")

if os.environ.get("NUPLAN_HYDRA_CONFIG_PATH") is not None:
    CONFIG_PATH = os.path.join("../../../../", CONFIG_PATH)

if os.path.basename(CONFIG_PATH) != "metrics":
    CONFIG_PATH = os.path.join(CONFIG_PATH, "metrics")
CONFIG_NAME = "default_metrics"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> Optional[TrainingEngine]:
    """
    Main entrypoint for diffusion experiments.
    :param cfg: omegaconf dictionary
    """
    # Fix random seed
    pl.seed_everything(cfg.seed, workers=True)

    # Configure logger
    build_logger(cfg)

    # Create output storage folder
    build_training_experiment_folder(cfg=cfg)

    # Build worker
    worker = build_worker(cfg)

    if cfg.py_func == "metric_caching":
        metric_caching(cfg, worker)
    elif cfg.py_func == "reconstruction":
        reconstruction_metrics(cfg, worker)
        # pass
    elif cfg.py_func == "generation":
        # run_scenario_caching(cfg)
        pass
    else:
        raise NameError(f"Function {cfg.py_func} does not exist")


if __name__ == "__main__":
    main()
