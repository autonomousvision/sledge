import logging
import os
from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from nuplan.planning.training.experiments.training import TrainingEngine
from nuplan.planning.script.builders.folder_builder import build_training_experiment_folder
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.utils import set_default_path

from sledge.diffusion.experiments.training import run_training_diffusion
from sledge.diffusion.experiments.scenario_caching import run_scenario_caching

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = os.getenv("NUPLAN_HYDRA_CONFIG_PATH", "config/diffusion")

if os.environ.get("NUPLAN_HYDRA_CONFIG_PATH") is not None:
    CONFIG_PATH = os.path.join("../../../../", CONFIG_PATH)

if os.path.basename(CONFIG_PATH) != "diffusion":
    CONFIG_PATH = os.path.join(CONFIG_PATH, "diffusion")
CONFIG_NAME = "default_diffusion"


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
    # worker = build_worker(cfg)
    print(cfg.py_func)

    if cfg.py_func == "training":
        run_training_diffusion(cfg)
    elif cfg.py_func == "scenario_caching":
        run_scenario_caching(cfg)
    else:
        raise NameError(f"Function {cfg.py_func} does not exist")


if __name__ == "__main__":
    main()
