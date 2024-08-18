import logging
from typing import List

from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.planning.script.builders.utils.utils_type import validate_type

from sledge.autoencoder.modeling.matching.abstract_matching import AbstractMatching

logger = logging.getLogger(__name__)


def build_matching(cfg: DictConfig) -> List[AbstractMatching]:
    """
    Build list of matchings based on config.
    :param cfg: Dict config.
    :return List of augmentor objects.
    """
    logger.info("Building matchings...")

    instantiated_matchings = []
    for matching_type in cfg.values():
        matching: AbstractMatching = instantiate(matching_type)
        validate_type(matching, AbstractMatching)
        instantiated_matchings.append(matching)

    logger.info("Building matchings...DONE!")
    return instantiated_matchings
