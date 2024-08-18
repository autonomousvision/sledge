from typing import List, Optional, Tuple

import numpy as np

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor
from nuplan.planning.training.data_augmentation.data_augmentation_util import ParameterToScale, ScalingDirection
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.common.actor_state.state_representation import StateSE2

from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVectorRaw
from sledge.autoencoder.data_augmentation.augmentation_utils import random_se2, transform_sledge_vector, element_dropout
from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import (
    sledge_raw_feature_processing,
)


class RVAEAugmenter(AbstractAugmentor):
    def __init__(
        self,
        config: RVAEConfig,
        se2_noise: Optional[Tuple[float, float, float]] = None,
        p_vehicle_dropout: Optional[float] = None,
        p_pedestrian_dropout: Optional[float] = None,
        p_static_dropout: Optional[float] = None,
    ) -> None:
        """
         Initialize the augmenter for RVAE.
        NOTE:
        - Object pre-processes and rasterizes the features
        - Enables augmentation and config changes with the same autoencoder cache.
        :param config: config dataclass of RVAE
        :param se2_noise: tuple of (x,y,θ) noise scale, defaults to None
        :param p_vehicle_dropout: probability of removing vehicle, defaults to None
        :param p_pedestrian_dropout: probability of removing pedestrian, defaults to None
        :param p_static_dropout: probability of removing static objects, defaults to None
        """
        self._config = config

        self._se2_augmentation = StateSE2(se2_noise[0], se2_noise[1], np.deg2rad(se2_noise[2])) if se2_noise else None
        self._p_vehicle_dropout = p_vehicle_dropout
        self._p_pedestrian_dropout = p_pedestrian_dropout
        self._p_static_dropout = p_static_dropout

    def augment(
        self,
        features: FeaturesType,
        targets: TargetsType,
        scenario: Optional[AbstractScenario] = None,
    ) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""

        sledge_vector_raw: SledgeVectorRaw = features["sledge_raw"]

        # TODO: Refactor / Add new augmentations
        if self._se2_augmentation:
            origin = random_se2(self._se2_augmentation)
            sledge_vector_raw = transform_sledge_vector(sledge_vector_raw, origin)

        if self._p_vehicle_dropout:
            sledge_vector_raw.vehicles = element_dropout(sledge_vector_raw.vehicles, self._p_vehicle_dropout)
        if self._p_pedestrian_dropout:
            sledge_vector_raw.pedestrians = element_dropout(sledge_vector_raw.pedestrians, self._p_pedestrian_dropout)
        if self._p_static_dropout:
            sledge_vector_raw.static_objects = element_dropout(sledge_vector_raw.static_objects, self._p_static_dropout)

        frame_vector, frame_raster = sledge_raw_feature_processing(sledge_vector_raw, self._config)

        del features["sledge_raw"]
        features["sledge_raster"] = frame_raster
        targets["sledge_vector"] = frame_vector

        return features, targets

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return []

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return []

    @property
    def augmentation_probability(self) -> ParameterToScale:
        """Inherited, see superclass."""
        return ParameterToScale(
            param=1.0,
            param_name=f"{self._augment_prob=}".partition("=")[0].split(".")[1],
            scaling_direction=ScalingDirection.MAX,
        )
