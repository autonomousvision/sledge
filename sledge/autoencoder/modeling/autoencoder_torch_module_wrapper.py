import abc
from typing import List

import torch.nn as nn

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder


class AutoencoderTorchModuleWrapper(TorchModuleWrapper):
    """Torch module wrapper that encapsulates builders for constructing model features and targets."""

    def __init__(
        self,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
    ):
        """
        Construct a model with feature and target builders.
        :param feature_builders: The list of builders which will compute features for this model.
        :param target_builders: The list of builders which will compute targets for this model.
        """
        super().__init__(
            future_trajectory_sampling=None,  # dummy value
            feature_builders=feature_builders,
            target_builders=target_builders,
        )

        self.feature_builders = feature_builders
        self.target_builders = target_builders

    @abc.abstractmethod
    def forward(self, features: FeaturesType, encoder_only: bool) -> TargetsType:
        """
        The main inference call for the model.
        :param features: _description_
        :param encoder_only: whether to only encode input in autoencoder
        :return: The results of the inference as a TargetsType.
        """
        pass

    @abc.abstractmethod
    def encode(self, features: FeaturesType) -> FeaturesType:
        """
        TODO:
        """
        pass

    @abc.abstractmethod
    def decode(self, features: FeaturesType) -> TargetsType:
        """
        TODO:
        """
        pass

    @abc.abstractmethod
    def get_encoder(self, features: FeaturesType) -> nn.Module:
        """
        TODO:
        """
        pass

    @abc.abstractmethod
    def get_decoder(self, features: FeaturesType) -> nn.Module:
        """
        TODO:
        """
        pass
