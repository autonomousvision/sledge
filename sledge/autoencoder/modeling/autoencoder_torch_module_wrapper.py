import abc
from typing import List

import torch.nn as nn
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder

from sledge.autoencoder.preprocessing.features.graph_feature import Graph, GraphConfig


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
        :param features: feature dict as model input
        :param encoder_only: whether to only encode input in autoencoder
        :return: The results of the inference as a TargetsType.
        """

    @abc.abstractmethod
    def predictions_to_graphs(self, predictions: TargetsType, graph_config: GraphConfig) -> List[Graph]:
        """
        Convert autoencoder predictions to line graphs.
        :param predictions: output feature dict of model
        :param graph_config: config dataclass for metric lane graph
        :return: list of graph dataclasses
        """

    @abc.abstractmethod
    def encode(self, features: FeaturesType) -> FeaturesType:
        """
        TODO:
        """

    @abc.abstractmethod
    def decode(self, features: FeaturesType) -> TargetsType:
        """
        TODO:
        """

    @abc.abstractmethod
    def get_encoder(self, features: FeaturesType) -> nn.Module:
        """
        TODO:
        """

    @abc.abstractmethod
    def get_decoder(self, features: FeaturesType) -> nn.Module:
        """
        TODO:
        """
