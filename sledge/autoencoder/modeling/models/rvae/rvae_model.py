import torch
import torch.nn as nn

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType

from sledge.autoencoder.modeling.autoencoder_torch_module_wrapper import AutoencoderTorchModuleWrapper
from sledge.autoencoder.modeling.models.rvae.rvae_encoder import RVAEEncoder
from sledge.autoencoder.modeling.models.rvae.rvae_decoder import RVAEDecoder
from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig
from sledge.autoencoder.preprocessing.features.latent_feature import Latent
from sledge.autoencoder.preprocessing.feature_builders.sledge_raw_feature_builder import SledgeRawFeatureBuilder
from sledge.autoencoder.preprocessing.target_builders.map_id_target_builder import MapIDTargetBuilder


class RVAEModel(AutoencoderTorchModuleWrapper):
    """Raster-Vector Autoencoder in of SLEDGE."""

    def __init__(self, config: RVAEConfig):
        """
        Initialize Raster-Vector Autoencoder.
        :param config: configuration dataclass of RVAE.
        """
        feature_builders = [SledgeRawFeatureBuilder(config)]
        target_builders = [MapIDTargetBuilder()]

        super().__init__(feature_builders=feature_builders, target_builders=target_builders)

        self._config = config

        self._raster_encoder = RVAEEncoder(config)
        self._vector_decoder = RVAEDecoder(config)

    @staticmethod
    def _reparameterize(latent: Latent) -> torch.Tensor:
        """
        Reparameterization method for variational autoencoder's.
        :param latent: dataclass for mu, logvar tensors.
        :return: combined latent tensor.
        """
        mu, log_var = latent.mu, latent.log_var
        assert mu.shape == log_var.shape
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, features: FeaturesType, encode_only: bool = False) -> TargetsType:
        """Inherited, see superclass."""
        predictions: TargetsType = {}

        # encoding
        predictions["latent"] = self._raster_encoder(features["sledge_raster"].data)
        latent = self._reparameterize(predictions["latent"])
        if encode_only:
            return predictions

        # decoding
        predictions["sledge_vector"] = self._vector_decoder(latent)
        return predictions

    def get_encoder(self) -> nn.Module:
        """Inherited, see superclass."""
        return self._raster_encoder

    def get_decoder(self) -> nn.Module:
        """Inherited, see superclass."""
        return self._vector_decoder
