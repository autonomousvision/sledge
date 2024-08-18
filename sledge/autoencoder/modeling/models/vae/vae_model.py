import torch
import torch.nn as nn

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType

from sledge.autoencoder.modeling.autoencoder_torch_module_wrapper import AutoencoderTorchModuleWrapper
from sledge.autoencoder.modeling.models.rvae.rvae_encoder import RVAEEncoder
from sledge.autoencoder.modeling.models.vae.vae_config import VAEConfig

from sledge.autoencoder.preprocessing.features.latent_feature import Latent
from sledge.autoencoder.preprocessing.features.sledge_raster_feature import SledgeRaster
from sledge.autoencoder.preprocessing.feature_builders.sledge_raw_feature_builder import SledgeRawFeatureBuilder
from sledge.autoencoder.preprocessing.target_builders.map_id_target_builder import MapIDTargetBuilder


class VAEModel(AutoencoderTorchModuleWrapper):
    """Raster Variation Autoencoder."""

    def __init__(self, config: VAEConfig):
        """
        Initialize Raster VAE.
        :param config: configuration dataclass of VAE.
        """
        feature_builders = [SledgeRawFeatureBuilder(config)]
        target_builders = [MapIDTargetBuilder()]

        super().__init__(feature_builders=feature_builders, target_builders=target_builders)

        self._config = config

        self._raster_encoder = RVAEEncoder(config)
        self._raster_decoder = RasterDecoder(config.latent_channel, config.num_input_channels)

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
        predictions["sledge_raster"] = self._raster_decoder(latent)
        return predictions

    def get_encoder(self) -> nn.Module:
        """Inherited, see superclass."""
        return self._raster_encoder

    def get_decoder(self) -> nn.Module:
        """Inherited, see superclass."""
        return self._raster_decoder


class RasterDecoder(nn.Module):
    """Simple Raster Decoder from latent."""

    def __init__(self, latent_channel: int, num_output_channels: int = 2):
        super(RasterDecoder, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(latent_channel, 256, kernel_size=4, stride=2, padding=1)  # Output: 128x16x16
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # Output: 64x32x32
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Output: 32x64x64
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # Output: 16x128x128
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

        self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # Output: 8x256x256
        self.bn5 = nn.BatchNorm2d(16)
        self.relu5 = nn.ReLU()

        # Final layer to adjust channels from 8 to 12
        self.final_conv = nn.Conv2d(16, num_output_channels, kernel_size=3, stride=1, padding=1)  # Output: 12x256x256

    def forward(self, x):
        x = self.relu1(self.bn1(self.deconv1(x)))
        x = self.relu2(self.bn2(self.deconv2(x)))
        x = self.relu3(self.bn3(self.deconv3(x)))
        x = self.relu4(self.bn4(self.deconv4(x)))
        x = self.relu5(self.bn5(self.deconv5(x)))
        x = self.final_conv(x)
        return SledgeRaster(data=x)
