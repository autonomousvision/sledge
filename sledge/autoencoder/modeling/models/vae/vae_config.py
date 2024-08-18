from typing import Tuple
from dataclasses import dataclass

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeConfig


@dataclass
class VAEConfig(SledgeConfig):
    """Configuration dataclass for Raster VAE."""

    # 1. features raw
    radius: int = 100
    pose_interval: int = 1.0

    # 2. features raster & vector
    frame: Tuple[int, int] = (64, 64)

    num_lines: int = 50
    num_vehicles: int = 50
    num_pedestrians: int = 20
    num_static_objects: int = 30
    num_green_lights: int = 20
    num_red_lights: int = 20

    num_line_poses: int = 20
    vehicle_max_velocity: float = 15
    pedestrian_max_velocity: float = 2

    pixel_size: float = 0.25
    line_dots_radius: int = 0

    # 3. raster encoder π
    model_name: str = "resnet50"
    down_factor: int = 32  # NOTE: specific to resnet
    num_input_channels: int = 12
    latent_channel: int = 64

    # loss
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.1

    # output
    threshold: float = 0.3

    def __post_init__(self):
        super().__post_init__()

    @property
    def pixel_frame(self) -> Tuple[int, int]:
        frame_width, frame_height = self.frame
        return int(frame_width / self.pixel_size), int(frame_height / self.pixel_size)

    @property
    def latent_frame(self) -> Tuple[int, int]:
        pixel_width, pixel_height = self.pixel_frame
        return int(pixel_width / self.down_factor), int(pixel_height / self.down_factor)
