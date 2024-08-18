from dataclasses import dataclass
from pathlib import Path
from typing import List
from datasets.builder import BuilderConfig
from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig


@dataclass
class RVAELatentBuilderConfig(BuilderConfig):

    feature_name: str = "rvae_latent"
    label_name: str = "map_id"
    rvae_config: RVAEConfig = RVAEConfig()

    def find_file_paths(self, root_path: Path) -> List[Path]:
        """
        Search for latent features in cache.
        :param root_path: root path of cache
        :return: list of latent file paths
        """

        # TODO: move somewhere else
        file_paths: List[Path] = []
        for log_path in root_path.iterdir():
            if log_path.name == "metadata":
                continue
            for scenario_type_path in log_path.iterdir():
                for token_path in scenario_type_path.iterdir():
                    if (token_path / f"{self.feature_name}.gz").is_file() and (
                        token_path / f"{self.label_name}.gz"
                    ).is_file():
                        file_paths.append(token_path)
        return file_paths
