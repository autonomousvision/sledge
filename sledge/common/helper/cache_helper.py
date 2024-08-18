from typing import List
from pathlib import Path


def find_feature_paths(root_path: Path, feature_name: str) -> List[Path]:
    """
    Simple helper function, collecting all available gzip files in a cache.
    :param root_path: path of feature cache
    :param feature_name: name of feature, excluding file ending
    :return: list of paths
    """

    # TODO: move somewhere else
    file_paths: List[Path] = []
    for log_path in root_path.iterdir():
        if log_path.name == "metadata":
            continue
        for scenario_type_path in log_path.iterdir():
            for token_path in scenario_type_path.iterdir():
                feature_path = token_path / f"{feature_name}.gz"
                if feature_path.is_file():
                    file_paths.append(token_path / feature_name)

    return file_paths
