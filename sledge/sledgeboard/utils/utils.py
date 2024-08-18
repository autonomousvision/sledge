import logging
from pathlib import Path
from typing import List

import pandas as pd

from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame
from sledge.sledgeboard.base.data_class import SledgeBoardFile

logger = logging.getLogger(__name__)


def metric_statistics_reader(parquet_file: Path) -> MetricStatisticsDataFrame:
    """
    Reader for a metric statistic parquet file.
    :param parquet_file: Parquet file path to read.
    :return MetricStatisticsDataFrame.
    """
    data_frame = MetricStatisticsDataFrame.load_parquet(parquet_file)
    return data_frame


def metric_aggregator_reader(parquet_file: Path) -> pd.DataFrame:
    """
    Reader for a metric aggregator parquet file.
    :param parquet_file: Parquet file path to read.
    :return Pandas data frame.
    """
    data_frame = pd.read_parquet(parquet_file)
    return data_frame


def check_sledgeboard_file_paths(main_paths: List[str]) -> List[Path]:
    """
    Check if given file paths are valid SledgeBoard files.
    :param main_paths: A list of file paths.
    :return A list of available SledgeBoard files.
    """
    available_paths = []
    for main_path in main_paths:
        main_folder_path: Path = Path(main_path)
        if main_folder_path.is_dir():
            # Search for sledgeboard event files.
            files = list(main_folder_path.iterdir())
            event_files = [file for file in files if file.name.endswith(SledgeBoardFile.extension())]

            if len(event_files) > 0:
                # Descending order.
                event_files = sorted(event_files, reverse=True)
                # Load the first file only.
                available_paths.append(event_files[0])
        elif main_folder_path.is_file() and main_folder_path.name.endswith(SledgeBoardFile.extension()):
            available_paths.append(main_folder_path)
        else:
            raise RuntimeError(f"{str(main_folder_path)} is not a valid SledgeBoard file")

        if len(available_paths) == 0:
            logger.info("No available SledgeBoard files are found.")

    return available_paths


def read_sledgeboard_file_paths(file_paths: List[Path]) -> List[SledgeBoardFile]:
    """
    Read a list of file paths to SledgeBoardFile data class.
    :param file_paths: A list of file paths.
    :return A list of SledgeBoard files.
    """
    sledgeboard_files = []
    for file_path in file_paths:
        sledgeboard_file = SledgeBoardFile.load_sledgeboard_file(file_path)
        sledgeboard_file.current_path = file_path.parents[0]
        sledgeboard_files.append(sledgeboard_file)

    return sledgeboard_files
