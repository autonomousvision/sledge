import logging
from typing import Any, Dict, List, Optional, Tuple
from omegaconf import DictConfig

import torch

from nuplan.planning.training.modeling.types import FeaturesType, move_features_type_to_device
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.training.data_loader.datamodule import create_dataset, DataModule
from nuplan.planning.training.data_loader.splitter import AbstractSplitter
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor

logger = logging.getLogger(__name__)


class AutoencoderDataModule(DataModule):
    """
    Autoencoder Datamodule in SLEDGE wrapping all and datasets for distributed pre-processing.
    NOTE: This wrapper ensures that
        - augmentation (with augmentation/pre-processing) is applied on all subsets.
        - Wrapper is compatible with updated lightning version
    """

    def __init__(
        self,
        feature_preprocessor: FeaturePreprocessor,
        splitter: AbstractSplitter,
        all_scenarios: List[AbstractScenario],
        train_fraction: float,
        val_fraction: float,
        test_fraction: float,
        dataloader_params: Dict[str, Any],
        scenario_type_sampling_weights: DictConfig,
        worker: WorkerPool,
        augmentors: Optional[List[AbstractAugmentor]] = None,
    ) -> None:
        """
        Initialize the class.
        :param feature_preprocessor: Feature preprocessor object.
        :param splitter: Splitter object used to retrieve lists of samples to construct train/val/test sets.
        :param train_fraction: Fraction of training examples to load.
        :param val_fraction: Fraction of validation examples to load.
        :param test_fraction: Fraction of test examples to load.
        :param dataloader_params: Parameter dictionary passed to the dataloaders.
        :param augmentors: Augmentor object for providing data augmentation to data samples.
        """
        super().__init__(
            feature_preprocessor,
            splitter=splitter,
            all_scenarios=all_scenarios,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            dataloader_params=dataloader_params,
            scenario_type_sampling_weights=scenario_type_sampling_weights,
            worker=worker,
            augmentors=augmentors,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up the dataset for each target set depending on the training stage.
        This is called by every process in distributed training.
        :param stage: Stage of training, can be "fit" or "test".
        """
        if stage is None:
            return

        # TODO: Refactor stage str in SLEDGE
        if stage == "fit":
            # Training Dataset
            train_samples = self._splitter.get_train_samples(self._all_samples, self._worker)
            assert len(train_samples) > 0, "Splitter returned no training samples"

            self._train_set = create_dataset(
                train_samples,
                self._feature_preprocessor,
                self._train_fraction,
                "train",
                self._augmentors,
            )

            # Validation Dataset
            val_samples = self._splitter.get_val_samples(self._all_samples, self._worker)
            assert len(val_samples) > 0, "Splitter returned no validation samples"

            self._val_set = create_dataset(
                val_samples,
                self._feature_preprocessor,
                self._val_fraction,
                "validation",
                self._augmentors,
            )
        elif stage == "test":
            # Testing Dataset
            test_samples = self._splitter.get_test_samples(self._all_samples, self._worker)
            assert len(test_samples) > 0, "Splitter returned no test samples"

            self._test_set = create_dataset(
                test_samples,
                self._feature_preprocessor,
                self._test_fraction,
                "test",
                self._augmentors,
            )
        else:
            raise ValueError(f'Stage must be one of ["fit", "test"], got ${stage}.')

    def transfer_batch_to_device(
        self,
        batch: Tuple[FeaturesType, ...],
        device: torch.device,
        dataloader_idx: int,
    ) -> Tuple[FeaturesType, ...]:
        """
        Transfer a batch to device.
        :param batch: Batch on origin device.
        :param device: Desired device.
        :param dataloader_idx: The index of the dataloader to which the batch belongs. (ignored)
        :return: Batch in new device.
        """
        return tuple(
            (
                move_features_type_to_device(batch[0], device),
                move_features_type_to_device(batch[1], device),
                batch[2],
            )
        )
