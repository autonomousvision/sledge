# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import warnings

import os
import gzip
import pickle
from tqdm import tqdm
import datasets

from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig
from sledge.diffusion.dataset.rvae_latent_builder_config import RVAELatentBuilderConfig

_CITATION = """
@inproceedings{Chitta2024ECCV, 
	author = {Kashyap Chitta and Daniel Dauner and Andreas Geiger}, 
	title = {SLEDGE: Synthesizing Driving Environments with Generative Models and Rule-Based Traffic}, 
	booktitle = {European Conference on Computer Vision (ECCV)}, 
	year = {2024}, 
}
"""

_DESCRIPTION = """Apache-2.0 license"""
_HOMEPAGE = "https://github.com/autonomousvision/sledge"
_LICENSE = ""


class RVAELatentDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.0.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'default')
    BUILDER_CONFIGS = [
        RVAELatentBuilderConfig(
            name="rvae_latent",
            version=VERSION,
            description="This part of my dataset covers a first domain",
            feature_name="rvae_latent",
            label_name="map_id",
            rvae_config=RVAEConfig(),
        ),
    ]

    DEFAULT_CONFIG_NAME = (
        "rvae_latent"  # It's not mandatory to have a default configuration. Just use one if it make sense.
    )

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset

        warnings.warn("Considering standard parameters from RVAEConfig for latent shape!")

        c = self.config.rvae_config.latent_channel
        w, h = self.config.rvae_config.latent_frame

        features = datasets.Features(
            {
                "features": datasets.Array3D(shape=(c, w, h), dtype="float32"),
                "label": datasets.Value("int64"),
            }
        )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive

        # TODO: sort tokens according to train / val splits

        file_paths = self.config.find_file_paths(Path(dl_manager._data_dir))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": file_paths,
                    "split": "train",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        for key, file_path in tqdm(enumerate(filepath)):

            with gzip.open(os.path.join(file_path, f"{self.config.feature_name}.gz"), "rb") as f:
                data = pickle.load(f)

            with gzip.open(os.path.join(file_path, f"{self.config.label_name}.gz"), "rb") as f:
                id = pickle.load(f)

            array = data["mu"]
            label = id["id"]

            yield key, {
                "features": array,
                "label": label,
            }
