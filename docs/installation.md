# Installation and download

To get started with SLEDGE, we provide the main instructions on how to install the devkit and download the necessary data for training and simulation.

### 1. Clone the sledge-devkit
Begin by cloning the SLEDGE repository and navigating to the repository directory:
```bash
git clone https://github.com/autonomousvision/sledge.git
cd sledge
```

### 2. Download the data
**Important:** Before downloading any data, please ensure you have read the [nuPlan license](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE).

You can find the bash scripts to download the datasplits in `scripts/download/`. For complete usage in SLEDGE, you need to download the nuPlan dataset (without sensor data) with the `download_nuplan.sh` script (~2TiB). Each part of nuPlan has different roles in the SLEDGE framework:
- `train` and `val` contain the training and validation logs, which primarily serve for autoencoder feature caching. The autoencoder and diffusion training pipeline will operate on this pre-processed cache.
- `test` contains the test logs that are used in the metrics (reconstruction / generative) and for "Lane->Agent" simulation. 
- `mini` has a few log files (~15GB) and can be used for simple experiments and demos.
- `maps` contains the maps of the four cities in nuPlan. These maps are necessary for autoencoder feature caching, the metrics, and "Lane->Agent" simulation.

If you don't want to change the autoencoder features, you can directly download the pre-processed training cache with the `download_cache.sh` script (~15GiB). As such, the [license and conditions](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE) of the nuPlan dataset apply. 

After downloading, organize the decompressed files in the following directory structure:
```angular2html
~/sledge_workspace
├── sledge (devkit)
├── exp
│   ├── caches
│   │   ├── autoencoder_cache
│   │   │   └── <cached logs>
│   │   ├── diffusion_cache
│   │   │   └── <hugging face dataset>
│   │   └── scenario_cache
│   │       └── <generated logs>
│   └── exp
│       └── <sledge_experiments>
└── dataset
    ├── maps
    │   ├── nuplan-maps-v1.0.json
    │   ├── sg-one-north
    │   │   └── ...
    │   ├── us-ma-boston
    │   │   └── ...
    │   ├── us-nv-las-vegas-strip
    │   │   └── ...
    │   └── us-pa-pittsburgh-hazelwood
    │       └── ...
    └── nuplan-v1.1
         ├── splits 
         │     ├── mini 
         │     │    ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
         │     │    ├── ...
         │     │    └── 2021.10.11.08.31.07_veh-50_01750_01948.db
         │     ├── test 
         │     │    ├── 2021.05.25.12.30.39_veh-25_00005_00215.db
         │     │    ├── ...
         │     │    └── 2021.10.06.08.34.20_veh-53_01089_01868.db
         │     └── trainval
         │          ├── 2021.05.12.19.36.12_veh-35_00005_00204.db
         │          ├── ...
         │          └── 2021.10.22.18.45.52_veh-28_01175_01298.db
         └── sensor_blobs (empty)
```
Several environment variables need to be added next to your `~/.bashrc` file. 
For the above, the environment variables are defined as:
```bash
export NUPLAN_DATA_ROOT="$HOME/sledge_workspace/dataset"
export NUPLAN_MAPS_ROOT="$HOME/sledge_workspace/dataset/maps"

export SLEDGE_EXP_ROOT="$HOME/sledge_workspace/exp"
export SLEDGE_DEVKIT_ROOT="$HOME/sledge_workspace/sledge"
```

### 3. Install the sledge-devkit
To install SLEDGE, create a new conda environment and install the necessary dependencies as follows:
```bash
conda env create --name sledge -f environment.yml
conda activate sledge
pip install -e .
```
With these steps completed, SLEDGE should be ready to use.