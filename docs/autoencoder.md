# Autoencoder

This guide provides instructions on how to run an autoencoder in SLEDGE. 
The tutorial below shows the key functionalities of the raster-to-vector autoencoder (RVAE).

### 1. Feature Caching
Similar to the [nuplan-devkit](https://github.com/motional/nuplan-devkit),
 pre-processing of the training data is recommended. 
The cache for the RVAE can be created by running:
```bash
cd $SLEDGE_DEVKIT_ROOT/scripts/autoencoder/rvae/
bash feature_caching_rvae.sh
```
This script pre-processes the vector features of several maps sequentially. The cached features only store the local map and agents in a general vector format. The features are further processed and rasterized on the fly during training. This two-step processing enables fast access to training data and allows data augmentation (e.g. random rotation and translation) for RVAE training. The feature cache is compatible with other autoencoders.

### 2. Training Autoencoder
After creating or downloading the autoencoder cache, you can start the training. We provide an example script in the same folder.
```bash
bash training_rvae.sh
```
You can find the experiment folder of training in `$SLEDGE_EXP_ROOT/exp` and monitor the run with tensorboard.

### 3. Latent Caching
You must first cache the latent variables to run a latent diffusion model with the trained autoencoder. In SLEDGE, we cache the latent variables into the autoencoder cache directory (i.e. `$SLEDGE_EXP_ROOT/caches/autoencoder_cache`). The bash script is provided in the RVAE folder.
```bash
bash latent_caching_rvae.sh
```
Importantly, data augmentation is disabled for latent caching. We also only cache the samples from the training split.

### 4. Evaluating Autoencoder
Coming soon!