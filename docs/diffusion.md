# Diffusion

This section provides instructions on how to utilize diffusion models within the SLEDGE framework. 

### 1. Training Diffusion
Before training a diffusion model, make sure you have a trained autoencoder checkpoint and latent cache as described in `docs/autoencoder.md`.
You can start a training experiment by running the script:
```bash
cd $SLEDGE_DEVKIT_ROOT/scripts/diffusion/
bash training_diffusion.sh
``` 
Please make sure you added the autoencoder checkpoint path to the bash script. Before training starts, the latent variables will be stored in a Hugging Face dataset format and saved to `$SLEDGE_EXP_ROOT/caches/diffusion_cache`. This format is compatible with the [`accelerate`](https://github.com/huggingface/accelerate) framework and has performance advantages. Read more [here](https://huggingface.co/docs/datasets/about_arrow) if you are interested. Our training pipeline supports [diffusion transformers (DiT)](https://arxiv.org/abs/2212.09748) in four sizes (S, B, L, XL). You can find the experiment folder and checkpoints in `$SLEDGE_EXP_ROOT/exp`. You can also monitor the training with tensorboard. 

### 2. Scenario Synthesis
Given the trained diffusion model, you can generate a set of samples used for driving simulation or the generative metrics. You can set the diffuser checkpoint path and run the following:
```bash
bash scenario_caching_diffusion.sh
```
The samples are stored in `$SLEDGE_EXP_ROOT/caches/scenario_cache` by default. These samples can be simulated in the v0.1 release.
Additional options for route extrapolation by inpainting will be added in a future update.

### 3. Evaluating Diffusion
Coming soon!