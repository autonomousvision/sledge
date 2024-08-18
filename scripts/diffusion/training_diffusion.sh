JOB_NAME=training_dit_diffusion
AUTOENCODER_CACHE_PATH=/path/to/exp/caches/autoencoder_cache
AUTOENCODER_CHECKPOINT=/path/to/rvae_checkpoint.ckpt
DIFFUSION_CHECKPOINT=null # set for weight intialization / continue training
DIFFUSION_MODEL=dit_b_model # [dit_s_model, dit_b_model, dit_l_model, dit_xl_model]
CLEANUP_DIFFUSION_CACHE=False
SEED=0

accelerate launch $SLEDGE_DEVKIT_ROOT/sledge/script/run_diffusion.py \
py_func=training \
seed=$SEED \
job_name=$JOB_NAME \
+diffusion=training_dit_model \
diffusion_model=$DIFFUSION_MODEL \
cache.autoencoder_cache_path=$AUTOENCODER_CACHE_PATH \
cache.cleanup_diffusion_cache=$CLEANUP_DIFFUSION_CACHE \
autoencoder_checkpoint=$AUTOENCODER_CHECKPOINT \
diffusion_checkpoint=$DIFFUSION_CHECKPOINT  