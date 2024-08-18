JOB_NAME=scenario_caching
AUTOENCODER_CHECKPOINT=/path/to/rvae_checkpoint.ckpt
DIFFUSION_CHECKPOINT=/path/to/diffusion/checkpoint
DIFFUSION_MODEL=dit_b_model # [dit_s_model, dit_b_model, dit_l_model, dit_xl_model]
SEED=0


python $SLEDGE_DEVKIT_ROOT/sledge/script/run_diffusion.py \
py_func=scenario_caching \
seed=$SEED \
job_name=$JOB_NAME \
py_func=scenario_cache \
+diffusion=training_dit_model \
diffusion_model=$DIFFUSION_MODEL \
autoencoder_checkpoint=$AUTOENCODER_CHECKPOINT \
diffusion_checkpoint=$DIFFUSION_CHECKPOINT 