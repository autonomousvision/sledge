JOB_NAME=latent_caching
AUTOENCODER_CACHE_PATH=/path/to/exp/caches/autoencoder_cache
AUTOENCODER_CHECKPOINT=/path/to/rvae_checkpoint.ckpt
USE_CACHE_WITHOUT_DATASET=True
SEED=0


python $SLEDGE_DEVKIT_ROOT/sledge/script/run_autoencoder.py \
py_func=latent_caching \
seed=$SEED \
job_name=$JOB_NAME \
+autoencoder=training_rvae_model \
data_augmentation=rvae_no_augmentation \
autoencoder_checkpoint=$AUTOENCODER_CHECKPOINT \
cache.autoencoder_cache_path=$AUTOENCODER_CACHE_PATH \
cache.latent_name="rvae_latent" \
cache.use_cache_without_dataset=$USE_CACHE_WITHOUT_DATASET \
callbacks="[]" 