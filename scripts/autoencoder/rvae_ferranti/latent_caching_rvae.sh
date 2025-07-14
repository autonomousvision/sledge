JOB_NAME=latent_caching
CACHE_PATH=/home/daniel/sledge_workspace/exp/caches/autoencoder_cache
AUTOENCODER_CHECKPOINT="/home/daniel/sledge_workspace/exp/exp/training_rvae_model/training_rvae_model/2024.08.09.22.58.31/best_model/epoch\=18-step\=18.ckpt"
USE_CACHE_WITHOUT_DATASET=True
SEED=1


python $SLEDGE_DEVKIT_ROOT/sledge/script/run_autoencoder.py \
py_func=latent_cache \
seed=$SEED \
job_name=$JOB_NAME \
+training_autoencoder=training_rvae_model \
data_augmentation=rvae_no_augmentation \
autoencoder_checkpoint=$AUTOENCODER_CHECKPOINT \
cache.autoencoder_cache_path=$AUTOENCODER_CACHE_PATH \
cache.latent_name="rvae_latent" \
cache.use_cache_without_dataset=$USE_CACHE_WITHOUT_DATASET \
callbacks="[]"
