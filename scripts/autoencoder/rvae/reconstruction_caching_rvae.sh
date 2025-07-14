JOB_NAME=reconstruction_caching
# AUTOENCODER_CACHE_PATH=$SLEDGE_CACHE_ROOT/autoencoder_cache

AUTOENCODER_CHECKPOINT="/home/daniel/epoch\=149-step\=149.ckpt"
AUTOENCODER_CACHE_PATH=/home/daniel/sledge_workspace/exp/caches/autoencoder_cache
RECONSTRUCTION_CACHE_PATH=$SLEDGE_CACHE_ROOT/reconstruction_cache

SEED=0


python $SLEDGE_DEVKIT_ROOT/sledge/script/run_autoencoder.py \
py_func=reconstruction_caching \
seed=$SEED \
job_name=$JOB_NAME \
+autoencoder=training_rvae_model \
data_augmentation=rvae_no_augmentation \
data_loader.params.drop_last=False \
autoencoder_checkpoint=$AUTOENCODER_CHECKPOINT \
cache.autoencoder_cache_path=$AUTOENCODER_CACHE_PATH \
cache.reconstruction_cache_path=$RECONSTRUCTION_CACHE_PATH \
cache.use_cache_without_dataset=True \
callbacks="[]"
