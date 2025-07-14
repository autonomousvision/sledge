JOB_NAME=feature_caching
USE_CACHE_WITHOUT_DATASET=True
SEED=1

AUTOENCODER_CACHE_PATH=/home/daniel/sledge_workspace/exp/caches/autoencoder_cache

for MAP in "filter_pgh" "filter_lav" "filter_sgp" "filter_bos"
do
    python $SLEDGE_DEVKIT_ROOT/sledge/script/run_autoencoder.py \
    py_func=feature_cache \
    seed=$SEED \
    job_name=$JOB_NAME \
    +training_autoencoder=training_rvae_model \
    scenario_builder=nuplan \
    scenario_filter=$MAP \
    cache.autoencoder_cache_path=$AUTOENCODER_CACHE_PATH \
    cache.use_cache_without_dataset=$USE_CACHE_WITHOUT_DATASET \
    callbacks="[]"
done
