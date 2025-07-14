JOB_NAME=feature_caching
AUTOENCODER_CACHE_PATH=$SLEDGE_CACHE_ROOT/autoencoder_cache
SEED=0

for MAP in "filter_pgh" "filter_lav" "filter_sgp" "filter_bos"
do
    echo Train/Val caching for $MAP
    python $SLEDGE_DEVKIT_ROOT/sledge/script/run_autoencoder.py \
    py_func=feature_caching \
    seed=$SEED \
    job_name=$JOB_NAME \
    +autoencoder=training_rvae_model \
    scenario_builder=nuplan \
    scenario_filter=$MAP \
    cache.autoencoder_cache_path=$AUTOENCODER_CACHE_PATH \
    callbacks="[]"
    echo Train/Val caching for $MAP...DONE!
done

for MAP in "filter_pgh" "filter_lav" "filter_sgp" "filter_bos"
do
    echo Test caching for $MAP...
    python $SLEDGE_DEVKIT_ROOT/sledge/script/run_autoencoder.py \
    py_func=feature_caching \
    seed=$SEED \
    job_name=$JOB_NAME \
    +autoencoder=training_rvae_model \
    scenario_builder=nuplan \
    scenario_builder.data_root="\${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/test/" \
    scenario_filter=$MAP \
    scenario_filter.log_names="\${splitter.log_splits.test}" \
    cache.autoencoder_cache_path=$AUTOENCODER_CACHE_PATH \
    callbacks="[]"
    echo Test caching for $MAP...DONE!
done
