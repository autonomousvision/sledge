JOB_NAME=metric_caching
METRIC_CACHE_PATH=$SLEDGE_CACHE_ROOT/metric_cache
SEED=0

for MAP in "filter_pgh" "filter_lav" "filter_sgp" "filter_bos"
do
    echo Test metric caching for $MAP...
    python $SLEDGE_DEVKIT_ROOT/sledge/script/run_metrics.py \
    py_func=metric_caching \
    seed=$SEED \
    job_name=$JOB_NAME \
    scenario_builder=nuplan \
    scenario_builder.data_root="\${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/test/" \
    scenario_filter=$MAP \
    scenario_filter.log_names="\${splitter.log_splits.test}" \
    cache.metric_cache_path=$METRIC_CACHE_PATH
    echo Test metric caching for $MAP...DONE!
done
