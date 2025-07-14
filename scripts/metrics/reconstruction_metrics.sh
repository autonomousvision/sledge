JOB_NAME=reconstruction_metrics
METRIC_CACHE_PATH=$SLEDGE_CACHE_ROOT/metric_cache
# METRIC_CACHE_PATH=/home/daniel/sledge_workspace/exp/caches/autoencoder_cache
# RECONSTRUCTION_CACHE_PATH=$SLEDGE_CACHE_ROOT/reconstruction_cache
RECONSTRUCTION_CACHE_PATH=$SLEDGE_CACHE_ROOT/reconstruction_cache
SEED=0


python $SLEDGE_DEVKIT_ROOT/sledge/script/run_metrics.py \
py_func=reconstruction \
seed=$SEED \
job_name=$JOB_NAME \
cache.metric_cache_path=$METRIC_CACHE_PATH  \
cache.reconstruction_cache_path=$RECONSTRUCTION_CACHE_PATH
