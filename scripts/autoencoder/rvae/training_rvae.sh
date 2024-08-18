JOB_NAME=training_rvae_model
AUTOENCODER_CACHE_PATH=/path/to/exp/caches/autoencoder_cache
AUTOENCODER_CHECKPOINT=null # set for weight intialization / continue training
USE_CACHE_WITHOUT_DATASET=True
SEED=0

python $SLEDGE_DEVKIT_ROOT/sledge/script/run_autoencoder.py \
py_func=training \
seed=$SEED \
job_name=$JOB_NAME \
+autoencoder=training_rvae_model \
autoencoder_checkpoint=$AUTOENCODER_CHECKPOINT \
cache.autoencoder_cache_path=$AUTOENCODER_CACHE_PATH \
cache.use_cache_without_dataset=$USE_CACHE_WITHOUT_DATASET \
callbacks="[learning_rate_monitor_callback, model_checkpoint_callback, time_logging_callback, rvae_visualization_callback]" 