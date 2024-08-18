JOB_NAME=training_vae_model
AUTOENCODER_CACHE_PATH=/path/to/exp/caches/autoencoder_cache
USE_CACHE_WITHOUT_DATASET=True
SEED=0


python $SLEDGE_DEVKIT_ROOT/sledge/script/run_autoencoder.py \
py_func=training \
seed=$SEED \
job_name=$JOB_NAME \
+autoencoder=training_vae_model \
cache.autoencoder_cache_path=$AUTOENCODER_CACHE_PATH \
cache.use_cache_without_dataset=$USE_CACHE_WITHOUT_DATASET \
callbacks="[learning_rate_monitor_callback, model_checkpoint_callback, time_logging_callback, vae_visualization_callback]" 