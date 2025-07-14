JOB_NAME=training_rvae_model
AUTOENCODER_CACHE_PATH=$SLEDGE_CACHE_ROOT/autoencoder_cache
AUTOENCODER_CHECKPOINT=null # set for weight intialization / continue training
USE_CACHE_WITHOUT_DATASET=True
SEED=0

SEED=1

export CUDA_VISIBLE_DEVICES=0

# Training params
TRAIN_EPOCHS=50
TRAIN_LR=5e-5
WEIGHT_DECAY=1e-5
GRADIENT_CLIP_VAL=0.0

LINE_RECONSTRUCTION_WEIGHT=4
LINE_CE_WEIGHT=10
BOX_RECONSTRUCTION_WEIGHT=4
BOX_CE_WEIGHT=10
EGO_RECONSTRUCTION_WEIGHT=1
KL_WEIGHT=0.1
NORM_BY_COUNT=False

BATCH_SIZE=64
NUM_WORKERS=16

echo "-------- PYTHON OUTPUT ----------"

python3 $SLEDGE_DEVKIT_ROOT/sledge/script/run_autoencoder.py \
py_func=training \
seed=$SEED \
job_name=$JOB_NAME \
optimizer=adamw \
lr_scheduler=one_cycle_lr \
+autoencoder=training_rvae_model \
autoencoder_checkpoint=$AUTOENCODER_CHECKPOINT \
optimizer.lr=$TRAIN_LR \
optimizer.weight_decay=$WEIGHT_DECAY \
lightning.trainer.params.max_epochs=$TRAIN_EPOCHS \
lightning.trainer.params.gradient_clip_val=$GRADIENT_CLIP_VAL \
data_loader.params.batch_size=$BATCH_SIZE \
data_loader.params.num_workers=$NUM_WORKERS \
cache.autoencoder_cache_path=$AUTOENCODER_CACHE_PATH \
cache.use_cache_without_dataset=$USE_CACHE_WITHOUT_DATASET \
autoencoder_model.config.line_reconstruction_weight=$LINE_RECONSTRUCTION_WEIGHT \
autoencoder_model.config.line_ce_weight=$LINE_CE_WEIGHT \
autoencoder_model.config.box_reconstruction_weight=$BOX_RECONSTRUCTION_WEIGHT \
autoencoder_model.config.box_ce_weight=$BOX_CE_WEIGHT \
autoencoder_model.config.ego_reconstruction_weight=$EGO_RECONSTRUCTION_WEIGHT \
autoencoder_model.config.kl_weight=$KL_WEIGHT \
autoencoder_model.config.norm_by_count=$NORM_BY_COUNT \
worker=single_machine_thread_pool \
callbacks="[learning_rate_monitor_callback, model_checkpoint_callback, time_logging_callback, rvae_visualization_callback]"
# optimizer=adamw_schedulefree \
# lr_scheduler=constant_lr \
