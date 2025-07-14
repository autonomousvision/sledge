#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=64         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --partition=h100-ferranti # gpu-v100-preemptable, gpu-2080ti-preemptable
#SBATCH --gres=gpu:4              # optionally type and number of gpus
#SBATCH --mem=128G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logs/rvae_job_%j.out  # File to which STDOUT will be written
#SBATCH --error=logs/rvae_job_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=FAIL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=daniel.dauner@uni-tuebingen.de  # Email to which notifications will be sent

# export GLOO_SOCKET_IFNAME=eth0  # or any interface that supports IPv4
# The first hostname is the master address

# print info about current job
echo "---------- JOB INFOS ------------"
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus

echo "---------- JOB SETUP ------------"
# Due to a potential bug, we need to manually load our bash configurations first
source $HOME/.bashrc

# Setup Phase
# # - copy singularity images or datasets to local on-compute-node storage like /scratch_local
# echo "Setup scratch_local folder..."
# # mkdir /scratch_local/gwb883-$SLURM_JOB_ID/autoencoder_cache

# # cp -r $WORK/sledge_workspace/exp/caches/train_rvae.zip /scratch_local/$SLURM_JOB_USER-$SLURM_JOBID/train_rvae.zip
# # unzip -q /scratch_local/$SLURM_JOB_USER-$SLURM_JOBID/train_rvae.zip -d /scratch_local/$SLURM_JOB_USER-$SLURM_JOBID
# cp -r $WORK/sledge_workspace/exp/caches/train_rvae.zip /scratch_local/$SLURM_JOB_USER-$SLURM_JOBID/train_rvae.zip
# unzip -q /scratch_local/$SLURM_JOB_USER-$SLURM_JOBID/train_rvae.zip -d /scratch_local/$SLURM_JOB_USER-$SLURM_JOBID

# echo "Setup scratch_local folder...DONE!"

# - loads virtual envs, like with anaconda
echo "Setup conda..."
conda activate sledge
echo "Setup conda...DONE!"

# - set environment variables
echo "Setup environment variables..."
export NUPLAN_DATA_ROOT="$HOME/nuplan/dataset"
export NUPLAN_MAPS_ROOT="$HOME/nuplan/dataset/maps"
export SLEDGE_EXP_ROOT="$HOME/sledge_workspace/exp"
export SLEDGE_DEVKIT_ROOT="$HOME/sledge_workspace/sledge"
export RAY_DEDUP_LOGS=0
export MASTER_ADDR=localhost
echo "Setup environment variables...DONE!"

# - determine commandline arguments for `srun` calls

JOB_NAME=original_long
AUTOENCODER_CACHE_PATH=$HOME/sledge_workspace/cache/autoencoder_cache
AUTOENCODER_CHECKPOINT=null
USE_CACHE_WITHOUT_DATASET=True
SEED=1

# Training params
TRAIN_EPOCHS=300
TRAIN_LR=5e-4
WEIGHT_DECAY=1e-4
GRADIENT_CLIP_VAL=0.0

LINE_RECONSTRUCTION_WEIGHT=4
LINE_CE_WEIGHT=10
BOX_RECONSTRUCTION_WEIGHT=2
BOX_CE_WEIGHT=5
EGO_RECONSTRUCTION_WEIGHT=1
KL_WEIGHT=0.1
NORM_BY_COUNT=False

LINE_DOT_RADIUS=0

FINAL_DIV_FACTOR=10


BATCH_SIZE=32
NUM_WORKERS=16

echo "-------- PYTHON OUTPUT ----------"

python3 $SLEDGE_DEVKIT_ROOT/sledge/script/run_autoencoder.py \
py_func=training \
seed=$SEED \
job_name=$JOB_NAME \
+autoencoder=training_rvae_model \
autoencoder_checkpoint=$AUTOENCODER_CHECKPOINT \
optimizer.lr=$TRAIN_LR \
optimizer.weight_decay=$WEIGHT_DECAY \
lr_scheduler.max_lr=$TRAIN_LR \
lightning.trainer.params.max_epochs=$TRAIN_EPOCHS \
lightning.trainer.params.gradient_clip_val=$GRADIENT_CLIP_VAL \
data_loader.params.batch_size=$BATCH_SIZE \
data_loader.params.num_workers=$NUM_WORKERS \
data_augmentation=rvae_no_augmentation \
cache.autoencoder_cache_path=$AUTOENCODER_CACHE_PATH \
cache.use_cache_without_dataset=$USE_CACHE_WITHOUT_DATASET \
autoencoder_model.config.line_reconstruction_weight=$LINE_RECONSTRUCTION_WEIGHT \
autoencoder_model.config.line_ce_weight=$LINE_CE_WEIGHT \
autoencoder_model.config.box_reconstruction_weight=$BOX_RECONSTRUCTION_WEIGHT \
autoencoder_model.config.box_ce_weight=$BOX_CE_WEIGHT \
autoencoder_model.config.ego_reconstruction_weight=$EGO_RECONSTRUCTION_WEIGHT \
autoencoder_model.config.kl_weight=$KL_WEIGHT \
autoencoder_model.config.norm_by_count=$NORM_BY_COUNT \
autoencoder_model.config.line_dots_radius=$LINE_DOT_RADIUS \
lr_scheduler.final_div_factor=$FINAL_DIV_FACTOR \
worker=single_machine_thread_pool \
callbacks="[learning_rate_monitor_callback, model_checkpoint_callback, time_logging_callback, rvae_visualization_callback]"
# data_augmentation=rvae_no_augmentation \

echo "-------- DONE ----------"

conda deactivate
