#!/bin/sh
#
#SBATCH --job-name=DEVGPU_waymo_training
#SBATCH --time=1:00:00
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH -p gpu
#SBATCH -G 1
#S____BATCH -C 'GPU_GEN:VLT&GPU_MEM:16GB'
#S____BATCH --gpu_cmode=shared

#############################
######### SETTINGS ##########
#############################

# Either `waymo` or `nuimages`
readonly DATASET_NAME="waymo"
# Do not modify
readonly CONFIG_PATH="configs/latent-diffusion/$DATASET_NAME-ldm-vq-4.yaml"
# Do not modify
readonly LOG_PATH="$SCRATCH/LOGS/diffusion-for-auto/$DATASET_NAME"

#############################
#############################

echo "SHERLOCK: Loading GCC"
ml load gcc/6.3.0

echo "SHERLOCK: Loading conda"
export CUDA_HOME=$CONDA_PREFIX

## METHOD 1: Provided by Bernard (doesn't work here)
# source ~/.bashrc
# conda init bash

### METHOD 2: Suggested by GitHub Issues Comment
###           Reference: https://github.com/conda/conda/issues/7980#issuecomment-441358406
source $GROUP_HOME/miniconda3/etc/profile.d/conda.sh

### METHOD 3: Include these lines .bashrc to support conda references from shell scripts
###	      Reference: https://github.com/conda/conda/issues/7980#issuecomment-472651966
# export -f conda
# export -f __conda_activate
# export -f __conda_reactivate
# export -f __conda_hashr
## After adding, include the following line in this submission script:
# source ~./bashrc

conda activate ldm-shounak
echo "SHERLOCK: If there are no error messages before this comment, we activated conda successfully."

echo "SHERLOCK: Current Working Directory $PWD"

CUDA_VISIBLE_DEVICES=0 python3 main.py --base $CONFIG_PATH --gpus=0, -t -l $LOG_PATH
echo "SHERLOCK: Command Execution Complete"
