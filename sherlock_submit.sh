#!/bin/sh
#
#SBATCH --job-name=TEST_diffusion_training_script
#SBATCH --time=0:30:00
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C 'GPU_GEN:VLT&GPU_MEM:16GB'
#SBATCH --gpu_cmode=shared

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

CUDA_VISIBLE_DEVICES=0 python main.py --base configs/latent-diffusion/CUSTOM-ldm-vq-4.yaml -t --gpus 0 -l "/scratch/users/shounak/LOGS-diffusion-for-auto"
echo "SHERLOCK: Command Execution Complete"
