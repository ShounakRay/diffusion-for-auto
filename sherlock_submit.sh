#!/bin/sh
#
#SBATCH --job-name=trial
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p gpu
#SBATCH -G 2
#SBATCH -C 'GPU_GEN:VLT&GPU_MEM:16GB'
#SBATCH --gpu_cmode=shared

ml load gcc/6.3.0
export CUDA_HOME=$CONDA_PREFIX
. ~/.bashrc
conda init bash
conda activate ldm-shounak
CUDA_VISIBLE_DEVICES=0,1 python main.py --base configs/latent-diffusion/CUSTOM-ldm-vq-4.yaml -t --gpus 0,1 -l "/home/shounak/LOGS/LOGS-diffusion-for-auto"
