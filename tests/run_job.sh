#!/bin/bash
#SBATCH --partition=gpu-h100
#SBATCH --time=4:00:00
#SBATCH --job-name=test_preprocessing
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=120GB
#SBATCH --gres=gpu:2

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate rag_env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python test_preprocessing.py