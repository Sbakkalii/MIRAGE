#!/bin/bash
#SBATCH --partition=gpu-h100
#SBATCH --time=48:00:00
#SBATCH --job-name=benchmark_embedding
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=200GB
#SBATCH --gres=gpu:3

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate rag_env

python embedding_benchmark.py