#!/bin/bash
#SBATCH --partition=gpu-h100
#SBATCH --time=48:00:00
#SBATCH --job-name=benchmark_reranking
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=250GB
#SBATCH --gres=gpu:4

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate rag_env

python retrieval_benchmark.py