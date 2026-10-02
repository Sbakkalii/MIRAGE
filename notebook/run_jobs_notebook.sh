#!/bin/bash

#SBATCH --partition=gpu-2080ti
#SBATCH --time=02:00:00
#SBATCH --job-name=JupyterNotebook
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh

ENV_NAME="notebook"

if ! conda info --envs | grep -q "^${ENV_NAME}"; then
    conda create -n ${ENV_NAME} python=3.11 -y
fi

conda activate ${ENV_NAME}
pip install notebook

jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
