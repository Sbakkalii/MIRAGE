#!/bin/bash
#SBATCH --partition=gpu-a40
#SBATCH --time=06:00:00
#SBATCH --job-name=benchmark_ner
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate stage_l3i

python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
python -m spacy download fr_dep_news_trf
python -m spacy download fr_core_news_sm
python -m spacy download xx_ent_wiki_sm
python -m spacy download xx_sent_ud_sm

python NER_benchmark.py