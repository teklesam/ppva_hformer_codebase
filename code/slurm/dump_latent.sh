#!/bin/bash
#SBATCH --job-name=latent_emb
#SBATCH --account=mrc-bsu2-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_hformer/logs/latent_emb_%j.log
source ~/.bashrc
conda activate ppvae
cd /rds/user/stm43/hpc-work/ppvae_hformer
python -u scripts/dump_latent_embeddings.py
