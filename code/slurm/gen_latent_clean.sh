#!/bin/bash
#SBATCH --job-name=latent_clean
#SBATCH --account=mrc-bsu2-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_results/logs/latent_clean_%j.out
#SBATCH --error=/rds/user/stm43/hpc-work/ppvae_results/logs/latent_clean_%j.err

source /home/stm43/miniconda3/etc/profile.d/conda.sh
conda activate ppvae

cd /rds/user/stm43/hpc-work/ppvae_hformer
python3 /rds/user/stm43/hpc-work/ppvae_hformer/scripts/generate_latent_clean.py
