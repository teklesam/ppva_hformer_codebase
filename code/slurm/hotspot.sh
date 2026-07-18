#!/bin/bash
#SBATCH --job-name=unc_hotspot
#SBATCH --account=mrc-bsu2-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_hformer/logs/unc_hotspot_%j.out

source ~/.bashrc
conda activate ppvae

cd /rds/user/stm43/hpc-work/ppvae_hformer
python scripts/generate_uncertainty_hotspot.py
