#!/bin/bash
#SBATCH --job-name=bm3d_roi
#SBATCH --account=mrc-bsu2-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_hformer/logs/bm3d_roi_%j.log

source ~/.bashrc
conda activate ppvae

cd /rds/user/stm43/hpc-work/ppvae_hformer
python scripts/generate_bm3d_roi_panel.py
