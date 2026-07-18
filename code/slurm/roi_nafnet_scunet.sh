#!/bin/bash
#SBATCH --job-name=roi_sota
#SBATCH --account=mrc-bsu2-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_hformer/logs/roi_sota_%j.out
#SBATCH --error=/rds/user/stm43/hpc-work/ppvae_hformer/logs/roi_sota_%j.err

source ~/.bashrc
conda activate ppvae
cd /rds/user/stm43/hpc-work/ppvae_hformer

python scripts/generate_nafnet_scunet_roi.py

echo "ROI job complete: $(date)"
