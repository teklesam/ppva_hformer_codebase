#!/bin/bash
#SBATCH --job-name=subgroup_pi
#SBATCH --account=mrc-bsu2-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_results/logs/subgroup_pi_%j.log
#SBATCH --error=/rds/user/stm43/hpc-work/ppvae_results/logs/subgroup_pi_%j.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppvae
cd /rds/user/stm43/hpc-work/ppvae_hformer
python -u scripts/subgroup_perimage.py
echo "subgroup per-image complete"
