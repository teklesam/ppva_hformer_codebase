#!/bin/bash
#SBATCH --job-name=trmetrics
#SBATCH --account=mrc-bsu2-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_results/logs/trmetrics_%j.log
#SBATCH --error=/rds/user/stm43/hpc-work/ppvae_results/logs/trmetrics_%j.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppvae
cd /rds/user/stm43/hpc-work/ppvae_hformer
python -u scripts/eval_trials_metrics.py
echo "trial metrics complete"
