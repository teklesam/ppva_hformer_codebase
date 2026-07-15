#!/bin/bash
#SBATCH --job-name=optuna_m
#SBATCH --account=mrc-bsu2-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_results/logs/optuna_m_%j.log
#SBATCH --error=/rds/user/stm43/hpc-work/ppvae_results/logs/optuna_m_%j.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppvae
cd /rds/user/stm43/hpc-work/ppvae_hformer
export SWEEP_EPOCHS=40 SWEEP_TRIALS=14
python -u scripts/optuna_sweep_edge.py
echo "optuna edge sweep complete"
