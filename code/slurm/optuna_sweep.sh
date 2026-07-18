#!/bin/bash
#SBATCH --job-name=optuna_d
#SBATCH --account=mrc-bsu2-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=34:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_hformer/logs/optuna_d_%j.log
source ~/.bashrc
conda activate ppvae
cd /rds/user/stm43/hpc-work/ppvae_hformer
export SWEEP_EPOCHS=40
export SWEEP_TRIALS=14
python -u scripts/optuna_sweep.py
