#!/bin/bash
#SBATCH --job-name=bm3d_eval
#SBATCH --account=mrc-bsu2-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_results/logs/%j_bm3d_eval.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=stm43@cam.ac.uk

module load cuda/12.1
source ~/.bashrc
conda activate ppvae

cd /rds/user/stm43/hpc-work/ppvae_hformer
python scripts/eval_bm3d.py
