#!/bin/bash
#SBATCH --job-name=nafnet_cxr
#SBATCH --account=mrc-bsu2-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_hformer/logs/nafnet_%j.out
#SBATCH --error=/rds/user/stm43/hpc-work/ppvae_hformer/logs/nafnet_%j.err

source ~/.bashrc
conda activate ppvae

cd /rds/user/stm43/hpc-work/ppvae_hformer

python scripts/train_nafnet.py \
    --data_dir  /rds/user/stm43/hpc-work/chest_xray \
    --output_dir /rds/user/stm43/hpc-work/ppvae_results/baselines/nafnet \
    --epochs 200 \
    --batch_size 16 \
    --lr 2e-4 \
    --resume

echo "NAFNet job complete: $(date)"
