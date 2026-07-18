#!/bin/bash
#SBATCH --job-name=scunet_cxr
#SBATCH --account=mrc-bsu2-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_hformer/logs/scunet_%j.out
#SBATCH --error=/rds/user/stm43/hpc-work/ppvae_hformer/logs/scunet_%j.err

source ~/.bashrc
conda activate ppvae

cd /rds/user/stm43/hpc-work/ppvae_hformer

python scripts/train_scunet.py \
    --data_dir  /rds/user/stm43/hpc-work/chest_xray \
    --output_dir /rds/user/stm43/hpc-work/ppvae_results/baselines/scunet \
    --epochs 200 \
    --batch_size 8 \
    --lr 2e-4 \
    --resume

echo "SCUNet job complete: $(date)"
