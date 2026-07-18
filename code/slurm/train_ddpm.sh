#!/bin/bash
#SBATCH --job-name=ppvae_ddpm
#SBATCH --account=mrc-bsu2-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_results/logs/%j_ddpm_train.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=stm43@cam.ac.uk

module load cuda/12.1
source ~/.bashrc
conda activate ppvae

cd /rds/user/stm43/hpc-work/ppvae_hformer

python scripts/train_ddpm.py \
    --data_dir    /rds/user/stm43/hpc-work/chest_xray \
    --out_dir     /rds/user/stm43/hpc-work/ppvae_results/arm_ddpm \
    --epochs      200 \
    --batch_size  16 \
    --lr          1e-4 \
    --T           1000 \
    --ddim_steps  50 \
    --num_workers 4
