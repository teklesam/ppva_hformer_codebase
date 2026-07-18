#!/bin/bash
#SBATCH --job-name=ppvae_eval_qrs
#SBATCH --account=mrc-bsu2-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_results/logs/%j_eval_qrs.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=stm43@cam.ac.uk

module load cuda/12.1
source ~/.bashrc
conda activate ppvae

cd /rds/user/stm43/hpc-work/ppvae_hformer

python scripts/evaluate_all.py   --data_dir    /rds/user/stm43/hpc-work/chest_xray   --results_dir /rds/user/stm43/hpc-work/ppvae_results   --output_dir  /rds/user/stm43/hpc-work/ppvae_results/evaluation_qrs   --arms arm_q_charb arm_r_ft_j arm_s_ft_d   --noise_levels low mid high   --bootstrap_n 1000
