#!/bin/bash
#SBATCH --job-name=armb
#SBATCH --account=mrc-bsu2-sl2-cpu
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_hformer/logs/armb_%j.log
source ~/.bashrc; conda activate ppvae; cd /rds/user/stm43/hpc-work/ppvae_hformer
python -u scripts/_arm_b_uncertainty_006.py; echo DONE
