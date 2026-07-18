#!/usr/bin/env bash
# deploy_csd3.sh  -- sync code to CSD3, verify imports, resubmit arms B-E
#
# Usage (from Mac terminal):
#   bash deploy_csd3.sh              # sync + verify + submit B-E
#   bash deploy_csd3.sh --sync-only  # just sync, no job submission
#
# Run from any directory; paths are absolute.

set -euo pipefail

HPC_USER="stm43"
HPC_HOST="login.hpc.cam.ac.uk"
REMOTE_PROJECT="/rds/user/stm43/hpc-work/ppvae_hformer"
REMOTE_VENV="/rds/user/stm43/hpc-work/ppvae_env"
REMOTE_DATA="/home/stm43/chest_xray"
REMOTE_RESULTS="/rds/user/stm43/hpc-work/ppvae_results"
LOCAL_SRC="/Users/sam/Documents/PPVAE Dissertation Project/PpCNN/code/ppvae_hformer"

SYNC_ONLY="${1:-}"

# ── 1. Rsync code ──────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Step 1: Syncing code → CSD3"
echo "========================================================"

rsync -avz --progress \
    --exclude='.DS_Store' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.pth' \
    --exclude='slurm/slurm-*.out' \
    --exclude='options/' \
    "${LOCAL_SRC}/" \
    "${HPC_USER}@${HPC_HOST}:${REMOTE_PROJECT}/"

echo ""
echo "Sync complete."

[[ "${SYNC_ONLY}" == "--sync-only" ]] && echo "Stopping after sync (--sync-only)." && exit 0

# ── 2. Verify structure + imports on CSD3 ─────────────────────────────────────
echo ""
echo "========================================================"
echo " Step 2: Verifying environment on CSD3"
echo "========================================================"

ssh "${HPC_USER}@${HPC_HOST}" bash -s -- \
    "${REMOTE_PROJECT}" "${REMOTE_VENV}" "${REMOTE_DATA}" <<'REMOTE_VERIFY'

PROJECT="$1"
VENV="$2"
DATA="$3"

set -euo pipefail

echo ""
echo "--- Project tree ---"
find "${PROJECT}/src" -name "*.py" | sort

echo ""
echo "--- Data directory ---"
if [ -d "${DATA}" ]; then
    echo "  chest_xray/ exists"
    ls "${DATA}/"
else
    echo "  ERROR: ${DATA} not found!"
    exit 1
fi

echo ""
echo "--- Activating venv and installing requirements ---"
source "${VENV}/bin/activate"
pip install -q -r "${PROJECT}/requirements.txt"
echo "  pip install OK"

echo ""
echo "--- Testing all imports ---"
cd "${PROJECT}"
python -c "
import sys
sys.path.insert(0, '.')

from src.models.ppvae_hformer import PPVAEHformer
from src.losses.composite_loss import PPVAEHformerLoss
from src.losses.nll_loss import HeteroscedasticNLL
from src.losses.ms_ssim_loss import MSSSIMLoss
from src.losses.focal_frequency import FocalFrequencyLoss
from src.losses.kl_loss import KLDivergence
from src.data.kermany_dataset import make_loaders
from src.data.noise_simulation import add_foi_noise_preset
from src.evaluation.metrics import psnr, ssim
from src.training.config import ABLATION_ARMS

import torch
# Quick sanity: build model and run forward pass
m = PPVAEHformer(base_channels=32, use_vae=False)
x = torch.randn(1, 1, 64, 64)
mu, lsa, _, _ = m(x)
assert mu.shape == x.shape, f'bad shape: {mu.shape}'

# NLL must stay float32 even when inputs are float16
nll = HeteroscedasticNLL()
mu16 = mu.half()
lsa16 = lsa.half()
y16 = x.half()
loss = nll(y16, mu16, lsa16)
assert loss.dtype == torch.float32, f'NLL returned {loss.dtype}, expected float32'

print('All imports and float32 NLL check: PASS')
"

REMOTE_VERIFY

# ── 3. Cancel pending jobs and resubmit B-E ───────────────────────────────────
echo ""
echo "========================================================"
echo " Step 3: Cancelling pending jobs and submitting arms B-E"
echo "========================================================"

ssh "${HPC_USER}@${HPC_HOST}" bash -s -- "${REMOTE_PROJECT}" <<'REMOTE_SUBMIT'

PROJECT="$1"
cd "${PROJECT}"

# Cancel any queued or running ppvae jobs (not Arm A results — those are saved to disk)
PENDING=$(squeue -u stm43 --state=PD -h -o "%i" 2>/dev/null || true)
RUNNING=$(squeue -u stm43 --state=R  -h -o "%i" 2>/dev/null || true)

for JID in ${PENDING} ${RUNNING}; do
    echo "  Cancelling job ${JID}"
    scancel "${JID}" 2>/dev/null || true
done

echo ""
echo "Submitting arm_b_nll ..."
JB=$(sbatch slurm/train_proposed.sh arm_b_nll | awk '{print $NF}')
echo "  -> job ${JB}"

echo "Submitting arm_c_nll_ssim ..."
JC=$(sbatch slurm/train_proposed.sh arm_c_nll_ssim | awk '{print $NF}')
echo "  -> job ${JC}"

echo "Submitting arm_d_nll_ssim_ffl ..."
JD=$(sbatch slurm/train_proposed.sh arm_d_nll_ssim_ffl | awk '{print $NF}')
echo "  -> job ${JD}"

echo "Submitting arm_e_ppvae ..."
JE=$(sbatch slurm/train_proposed.sh arm_e_ppvae | awk '{print $NF}')
echo "  -> job ${JE}"

echo ""
echo "=== Queue ==="
squeue -u stm43

echo ""
echo "Monitor with:"
echo "  squeue -u stm43"
echo "  tail -f /rds/user/stm43/hpc-work/ppvae_results/logs/<JOBID>_ppvae.out"

REMOTE_SUBMIT

echo ""
echo "All done."
