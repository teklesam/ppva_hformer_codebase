"""Dump arm-D reconstructions at four (lambda_ssim, lambda_ffl) points from the
Optuna sweep checkpoints, for a single Normal case, so a ROI montage can show how
loss-weighting changes fine structural detail. Supplements the sensitivity figure."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # project root -> src.*
sys.path.insert(0, HERE)                     # scripts/ -> generate_roi_panels_v2
import generate_roi_panels_v2 as G
import torch
from src.models.ppvae_hformer import PPVAEHformer
from src.training.config import ABLATION_ARMS, ExperimentConfig

SWEEP = "/rds/user/stm43/hpc-work/ppvae_results/optuna_sweep_d"
OUT   = "/rds/user/stm43/hpc-work/ppvae_results/lambda_roi_recons.npz"
CASE_CLS, CASE_IDX, ETA = "NORMAL", 9, 200
# (trial, lambda_ssim, lambda_ffl, sweep val_psnr) — spread across the landscape
TRIALS = [(6,0.119,0.454,30.77),(0,0.500,0.100,31.54),(12,0.609,0.023,32.13),(10,0.957,0.020,32.78)]

def build_arm_d():
    cfg = ExperimentConfig()
    for k,v in ABLATION_ARMS["arm_d_nll_ssim_ffl"].get("model",{}).items():
        setattr(cfg.model,k,v)
    return PPVAEHformer(in_channels=cfg.model.in_channels, base_channels=cfg.model.base_channels,
        num_blocks=cfg.model.num_blocks, num_scales=cfg.model.num_scales, num_heads=cfg.model.num_heads,
        window_size=cfg.model.window_size, use_vae=cfg.model.use_vae,
        activation=getattr(cfg.model,"activation","gelu"))

import piq
_lpips = piq.LPIPS(reduction="none").to(G.DEVICE)
clean = G.load_image(CASE_CLS, CASE_IDX)
noisy = G.add_noise(clean, ETA, seed=42)
noisy_t = torch.from_numpy(noisy).float().unsqueeze(0)
_y = torch.from_numpy(clean)[None,None].float().to(G.DEVICE)
def perc(img):  # FSIM + LPIPS of img vs clean, on the piq scale (matches the thesis)
    o = torch.from_numpy(np.clip(img,0,1).astype(np.float32))[None,None].to(G.DEVICE)
    fs = piq.fsim(o,_y,data_range=1.0,chromatic=False).item()
    lp = _lpips(o.repeat(1,3,1,1), _y.repeat(1,3,1,1)).item()
    return fs, lp
nf, nl = perc(noisy)
print(f"case {CASE_CLS}/{CASE_IDX}  noisy PSNR={G.psnr(clean,noisy):.2f}  device={G.DEVICE}", flush=True)
recons, labels, psnrs, fsims, lpipss = [], [], [], [], []
for tr,ss,ff,vp in TRIALS:
    ckpt = os.path.join(SWEEP, f"trial_{tr:03d}", "best_model.pth")
    m = build_arm_d(); m.load_state_dict(G._load_ckpt(ckpt)); m.to(G.DEVICE).eval()
    recon,_ = G.infer(m, noisy_t, "arm_d")
    fs, lp = perc(recon)
    recons.append(recon.astype(np.float32)); labels.append(f"ssim={ss:.2f} ffl={ff:.3f}")
    psnrs.append(G.psnr(clean, recon)); fsims.append(fs); lpipss.append(lp)
    print(f"  trial {tr:2d}: ssim={ss:.3f} ffl={ff:.3f} PSNR={psnrs[-1]:.3f} FSIM={fs:.3f} LPIPS={lp:.3f}", flush=True)
np.savez_compressed(OUT, clean=clean, noisy=noisy, recons=np.stack(recons),
    labels=np.array(labels), psnrs=np.array(psnrs,dtype=np.float32),
    fsims=np.array(fsims,dtype=np.float32), lpipss=np.array(lpipss,dtype=np.float32),
    noisy_psnr=np.float32(G.psnr(clean,noisy)), noisy_fsim=np.float32(nf), noisy_lpips=np.float32(nl),
    trials=np.array([t[0] for t in TRIALS]), ssim=np.array([t[1] for t in TRIALS]),
    ffl=np.array([t[2] for t in TRIALS]), sweep_psnr=np.array([t[3] for t in TRIALS],dtype=np.float32))
print("wrote", OUT, flush=True)
