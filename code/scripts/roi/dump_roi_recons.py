#!/usr/bin/env python
"""Dump per-arm reconstructions for a single case so ROI montages can be
assembled offline (no GPU) with any choice of ROI regions.

Reuses model loading + inference from generate_roi_panels_v2.py.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_roi_panels_v2 as G

CASE_CLS = "NORMAL"
CASE_IDX = 9          # vessel-rich normal (panel A / hilum case)
ETA      = 200
OUT      = "/rds/user/stm43/hpc-work/ppvae_results/roi_dump_normal9.npz"

# arm key -> display label (columns, left to right, all 24 conditions)
ORDER = [
    ("arm_a","A L2"),("arm_i","I L1"),("arm_j","J L1+SSIM+FFL"),("arm_q","Q Charb"),
    ("arm_r","R FT-L1"),("arm_s","S FT-NLL"),
    ("arm_b","B NLL"),("arm_c","C NLL+SSIM"),("arm_d","D NLL+SSIM+FFL"),
    ("arm_k","K NLL+L1"),("arm_l","L NLL+Edge+FFL"),("arm_m","M +Edge"),("arm_n","N +Perc"),
    ("arm_e","E VAE lin"),("arm_f","F VAE cyc"),("arm_g","G VAE fb"),
    ("arm_h","H VAE cyc+fb"),("arm_p","P VAE best"),("arm_o","O PReLU"),
    ("dncnn","DnCNN"),("ircnn","IRCNN"),("ffdnet","FFDNet"),("drunet","DRUNet"),("swinir","SwinIR"),
]

def main():
    import torch
    clean = G.load_image(CASE_CLS, CASE_IDX)
    noisy = G.add_noise(clean, ETA, seed=42)
    noisy_t = torch.from_numpy(noisy).float().unsqueeze(0)   # [1,H,W]: add channel dim (infer() adds batch)
    print(f"case {CASE_CLS}/{CASE_IDX}  img {clean.shape}  noisy PSNR={G.psnr(clean,noisy):.2f}")

    models = G.load_all_models()
    recons, labels, keys, psnrs = [], [], [], []
    for key, lab in ORDER:
        m = models.get(key)
        if m is None:
            print(f"  [skip] {key} (no model)"); continue
        recon, _ = G.infer(m, noisy_t, key)
        recons.append(recon.astype(np.float32)); labels.append(lab); keys.append(key)
        psnrs.append(G.psnr(clean, recon))
        print(f"  {key:8s} PSNR={psnrs[-1]:.3f}")

    np.savez_compressed(OUT, clean=clean, noisy=noisy,
                        recons=np.stack(recons), labels=np.array(labels),
                        keys=np.array(keys), psnrs=np.array(psnrs, dtype=np.float32))
    print(f"wrote {OUT}  ({len(keys)} arms)")

if __name__ == "__main__":
    main()
