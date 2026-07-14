#!/usr/bin/env python
"""Per-image PSNR/SSIM/LPIPS with the three-way diagnostic label (Normal /
Bacterial / Viral) for the five Table-4.7 models INCLUDING SCUNet, in one
consistent Foi-noise pipeline. Output feeds a ggstatsplot subgroup figure."""
import os, sys, glob, math, csv, numpy as np, torch, piq
from PIL import Image
PROJECT = "/rds/user/stm43/hpc-work/ppvae_hformer"
sys.path.insert(0, PROJECT + "/scripts")

# -- PP-VAE arms + DnCNN via the ROI loader (handles the (mu, lsa, ...) tuple) --
import generate_roi_panels_v2 as G
allm = G.load_all_models()
KEEP = ("arm_a", "arm_d", "arm_h", "dncnn")
models = {k: allm[k] for k in KEEP if allm.get(k) is not None}
del allm
torch.cuda.empty_cache()

# -- SCUNet via the head of the SOTA script (reuses exact weights) --
sota = PROJECT + "/scripts/generate_nafnet_scunet_roi.py"
src = open(sota).read(); i = src.index("Benchmarking inference time")
gg = {"__name__": "loaded_head"}
exec(compile(src[:src.rfind("\n", 0, i)], sota, "exec"), gg)
scunet, DEVICE = gg["scunet"], gg["DEVICE"]
print("loaded:", list(models) + ["scunet"], flush=True)

DATA = "/rds/user/stm43/hpc-work/chest_xray/test"
FOI_A, FOI_B = 0.03, 0.005
lpips_fn = piq.LPIPS(reduction="none").to(DEVICE)
def load_img(p): return (np.array(Image.open(p).convert("L").resize((256, 256))) / 255.0).astype(np.float32)
def add_noise(c, seed):
    rng = np.random.default_rng(seed)
    return np.clip(c + rng.standard_normal(c.shape) * np.sqrt(FOI_A * np.clip(c, 0, None) + FOI_B), 0, 1).astype(np.float32)
def psnr_np(a, b):
    mse = np.mean((a - b) ** 2); return 100.0 if mse < 1e-14 else 20 * math.log10(1.0 / math.sqrt(mse))

# build (path, class) list; class from folder + filename prefix
items = []
for dirn, base in (("NORMAL", None), ("PNEUMONIA", None)):
    for p in sorted(glob.glob(f"{DATA}/{dirn}/*.jpeg") + glob.glob(f"{DATA}/{dirn}/*.png")):
        if dirn == "NORMAL":
            cls = "Normal"
        else:
            cls = "Bacterial" if os.path.basename(p).upper().startswith("BACTERIA") else "Viral"
        items.append((p, cls))
from collections import Counter
print("images:", len(items), dict(Counter(c for _, c in items)), flush=True)

MODEL_LABEL = {"arm_a": "A (L2)", "arm_d": "D (NLL+SSIM+FFL)", "arm_h": "H (PP-VAE)",
               "dncnn": "DnCNN", "scunet": "SCUNet"}
rows = []
for idx, (p, cls) in enumerate(items):
    clean = load_img(p); noisy = add_noise(clean, idx + 42)
    y = torch.tensor(clean)[None, None].to(DEVICE)
    for key in list(models) + ["scunet"]:
        if key == "scunet":
            x = torch.tensor(noisy)[None, None].to(DEVICE)
            with torch.no_grad():
                recon_t = scunet(x).clamp(0, 1)
            recon = recon_t.squeeze().cpu().numpy()
        else:
            recon, _ = G.infer(models[key], torch.tensor(noisy)[None].float(), key)
            recon_t = torch.tensor(recon)[None, None].to(DEVICE).clamp(0, 1)
        lp = lpips_fn(recon_t.repeat(1, 3, 1, 1), y.repeat(1, 3, 1, 1)).item()
        ss = piq.ssim(recon_t, y, data_range=1.0).item()
        rows.append((MODEL_LABEL[key], cls, psnr_np(recon, clean), ss, lp))
    if idx % 100 == 0:
        print(f"  {idx}/{len(items)}", flush=True)

OUT = "/rds/user/stm43/hpc-work/ppvae_results/evaluation/subgroup_perimage.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "class", "psnr", "ssim", "lpips"]); w.writerows(rows)
print("wrote", OUT, len(rows), "rows", flush=True)
