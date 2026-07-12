"""
preset_stats.py — Statistical characterisation of the three Foi degradation presets.

Computes per-image pre-denoising PSNR (noisy input vs clean reference) over the
whole test set for each preset, then tests the two properties claimed in the
Results:
  (1) the three severity levels are well-separated  (one-way ANOVA + pairwise
      Welch t-test + Cohen's d across Low/Mid/High);
  (2) the degradation is class-agnostic             (per level, Welch t-test +
      Cohen's d for Normal vs Pneumonia).

Preprocessing matches the dataset loader (greyscale, to_tensor, antialiased
resize to 256x256); noise uses the documented Foi law Var(z|y)=a*y+b with a
fixed per-image seed.

Usage:
    python scripts/preset_stats.py --data /path/to/chest_xray/test
"""
from __future__ import annotations
import os, sys, glob, argparse
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.stats import f_oneway, ttest_ind

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.noise_simulation import PRESETS


def cohend(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / sp


def psnr(clean, noisy):
    mse = float(((noisy - clean) ** 2).mean())
    return -10 * np.log10(max(mse, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="test dir with NORMAL/ and PNEUMONIA/ subfolders")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    resize = T.Resize((256, 256), antialias=True)
    files = [(f, "Normal") for f in sorted(glob.glob(f"{args.data}/NORMAL/*.jpeg"))]
    files += [(f, "Pneumonia") for f in sorted(glob.glob(f"{args.data}/PNEUMONIA/*.jpeg"))]
    cleans = [(resize(TF.to_tensor(Image.open(f).convert("L"))), c) for f, c in files]
    print(f"loaded {len(cleans)} images; mean intensity={np.mean([t.mean().item() for t,_ in cleans]):.3f}")

    data = {}
    for name in ["low", "mid", "high"]:
        a, b = PRESETS[name]["a"], PRESETS[name]["b"]
        allp, norm, pneu = [], [], []
        for i, (clean, cls) in enumerate(cleans):
            g = torch.Generator(); g.manual_seed(args.seed + i)
            noise = torch.randn(clean.shape, generator=g)
            noisy = (clean + noise * (a * clean.clamp(min=0.0) + b).sqrt()).clamp(0.0, 1.0)
            p = psnr(clean, noisy); allp.append(p)
            (norm if cls == "Normal" else pneu).append(p)
        data[name] = {k: np.array(v) for k, v in dict(all=allp, Normal=norm, Pneumonia=pneu).items()}

    print("\n== Claim 1: three well-separated severity levels ==")
    F, pF = f_oneway(*[data[n]["all"] for n in ["low", "mid", "high"]])
    print(f"one-way ANOVA: F={F:.1f}, p={pF:.2e}")
    for x, y in [("low", "mid"), ("mid", "high"), ("low", "high")]:
        t, pv = ttest_ind(data[x]["all"], data[y]["all"], equal_var=False)
        print(f"  {x} vs {y}: dPSNR={data[x]['all'].mean()-data[y]['all'].mean():+.2f} dB, "
              f"Welch t={t:.1f}, p={pv:.2e}, Cohen d={cohend(data[x]['all'], data[y]['all']):+.2f}")

    print("\n== Claim 2: Normal vs Pneumonia per level ==")
    for name in ["low", "mid", "high"]:
        N, P = data[name]["Normal"], data[name]["Pneumonia"]
        t, pv = ttest_ind(N, P, equal_var=False)
        print(f"  {name}: Normal={N.mean():.2f}+/-{N.std(ddof=1):.2f} vs Pneumonia={P.mean():.2f}+/-{P.std(ddof=1):.2f}, "
              f"dPSNR={N.mean()-P.mean():+.3f}, Welch t={t:.2f}, p={pv:.2f}, Cohen d={cohend(N, P):+.3f}")


if __name__ == "__main__":
    main()
