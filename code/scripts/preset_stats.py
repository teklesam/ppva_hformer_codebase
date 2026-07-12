"""
preset_stats.py — Statistical characterisation of the three Foi degradation presets.

Computes per-image pre-denoising PSNR (noisy input vs clean reference) over the
whole test set for each preset, then tests the two properties claimed in the
Results:
  (1) the three severity levels are well-separated  (one-way ANOVA + pairwise
      Welch t-test + Cohen's d across Low/Mid/High);
  (2) the degradation is class-agnostic             (per level, Welch t-test +
      Cohen's d for every pair of the three diagnostic classes Normal /
      Bacterial / Viral).

Preprocessing matches the dataset loader (greyscale, to_tensor, antialiased
resize to 256x256); noise uses the documented Foi law Var(z|y)=a*y+b with a
fixed per-image seed. Classes are read from folder / filename: NORMAL/*, and
PNEUMONIA/* split into Bacterial vs Viral by the BACTERIA/VIRUS filename prefix.

Usage:
    python scripts/preset_stats.py --data /path/to/chest_xray/test
"""
from __future__ import annotations
import os, sys, glob, argparse
from itertools import combinations
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.stats import f_oneway, ttest_ind

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.noise_simulation import PRESETS

CLASSES = ["Normal", "Bacterial", "Viral"]


def cohend(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / sp


def psnr(clean, noisy):
    return -10 * np.log10(max(float(((noisy - clean) ** 2).mean()), 1e-12))


def label(path):
    if f"{os.sep}NORMAL{os.sep}" in path or path.upper().split(os.sep)[-2] == "NORMAL":
        return "Normal"
    return "Bacterial" if "bacteria" in os.path.basename(path).lower() else "Viral"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="test dir with NORMAL/ and PNEUMONIA/ subfolders")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    resize = T.Resize((256, 256), antialias=True)
    files = sorted(glob.glob(f"{args.data}/NORMAL/*.jpeg")) + sorted(glob.glob(f"{args.data}/PNEUMONIA/*.jpeg"))
    cleans = [(resize(TF.to_tensor(Image.open(f).convert("L"))), label(f)) for f in files]
    n_pairs_total = len(PRESETS) * len(list(combinations(CLASSES, 2)))   # 3 presets x 3 class-pairs
    alpha_adj = args.alpha / n_pairs_total
    print(f"loaded {len(cleans)} images "
          f"({', '.join(f'{c}={sum(1 for _,l in cleans if l==c)}' for c in CLASSES)}); "
          f"class-comparison Bonferroni alpha = {args.alpha}/{n_pairs_total} = {alpha_adj:.3e}")

    data = {}
    for name in ["low", "mid", "high"]:
        a, b = PRESETS[name]["a"], PRESETS[name]["b"]
        d = {c: [] for c in CLASSES}
        for i, (clean, cls) in enumerate(cleans):
            g = torch.Generator(); g.manual_seed(args.seed + i)
            noisy = (clean + torch.randn(clean.shape, generator=g)
                     * (a * clean.clamp(min=0.0) + b).sqrt()).clamp(0.0, 1.0)
            d[cls].append(psnr(clean, noisy))
        data[name] = {c: np.array(v) for c, v in d.items()}

    print("\n== Claim 1: three well-separated severity levels ==")
    F, pF = f_oneway(*[np.concatenate([data[n][c] for c in CLASSES]) for n in ["low", "mid", "high"]])
    print(f"one-way ANOVA: F={F:.1f}, p={pF:.2e}")
    for x, y in [("low", "mid"), ("mid", "high"), ("low", "high")]:
        ax = np.concatenate([data[x][c] for c in CLASSES]); ay = np.concatenate([data[y][c] for c in CLASSES])
        t, pv = ttest_ind(ax, ay, equal_var=False)
        print(f"  {x} vs {y}: dPSNR={ax.mean()-ay.mean():+.2f} dB, Welch t={t:.1f}, p={pv:.2e}, Cohen d={cohend(ax, ay):+.2f}")

    print("\n== Claim 2: class-agnostic (Normal / Bacterial / Viral) per level ==")
    for name in ["low", "mid", "high"]:
        print(f"  [{name}] " + ", ".join(f"{c}={data[name][c].mean():.2f}+/-{data[name][c].std(ddof=1):.2f}" for c in CLASSES))
        for x, y in combinations(CLASSES, 2):
            t, pv = ttest_ind(data[name][x], data[name][y], equal_var=False)
            sig = "" if pv < alpha_adj else " (n.s.)"
            print(f"      {x} vs {y}: d={cohend(data[name][x], data[name][y]):+.3f}, Welch t={t:.2f}, p={pv:.3f}{sig}")


if __name__ == "__main__":
    main()
