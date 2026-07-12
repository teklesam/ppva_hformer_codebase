"""
foi_validate.py — Validate the Foi Poisson-Gaussian degradation pipeline.

Confirms that add_foi_noise() injects noise whose realised variance follows the
specified law Var(z | y) = a*y + b. For each preset the script draws many noise
realisations at a grid of clean intensities y in [0, 1], measures the realised
variance, and fits it back to recover (a, b); a perfect implementation returns
the specified coefficients with R^2 = 1.

Reproduces the degradation-pipeline validation figure in the dissertation
(Results, "Dataset Characteristics and Degradation Pipeline Validation").

Usage:
    python scripts/foi_validate.py --outdir figures
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# make the package importable regardless of the working directory
# (scripts/ is a sibling of src/ under the code root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.noise_simulation import add_foi_noise, PRESETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".", help="directory for the output figure")
    ap.add_argument("--levels", type=int, default=51, help="number of intensity levels in [0,1]")
    ap.add_argument("--samples", type=int, default=40000, help="noise realisations per level")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    levels = np.linspace(0.0, 1.0, args.levels)
    colors = {"low": "#1a9850", "mid": "#f39c12", "high": "#d73027"}

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    print(f"{'preset':6} {'a_spec':>7} {'a_rec':>7} {'b_spec':>8} {'b_rec':>9} {'R2':>8}")
    for name in ["low", "mid", "high"]:
        a, b = PRESETS[name]["a"], PRESETS[name]["b"]
        var_meas = []
        for xv in levels:
            x = torch.full((args.samples,), float(xv))
            z = add_foi_noise(x, a, b, clip=False)          # validate the pure variance law
            var_meas.append(float(((z - x).numpy()).var()))
        var_meas = np.array(var_meas)
        # least-squares fit  var = slope*y + intercept  -> recovered (a, b)
        A = np.vstack([levels, np.ones_like(levels)]).T
        (slope, intercept), *_ = np.linalg.lstsq(A, var_meas, rcond=None)
        pred = slope * levels + intercept
        r2 = 1 - ((var_meas - pred) ** 2).sum() / ((var_meas - var_meas.mean()) ** 2).sum()
        print(f"{name:6} {a:7.3f} {slope:7.4f} {b:8.4f} {intercept:9.5f} {r2:8.5f}")
        ax.scatter(levels, var_meas, s=10, color=colors[name], alpha=0.6,
                   label=f"{name.capitalize()} (a={a}, b={b})")
        ax.plot(levels, a * levels + b, color=colors[name], lw=1.4)

    ax.set_xlabel("clean intensity $y$")
    ax.set_ylabel(r"realised noise variance  $\mathrm{Var}(z-y)$")
    ax.legend(frameon=False, fontsize=8, title=r"specified law  Var $= a\,y+b$")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.outdir, f"foi_variance_validation.{ext}"),
                    dpi=150, bbox_inches="tight")
    print(f"saved foi_variance_validation.{{png,pdf}} to {args.outdir}")


if __name__ == "__main__":
    main()
