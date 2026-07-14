"""
uq_calibration_stats.py — Quantify what the aleatoric map tracks (C4 calibration).

For every image in the held-out test set and each NLL/VAE arm, computes the
Spearman rank correlation of the predicted aleatoric std sigma_a = exp(0.5 * log_var)
against:
    (a) the local image gradient magnitude |grad y|  (structural contrast / edges)
    (b) the raw clean intensity y                     (which sets the injected Foi variance a*y+b)

This is the raw-array version of the figure-recovered estimate reported in the
dissertation discussion (sigma_a is gradient/contrast-driven, ~flat vs intensity).
Reports per-image mean +/- SD (the rigorous statistic), the pooled-pixel value,
and top-vs-bottom quartile mean-sigma ratios. Writes a CSV and prints a
LaTeX-ready summary sentence.

Run on a GPU node (raw checkpoints live under RESULTS):
    srun --account=... --partition=ampere --gres=gpu:1 --time=0:30:00 --pty bash
    conda activate ppvae
    cd ~/hpc-work/ppvae_hformer
    python scripts/uq_calibration_stats.py --arms arm_b arm_d arm_h --eta 200
"""
from __future__ import annotations
import os, sys, glob, argparse, csv
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from scipy.ndimage import sobel

# Reuse the *exact* model-loading / inference / data interfaces from the figure
# generator so this analysis is guaranteed consistent with the published maps.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # ensure scripts/ is importable
from generate_roi_panels_v2 import (
    load_all_models, infer, add_noise, DATA, DEVICE,
)

NLL_ARMS = ("arm_b", "arm_d", "arm_h")  # only NLL-trained arms have a meaningful sigma_a


def list_test_images():
    """All test images across class subdirs, sorted, as (path) list."""
    paths = []
    for cls in sorted(os.listdir(DATA)):
        d = os.path.join(DATA, cls)
        if not os.path.isdir(d):
            continue
        paths += sorted(glob.glob(os.path.join(d, "*.jpeg")) +
                        glob.glob(os.path.join(d, "*.png")))
    return paths


def load_clean(path):
    from PIL import Image
    return (np.array(Image.open(path).convert("L").resize((256, 256))) / 255.0).astype(np.float32)


def grad_mag(y):
    gx = sobel(y, axis=0); gy = sobel(y, axis=1)
    return np.hypot(gx, gy)


def quartile_ratio(sig, var):
    """mean sigma in top vs bottom quartile of `var`."""
    q1, q3 = np.quantile(var, [0.25, 0.75])
    lo = sig[var <= q1].mean()
    hi = sig[var >= q3].mean()
    return hi / max(lo, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(NLL_ARMS))
    ap.add_argument("--eta", type=int, default=200, help="Foi noise level (200 = mid)")
    ap.add_argument("--body-thresh", type=float, default=0.05,
                    help="clean intensity above which a pixel counts as body (excludes background)")
    ap.add_argument("--out", default="uq_calibration_stats.csv")
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    models = load_all_models()
    paths = list_test_images()
    print(f"Test images: {len(paths)}")

    rows = []
    for arm in args.arms:
        model = models.get(arm)
        if model is None:
            print(f"[SKIP] {arm}: model not loaded")
            continue

        # per-image correlations (body-masked) and pooled pixels
        sp_grad, sp_int = [], []
        pr_grad, pr_int = [], []
        qr_grad, qr_int = [], []
        pooled_s, pooled_g, pooled_y = [], [], []

        for i, p in enumerate(paths):
            clean = load_clean(p)
            noisy = add_noise(clean, args.eta, seed=42 + i)
            noisy_t = torch.from_numpy(noisy).float()[None]        # (1,256,256)
            _, lsa = infer(model, noisy_t, arm)
            if lsa is None:
                continue
            sigma = np.exp(0.5 * lsa)                              # aleatoric std, raw
            g = grad_mag(clean)

            body = clean > args.body_thresh
            s, gg, yy = sigma[body], g[body], clean[body]
            if s.size < 100 or s.std() < 1e-8:
                continue
            sp_grad.append(spearmanr(s, gg)[0])
            sp_int.append(spearmanr(s, yy)[0])
            pr_grad.append(pearsonr(s, gg)[0])
            pr_int.append(pearsonr(s, yy)[0])
            qr_grad.append(quartile_ratio(s, gg))
            qr_int.append(quartile_ratio(s, yy))
            # subsample for pooled correlation (keep memory bounded)
            idx = np.random.default_rng(i).choice(s.size, size=min(2000, s.size), replace=False)
            pooled_s.append(s[idx]); pooled_g.append(gg[idx]); pooled_y.append(yy[idx])

        if not sp_grad:
            print(f"[WARN] {arm}: no valid images")
            continue

        S = np.concatenate(pooled_s); G = np.concatenate(pooled_g); Y = np.concatenate(pooled_y)
        rec = dict(
            arm=arm, n_images=len(sp_grad),
            sp_grad_mean=np.nanmean(sp_grad), sp_grad_sd=np.nanstd(sp_grad),
            sp_int_mean=np.nanmean(sp_int),  sp_int_sd=np.nanstd(sp_int),
            pr_grad_mean=np.nanmean(pr_grad), pr_int_mean=np.nanmean(pr_int),
            qr_grad_mean=np.nanmean(qr_grad), qr_int_mean=np.nanmean(qr_int),
            pooled_sp_grad=spearmanr(S, G)[0],
            pooled_sp_int=spearmanr(S, Y)[0],
        )
        rows.append(rec)
        print(f"\n=== {arm}  (n={rec['n_images']} images, eta={args.eta}, body-masked) ===")
        print(f"  sigma_a vs |grad y| : Spearman {rec['sp_grad_mean']:+.3f} +/- {rec['sp_grad_sd']:.3f}"
              f"  (pooled {rec['pooled_sp_grad']:+.3f}); top/bottom-quartile sigma ratio {rec['qr_grad_mean']:.2f}x")
        print(f"  sigma_a vs intensity: Spearman {rec['sp_int_mean']:+.3f} +/- {rec['sp_int_sd']:.3f}"
              f"  (pooled {rec['pooled_sp_int']:+.3f}); top/bottom-quartile sigma ratio {rec['qr_int_mean']:.2f}x")

    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {args.out}")
        # LaTeX-ready sentence built from Arm D if present, else first row
        r = next((x for x in rows if x["arm"] == "arm_d"), rows[0])
        print("\n--- LaTeX-ready sentence (edit arm label as needed) ---")
        print(
            f"Across the {r['n_images']} test images, $\\hat{{\\sigma}}_a$ correlates with the local "
            f"image gradient (Spearman $\\rho={r['sp_grad_mean']:.2f}\\pm{r['sp_grad_sd']:.2f}$) but is "
            f"essentially independent of raw intensity within the body "
            f"($\\rho={r['sp_int_mean']:.2f}\\pm{r['sp_int_sd']:.2f}$), confirming that the aleatoric "
            f"map is driven by structural contrast (edges) rather than the intensity that sets the "
            f"injected noise variance."
        )


if __name__ == "__main__":
    main()
