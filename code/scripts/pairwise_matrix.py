"""
pairwise_matrix.py — Systematic pairwise effect-size matrix for all conditions.

Reads per-image metric values (per_image_metrics.csv: columns arm, noise_level,
img_idx, class, psnr, ...) and computes, for a given metric and noise level, the
complete set of pairwise comparisons between every condition:
  Welch's t-test, Cohen's d, and Bonferroni-corrected p (divisor = number of
  pairs actually formed, so the correction always matches the data).

Outputs:
  - <outdir>/pairwise_cohend_matrix.{png,pdf}  a Cohen's d heatmap of ALL pairs,
    grouped by loss family, with non-significant pairs marked;
  - <outdir>/pairwise_stats_full.csv           the full numeric table.

Run wherever per_image_metrics.csv holds every condition (e.g. CSD3, which has
DnCNN's per-image PSNR): there it yields the full 21-condition / 210-pair matrix;
on a partial copy it yields whatever conditions are present.

Usage:
    python scripts/pairwise_matrix.py --csv results/per_image_metrics.csv \
        --noise mid --metric psnr --outdir results/figures
"""
from __future__ import annotations
import os, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# preferred display order + short labels (unknown arms are appended alphabetically)
ORDER = ["arm_a_l2","arm_i_l1","arm_j_l1_ssim_ffl","arm_k_nll_l1",
         "arm_b_nll","arm_c_nll_ssim","arm_d_nll_ssim_ffl","arm_l_nll_edge_ffl","arm_n_perc","arm_m_full_det",
         "arm_e_ppvae","arm_f_kl_cyc","arm_g_kl_fb","arm_h_kl_cyc_fb","arm_p_best","arm_o_prelu",
         "dncnn","ffdnet","ircnn","drunet","swinir","nafnet","scunet"]
LAB = {"arm_a_l2":"A·L2","arm_i_l1":"I·L1","arm_j_l1_ssim_ffl":"J","arm_k_nll_l1":"K",
       "arm_b_nll":"B·NLL","arm_c_nll_ssim":"C","arm_d_nll_ssim_ffl":"D","arm_l_nll_edge_ffl":"L",
       "arm_n_perc":"N","arm_m_full_det":"M","arm_e_ppvae":"E","arm_f_kl_cyc":"F","arm_g_kl_fb":"G",
       "arm_h_kl_cyc_fb":"H","arm_p_best":"P","arm_o_prelu":"O","dncnn":"DnCNN","ffdnet":"FFDNet",
       "ircnn":"IRCNN","drunet":"DRUNet","swinir":"SwinIR","nafnet":"NAFNet","scunet":"SCUNet"}
FAMILY_BREAKS_AFTER = {"arm_k_nll_l1","arm_m_full_det","arm_o_prelu"}  # pixel | NLL | VAE | baselines


def cohend(a, b):
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1-1)*a.var(ddof=1) + (n2-1)*b.var(ddof=1)) / (n1+n2-2))
    return (a.mean() - b.mean()) / sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--noise", default="mid")
    ap.add_argument("--metric", default="psnr")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # collect per-arm metric vectors (ordered by img_idx)
    rows = list(csv.DictReader(open(args.csv)))
    vecs = {}
    for r in rows:
        if r["noise_level"] != args.noise:
            continue
        v = r.get(args.metric, "")
        if v in ("", "nan"):
            continue
        vecs.setdefault(r["arm"], []).append((int(r["img_idx"]), float(v)))
    arms_present = [a for a in ORDER if a in vecs] + sorted(a for a in vecs if a not in ORDER)
    data = {a: np.array([v for _, v in sorted(vecs[a])]) for a in arms_present}
    N = len(arms_present)
    n_pairs = N * (N - 1) // 2
    alpha_adj = args.alpha / n_pairs
    print(f"{N} conditions, {n_pairs} pairs; Bonferroni alpha = {args.alpha}/{n_pairs} = {alpha_adj:.3e}")

    D = np.full((N, N), np.nan); SIG = np.zeros((N, N), bool)
    out = [("arm1","arm2","t","p_raw","p_bonf","cohens_d","sig")]
    for i in range(N):
        for j in range(i+1, N):
            a, b = data[arms_present[i]], data[arms_present[j]]
            t, p = ttest_ind(a, b, equal_var=False)
            pb = min(1.0, p * n_pairs); d = cohend(a, b)
            D[i, j] = d; D[j, i] = -d; SIG[i, j] = SIG[j, i] = (pb < args.alpha)
            out.append((arms_present[i], arms_present[j], f"{t:.3f}", f"{p:.4e}", f"{pb:.4e}",
                        f"{d:.3f}", "" if pb < args.alpha else "n.s."))
    with open(os.path.join(args.outdir, "pairwise_stats_full.csv"), "w", newline="") as f:
        csv.writer(f).writerows(out)
    nns = int(((~SIG) & ~np.isnan(D)).sum() // 2)
    print(f"significant: {n_pairs-nns}, n.s.: {nns}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(0.46*N+1.2, 0.42*N+1.0))
    cmap = plt.cm.RdBu_r.copy(); cmap.set_bad("#eeeeee")
    im = ax.imshow(np.clip(D, -3, 3), cmap=cmap, vmin=-3, vmax=3)
    for i in range(N):
        for j in range(N):
            if i != j and not np.isnan(D[i, j]) and not SIG[i, j]:
                ax.plot(j, i, 'o', ms=3.2, mfc='k', mec='none')
    labels = [LAB.get(a, a) for a in arms_present]
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels(labels, rotation=90, fontsize=7); ax.set_yticklabels(labels, fontsize=7)
    for k, a in enumerate(arms_present[:-1]):
        if a in FAMILY_BREAKS_AFTER:
            ax.axhline(k+0.5, color='k', lw=0.7); ax.axvline(k+0.5, color='k', lw=0.7)
    ax.set_title(f"Pairwise Cohen's $d$ ({args.metric.upper()}, {args.noise} noise): all {n_pairs} comparisons",
                 fontsize=10.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, extend='both')
    cb.set_label("Cohen's $d$  (row $-$ column; + = row arm higher)", fontsize=8.5)
    ax.plot([], [], 'o', ms=4, mfc='k', mec='none', label="not significant (Bonferroni)")
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.14), frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.outdir, f"pairwise_cohend_matrix.{ext}"), dpi=150, bbox_inches="tight")
    print(f"saved pairwise_cohend_matrix.{{png,pdf}} to {args.outdir}")


if __name__ == "__main__":
    main()
