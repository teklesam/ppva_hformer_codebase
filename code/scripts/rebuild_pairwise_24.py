"""
rebuild_pairwise_24.py -- pairwise effect-size test across all 24 conditions
(16 arms A-P + KAIR x5 + NAFNet + SCUNet + SharpXR), sourcing each metric from
the file that holds it on a CONSISTENT scale:

  PSNR, SSIM  <- per_image_metrics.csv        (arms + KAIR)
  FSIM, LPIPS <- per_image_fsim.csv           (arms + KAIR; piq/VGG-16 scale, as Table 4.2)
  all four    <- sota_unified_perimage.csv    (NAFNet, SCUNet; piq)
                 sharpxr_perimage.csv         (SharpXR;        piq)

This avoids the two-LPIPS-scale trap: per_image_metrics.csv carries the old
AlexNet-scale LPIPS (~0.13), whereas Table 4.2 and the retrained baselines use
piq/VGG-16 (~0.22). LPIPS here is piq throughout.

Outputs (results/):
  pairwise_stats.csv           PSNR:  arm1,arm2,t,p_raw,p_bonf,cohens_d,sig
  pairwise_stats_metrics.csv   SSIM/FSIM/LPIPS: metric,arm1,arm2,cohens_d,sig  (LPIPS sign-flipped)

Welch's t-test (unpaired), Cohen's d (pooled SD). Bonferroni divisor = C(24,2)=276.
"""
from __future__ import annotations
import csv, itertools, os
import numpy as np
from scipy.stats import ttest_ind

RES = os.path.join(os.path.dirname(__file__), "..", "..", "results")
ORDER = ["arm_a_l2","arm_i_l1","arm_j_l1_ssim_ffl","arm_k_nll_l1","arm_b_nll","arm_c_nll_ssim",
    "arm_d_nll_ssim_ffl","arm_l_nll_edge_ffl","arm_n_perc","arm_m_full_det","arm_e_ppvae",
    "arm_f_kl_cyc","arm_g_kl_fb","arm_h_kl_cyc_fb","arm_p_best","arm_o_prelu",
    "dncnn_baseline","ffdnet","ircnn","drunet","swinir","nafnet","scunet","sharpxr"]
NEWBASE = {"nafnet":"sota_unified_perimage.csv","scunet":"sota_unified_perimage.csv",
           "sharpxr":"sharpxr_perimage.csv"}


def cohend(a, b):
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1-1)*a.var(ddof=1) + (n2-1)*b.var(ddof=1)) / (n1+n2-2))
    return (a.mean() - b.mean()) / sp


def stars(pb):
    return "***" if pb < 1e-3 else "**" if pb < 1e-2 else "*" if pb < 5e-2 else "n.s."


def mid_vecs(path, metric):
    """arm -> np.array of mid-noise metric values (arms/KAIR files have noise_level)."""
    out = {}
    for r in csv.DictReader(open(os.path.join(RES, path))):
        if r.get("noise_level", "mid") != "mid":
            continue
        v = r.get(metric, "")
        if v in ("", "nan"):
            continue
        out.setdefault(r["arm"], []).append(float(v))
    return {k: np.array(v) for k, v in out.items()}


def base_vecs(model, path, metric):
    key = model  # model column value ('nafnet'/'scunet'/'SharpXR')
    col = model if model != "sharpxr" else "SharpXR"
    vals = [float(r[metric]) for r in csv.DictReader(open(os.path.join(RES, path)))
            if r["model"] == col and r[metric] not in ("", "nan")]
    return np.array(vals)


def build_metric(metric, arm_src):
    """Assemble {condition: vector} for one metric from the correct sources."""
    data = mid_vecs(arm_src, metric)                      # arms + KAIR
    for model, path in NEWBASE.items():
        data[model] = base_vecs(model, path, metric)      # new baselines (piq)
    return {a: data[a] for a in ORDER if a in data}


def main():
    SRC = {"psnr": "per_image_metrics.csv", "ssim": "per_image_metrics.csv",
           "fsim": "per_image_fsim.csv",    "lpips": "per_image_fsim.csv"}
    present = None
    n_pairs = None
    psnr_out = [("arm1","arm2","t","p_raw","p_bonf","cohens_d","sig")]
    met_out  = [("metric","arm1","arm2","cohens_d","sig")]
    for metric in ("psnr","ssim","fsim","lpips"):
        data = build_metric(metric, SRC[metric])
        conds = [a for a in ORDER if a in data]
        if present is None:
            present = conds; n_pairs = len(conds)*(len(conds)-1)//2
            print(f"{len(conds)} conditions, {n_pairs} pairs, alpha = {0.05/n_pairs:.3e}")
        for a1, a2 in itertools.combinations(conds, 2):
            t, p = ttest_ind(data[a1], data[a2], equal_var=False)
            pb = min(1.0, p * n_pairs)
            d = cohend(data[a1], data[a2])
            if metric == "lpips":
                d = -d                                     # lower better -> + = row better
            s = stars(pb)
            if metric == "psnr":
                psnr_out.append((a1, a2, f"{t:.3f}", f"{p:.4e}", f"{pb:.4e}", f"{d:.3f}", s))
            else:
                met_out.append((metric, a1, a2, f"{d:.3f}", s))

    with open(os.path.join(RES, "pairwise_stats.csv"), "w", newline="") as f:
        csv.writer(f).writerows(psnr_out)
    with open(os.path.join(RES, "pairwise_stats_metrics.csv"), "w", newline="") as f:
        csv.writer(f).writerows(met_out)
    ns = sum(1 for r in psnr_out[1:] if r[6] == "n.s.")
    print(f"wrote pairwise_stats.csv ({len(psnr_out)-1}) + pairwise_stats_metrics.csv ({len(met_out)-1})")
    print(f"PSNR: {n_pairs-ns} significant, {ns} n.s.")


if __name__ == "__main__":
    main()
