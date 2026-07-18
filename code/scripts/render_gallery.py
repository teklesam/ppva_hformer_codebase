#!/usr/bin/env python3
"""render_gallery.py -- Appendix reconstruction gallery (Figure E, fig:app:gallery).

Assembles the denoised reconstruction (mu-hat) of the 16 core ablation arms (A-P)
and the 5 retrained KAIR baselines (DnCNN, IRCNN, FFDNet, DRUNet, SwinIR) for a
single shared Bacterial case at mid noise (eta=200) onto one page, with the noisy
input and clean reference for comparison. Panels are family-colour-coded and
badged with each model's mean test-set PSNR.

The reconstructions are cropped directly from the per-arm evaluation panels in
figures/evaluation_v2/qualitative/qualitative_<arm>_<class>.png so that no model
checkpoint is needed to regenerate the figure. To add the fine-tuning arms (Q, R, S)
or the extended SOTA baselines (NAFNet, SCUNet, SharpXR), first dump their
qualitative_*_bacterial.png panels from the checkpoints, then extend ITEMS below.

Usage:  python scripts/render_gallery.py [path/to/qualitative_dir]
Output: <qual_dir>/gallery_all_arms_bacterial.{png,pdf}
"""
import os, sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLS = "bacterial"

# --- per-arm mean test-set PSNR at mid noise (from results/per_image_metrics.csv) ---
PSNR = {
    "arm_a_l2": 34.62, "arm_i_l1": 34.44, "arm_q_charb": 34.54, "arm_b_nll": 33.46,
    "arm_c_nll_ssim": 32.36, "arm_d_nll_ssim_ffl": 33.44, "arm_k_nll_l1": 33.28,
    "arm_l_nll_edge_ffl": 33.14, "arm_m_full_det": 31.69, "arm_n_perc": 32.42,
    "arm_o_prelu": 29.07, "arm_j_l1_ssim_ffl": 34.32, "arm_r_ft_j": 34.55, "arm_s_ft_d": 34.61,
    "arm_e_ppvae": 31.64, "arm_f_kl_cyc": 32.73, "arm_g_kl_fb": 32.48, "arm_h_kl_cyc_fb": 32.76,
    "arm_p_best": 31.67, "dncnn_baseline": 32.80, "ircnn": 32.66, "ffdnet": 32.63,
    "drunet": 32.58, "swinir": 31.18, "nafnet": 33.80, "scunet": 34.15, "sharpxr": 33.56,
}
FAM = {"pix": "#2f6fb0", "nll": "#c05a1e", "comp": "#3f8f4f",
       "vae": "#7a4fb0", "base": "#555555", "sota": "#8a6d1f", "anchor": "#111111"}
# (filekey, label, family, source): "panel" crops mu from qualitative_<key>_bacterial.png;
# "direct" loads mu_<key>.png (rendered on CSD3 from the checkpoint on the SAME shared input).
ITEMS = [
    ("__noisy__", "Noisy input", "anchor", "panel"), ("__ref__", "Reference (clean)", "anchor", "panel"),
    ("arm_a_l2", "A · L2", "pix", "panel"), ("arm_i_l1", "I · L1", "pix", "panel"),
    ("arm_q_charb", "Q · Charbonnier", "pix", "direct"),
    ("arm_b_nll", "B · NLL", "nll", "panel"), ("arm_c_nll_ssim", "C · NLL+SSIM", "nll", "panel"),
    ("arm_d_nll_ssim_ffl", "D · NLL+SSIM+FFL", "nll", "panel"), ("arm_k_nll_l1", "K · NLL+L1", "nll", "panel"),
    ("arm_l_nll_edge_ffl", "L · NLL+Edge+FFL", "nll", "panel"), ("arm_m_full_det", "M · NLL+SSIM+Edge", "nll", "panel"),
    ("arm_n_perc", "N · NLL+VGG", "nll", "panel"), ("arm_o_prelu", "O · PReLU (failure)", "nll", "panel"),
    ("arm_j_l1_ssim_ffl", "J · L1+SSIM+FFL", "comp", "panel"),
    ("arm_r_ft_j", "R · FT L1+SSIM+FFL", "comp", "direct"), ("arm_s_ft_d", "S · FT NLL+SSIM+FFL", "comp", "direct"),
    ("arm_e_ppvae", "E · VAE linear-KL", "vae", "panel"), ("arm_f_kl_cyc", "F · VAE cyclic-KL", "vae", "panel"),
    ("arm_g_kl_fb", "G · VAE free-bits", "vae", "panel"), ("arm_h_kl_cyc_fb", "H · VAE cyc+fb", "vae", "panel"),
    ("arm_p_best", "P · VAE best", "vae", "panel"),
    ("dncnn_baseline", "DnCNN", "base", "panel"), ("ircnn", "IRCNN", "base", "panel"),
    ("ffdnet", "FFDNet", "base", "panel"), ("drunet", "DRUNet", "base", "panel"),
    ("swinir", "SwinIR", "base", "panel"),
    ("nafnet", "NAFNet", "sota", "direct"), ("scunet", "SCUNet", "sota", "direct"),
    ("sharpxr", "SharpXR", "sota", "direct"),
]


def _largest_run(mask1d):
    best = (0, 0); s = None
    for i, v in enumerate(mask1d):
        if v and s is None:
            s = i
        if (not v) and s is not None:
            if i - s > best[1] - best[0]:
                best = (s, i)
            s = None
    if s is not None and len(mask1d) - s > best[1] - best[0]:
        best = (s, len(mask1d))
    return best


def _gray_panels_top_row(fn):
    """Segment the top row of a qualitative panel into image sub-panels using
    whitespace gutters, and flag each as grayscale (low saturation) or heatmap."""
    rgb = np.asarray(Image.open(fn).convert("RGB"), dtype=np.float32)
    H, W, _ = rgb.shape
    top = rgb[0:H // 3]
    mx, mn = top.max(2), top.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    nonwhite = mx < 245
    col_content = nonwhite.mean(0)
    col_color = ((sat > 0.15) & nonwhite).mean(0)
    is_gutter = col_content < 0.02
    spans = []; s = None
    for x in range(W):
        if not is_gutter[x] and s is None:
            s = x
        if is_gutter[x] and s is not None:
            if x - s > 30:
                spans.append((s, x))
            s = None
    if s is not None and W - s > 30:
        spans.append((s, W))
    panels = [(x0, x1, col_color[x0:x1].mean() < 0.03) for (x0, x1) in spans]
    return top, panels


def get_gray(fn, which):
    """Return the main image block (uint8) for the Noisy/Reference/mu column of
    the top row, dropping the source panel's column title and burnt-in labels."""
    top, panels = _gray_panels_top_row(fn)
    grays = [(x0, x1) for x0, x1, g in panels if g]
    x0, x1 = grays[{"noisy": 0, "ref": 1, "mu": 2}[which]]
    cell = np.asarray(Image.fromarray(top[:, x0:x1].astype("uint8")).convert("L"))
    r0, r1 = _largest_run((cell < 245).mean(1) > 0.35)
    cell = cell[r0:r1]
    c0, c1 = _largest_run((cell < 245).mean(0) > 0.30)
    return cell[:, c0:c1]


def find_file(qdir, key, cls):
    for f in os.listdir(qdir):
        if f.startswith("qualitative_" + key) and f.endswith(cls + ".png"):
            return os.path.join(qdir, f)
    return None


def main(qdir):
    anchor = find_file(qdir, "arm_a_l2", CLS)
    imgs = []
    for key, label, fam, src in ITEMS:
        if key == "__noisy__":
            imgs.append((get_gray(anchor, "noisy"), label, fam, None))
        elif key == "__ref__":
            imgs.append((get_gray(anchor, "ref"), label, fam, None))
        elif src == "direct":
            mu = np.asarray(Image.open(os.path.join(qdir, f"mu_{key}.png")).convert("L"))
            imgs.append((mu, label, fam, PSNR.get(key)))
        else:
            imgs.append((get_gray(find_file(qdir, key, CLS), "mu"), label, fam, PSNR.get(key)))

    ncol = 5
    nrow = int(np.ceil(len(imgs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.05, nrow * 2.35))
    plt.subplots_adjust(left=0.008, right=0.992, top=0.965, bottom=0.008, wspace=0.06, hspace=0.22)
    for i, ax in enumerate(axes.ravel()):
        ax.set_xticks([]); ax.set_yticks([])
        if i >= len(imgs):
            ax.axis("off"); continue
        img, label, fam, psnr = imgs[i]
        col = FAM[fam]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="equal")
        ax.set_title(label, fontsize=9.3, color=col, fontweight="bold", pad=3)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(col); sp.set_linewidth(1.4)
        ax.set_xticks([]); ax.set_yticks([])
        if psnr is not None:
            ax.text(0.97, 0.04, f"{psnr:.2f} dB", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8.0, color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", fc="black", ec="none", alpha=0.62))
    out = os.path.join(qdir, "gallery_all_arms_bacterial")
    fig.savefig(out + ".png", dpi=150)
    fig.savefig(out + ".pdf")
    print("wrote", out + ".png / .pdf")


if __name__ == "__main__":
    qdir = sys.argv[1] if len(sys.argv) > 1 else "figures/evaluation_v2/qualitative"
    main(qdir)
