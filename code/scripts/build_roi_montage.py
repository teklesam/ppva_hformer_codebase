#!/usr/bin/env python
"""Fig 4.10 ROI montage (publication quality, v3): overview CXR with arrowed ROI
boxes, then four ROI crop rows across two self-contained blocks (Noisy, Reference,
12 arms each). Short arm-letter column labels; Roman-numeral row labels; clean
colour-blind-safe arrows (thin dark stroke, no white box). Offline, no GPU."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import patches
from matplotlib.gridspec import GridSpec

NPZ = "/tmp/roi_dump_normal9.npz"   # base dump (27 conditions); BM3D is appended inline below
OUT = "/Users/sam/Documents/PPVAE Dissertation Project/PpCNN/dissertation-latex/dissertation-latex/figures/roi_panels_v2/fig_roi_comprehensive_grid.pdf"

# ROI: (roman, desc, colour, x, y, w, h, arrow_tail_frac, arrow_tip_frac). Wong palette.
ROIS = [
    ("I",   "fine pulmonary vessels", "#0072B2",  56, 120, 50, 50, (0.80, 0.20), (0.52, 0.55)),
    ("II",  "posterior rib margin",   "#D55E00", 150, 118, 54, 54, (0.82, 0.20), (0.52, 0.48)),
    ("III", "costophrenic angle",     "#009E73", 204, 192, 52, 56, (0.24, 0.22), (0.60, 0.66)),
    ("IV",  "perihilar vessels",      "#CC79A7",  90, 128, 50, 50, (0.82, 0.80), (0.52, 0.52)),
]
plt.rcParams.update({"font.family": "serif", "font.size": 7})

def short(lab):
    """Arm letter only (e.g. 'J L1+SSIM+FFL' -> 'J'); baselines keep their name."""
    tok = lab.split()[0]
    return tok if (len(tok) == 1 and tok.isalpha()) else lab

def crop_arrow(ax, colr, tail, tip):
    a = ax.annotate("", xy=tip, xytext=tail, xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=colr, lw=1.3, mutation_scale=9,
                                    shrinkA=0, shrinkB=0), zorder=20)
    a.arrow_patch.set_path_effects([pe.withStroke(linewidth=2.1, foreground="black", alpha=0.7)])

def main():
    d = np.load(NPZ, allow_pickle=True)
    clean, noisy = d["clean"], d["noisy"]
    recons, labels = d["recons"], [str(x) for x in d["labels"]]
    # Append classical BM3D (given oracle noise knowledge, per the study): sigma from
    # this case's noisy-clean residual, denoise the same noisy input, add as a column.
    if "BM3D" not in labels:
        import bm3d
        sigma = float(np.std(noisy.astype(np.float32) - clean.astype(np.float32)))
        den = np.clip(bm3d.bm3d(noisy.astype(np.float32), sigma_psd=sigma), 0, 1).astype(recons.dtype)
        recons = np.concatenate([recons, den[None]], axis=0)
        labels = labels + ["BM3D"]
    col_imgs = [noisy, clean] + [recons[i] for i in range(len(labels))]
    col_labs = ["Noisy", "Reference"] + [short(l) for l in labels]
    ncol, nroi = len(col_imgs), len(ROIS)
    narm = len(labels); half = narm // 2
    groups = [[0, 1] + list(range(2, 2 + half)), [0, 1] + list(range(2 + half, ncol))]
    GROUP = 2 + half
    nblk = len(groups)

    # rows: overview, SPACER, blockA(nroi), SPACER, blockB(nroi)
    hr = [2.7, 0.55] + [1.0]*nroi + [0.55] + [1.0]*nroi
    fig = plt.figure(figsize=(16.5, 9.2))
    gs = GridSpec(len(hr), GROUP, figure=fig, hspace=0.10, wspace=0.04, height_ratios=hr)

    # -- overview with arrowed ROI boxes --
    ax0 = fig.add_subplot(gs[0, 0:5])
    ax0.imshow(clean, cmap="gray", vmin=0, vmax=1)
    for rom, desc, colr, x, y, w, h, *_ in ROIS:
        ax0.add_patch(patches.Rectangle((x, y), w, h, lw=2.1, edgecolor=colr, facecolor="none"))
        an = ax0.annotate("", xy=(x + w*0.5, y + h*0.5), xytext=(x + w*0.5, y - 17),
                          arrowprops=dict(arrowstyle="-|>", color=colr, lw=1.6, mutation_scale=12), zorder=10)
        an.arrow_patch.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white", alpha=0.8)])
        ax0.text(x + w/2, y - 21, rom, color=colr, fontsize=12, fontweight="bold", ha="center", va="bottom",
                 path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    ax0.set_title("Reference CXR (Normal, mid noise $\\eta{=}200$)", fontsize=10)
    ax0.axis("off")
    axL = fig.add_subplot(gs[0, 5:]); axL.axis("off")
    for i, (rom, desc, colr, *_ ) in enumerate(ROIS):
        axL.text(0.02, 0.90 - i*0.19, f"{rom}", color=colr, fontsize=13, fontweight="bold",
                 transform=axL.transAxes, va="top")
        axL.text(0.10, 0.90 - i*0.19, desc, color="#222222", fontsize=11.5,
                 transform=axL.transAxes, va="top")
    axL.text(0.02, 0.05, "Two blocks; each begins with Noisy + Reference, then half of the conditions. "
             "Arrows mark the feature; compare each condition against the Reference.",
             fontsize=8.5, style="italic", color="#444444", transform=axL.transAxes, va="bottom")

    block_row0 = [2, 2 + nroi + 1]            # after overview + spacer, then spacer between blocks
    hdr = {"Noisy": "Noisy", "Reference": "Ref"}
    for b, cols in enumerate(groups):
        for r, (rom, desc, colr, x, y, w, h, tail, tip) in enumerate(ROIS):
            grow = block_row0[b] + r
            for cpos, ci in enumerate(cols):
                img, lab = col_imgs[ci], col_labs[ci]
                ax = fig.add_subplot(gs[grow, cpos])
                ax.imshow(img[y:y+h, x:x+w], cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color(colr); s.set_linewidth(1.4)
                crop_arrow(ax, colr, tail, tip)
                if r == 0:                     # horizontal short column headers (no rotation)
                    fw = "bold" if lab in ("Noisy", "Reference") else "normal"
                    ax.set_title(hdr.get(lab, lab), fontsize=8, rotation=0, va="bottom",
                                 ha="center", pad=3, fontweight=fw)
                if cpos == 0:
                    ax.set_ylabel(rom, color=colr, fontsize=12, fontweight="bold", rotation=0,
                                  ha="right", va="center", labelpad=8)
    fig.savefig(OUT, bbox_inches="tight", dpi=300)
    fig.savefig(OUT.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
