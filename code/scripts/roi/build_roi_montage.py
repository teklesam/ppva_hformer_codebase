#!/usr/bin/env python
"""Fig 4.11 ROI montage: overview CXR + four ROI crop rows across all 26 columns
(Noisy, Reference, 24 arms), split into TWO stacked groups of 13 columns so the
crops are large enough to read the per-arm perception-distortion differences.
Assembled offline from dumped per-arm reconstructions (no GPU)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec

NPZ = "/tmp/roi_dump_normal9.npz"
OUT = "/Users/sam/Documents/PPVAE Dissertation Project/PpCNN/dissertation-latex/dissertation-latex/figures/roi_panels_v2/fig_roi_comprehensive_grid.pdf"

# ROI regions: (name, colour, x, y, w, h). Chosen from the inter-arm disagreement map.
ROIS = [
    ("I  fine pulmonary vessels", "#4c9eeb",  30,  92, 54, 54),
    ("II  posterior rib margin",  "#f2b134", 150, 118, 54, 54),
    ("III  costophrenic angle",   "#5fd35f", 196, 183, 52, 52),
    ("IV  perihilar vessels",     "#e26dd2",  90, 128, 50, 50),
]
GROUP = 13   # columns per stacked group

plt.rcParams.update({"font.family": "serif", "font.size": 7})

def main():
    d = np.load(NPZ, allow_pickle=True)
    clean, noisy = d["clean"], d["noisy"]
    recons, labels = d["recons"], [str(x) for x in d["labels"]]

    col_imgs = [noisy, clean] + [recons[i] for i in range(len(labels))]
    col_labs = ["Noisy", "Reference"] + labels
    ncol, nroi = len(col_imgs), len(ROIS)

    # split 26 columns into two contiguous groups of 13
    groups = [list(range(0, GROUP)), list(range(GROUP, ncol))]
    nblk = len(groups)

    # gridspec: row 0 = overview band; then nblk blocks of nroi crop rows each
    total_rows = 1 + nblk * nroi
    fig_w = 15.5
    fig_h = 3.2 + nblk * nroi * 0.92
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(total_rows, GROUP, figure=fig, hspace=0.06, wspace=0.03,
                  height_ratios=[3.0] + [1.0] * (nblk * nroi))

    # -- overview CXR + legend --
    ax0 = fig.add_subplot(gs[0, 0:5])
    ax0.imshow(clean, cmap="gray", vmin=0, vmax=1)
    for name, colr, x, y, w, h in ROIS:
        ax0.add_patch(patches.Rectangle((x, y), w, h, lw=2.0, edgecolor=colr, facecolor="none"))
        ax0.text(x + w/2, y - 4, name.split()[0], color=colr, fontsize=11, fontweight="bold",
                 ha="center", va="bottom")
    ax0.set_title("Reference (Normal, mid noise $\\eta{=}200$)", fontsize=10)
    ax0.axis("off")
    axL = fig.add_subplot(gs[0, 5:])
    axL.axis("off")
    for i, (name, colr, *_ ) in enumerate(ROIS):
        axL.text(0.02, 0.92 - i * 0.19, name, color=colr, fontsize=13, fontweight="bold",
                 transform=axL.transAxes, va="top")
    axL.text(0.02, 0.10, "Each block: four ROI rows across a group of columns "
             "(noisy input, clean reference, then arms/baselines).",
             fontsize=9, style="italic", color="#333333", transform=axL.transAxes, va="bottom")

    # -- two stacked groups, each with nroi ROI crop rows --
    for b, cols in enumerate(groups):
        for r, (name, colr, x, y, w, h) in enumerate(ROIS):
            grow = 1 + b * nroi + r
            for cpos, ci in enumerate(cols):
                img, lab = col_imgs[ci], col_labs[ci]
                ax = fig.add_subplot(gs[grow, cpos])
                ax.imshow(img[y:y+h, x:x+w], cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color(colr); s.set_linewidth(1.3)
                if r == 0:   # column header at top of each block
                    fw = "bold" if lab in ("Noisy", "Reference") else "normal"
                    ax.set_title(lab, fontsize=7.2, rotation=90, va="bottom", ha="center",
                                 pad=2, fontweight=fw)
                if cpos == 0:
                    ax.set_ylabel(name.split(None, 1)[1], color=colr, fontsize=8.5, fontweight="bold")
    fig.savefig(OUT, bbox_inches="tight", dpi=300)
    fig.savefig(OUT.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
