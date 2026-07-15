import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

d = np.load("/tmp/lambda_roi_recons.npz", allow_pickle=True)
clean, noisy, recons = d["clean"], d["noisy"], d["recons"]
ssim, ffl, psnrs, fsims, lpipss = d["ssim"], d["ffl"], d["psnrs"], d["fsims"], d["lpipss"]
rx, ry, rw, rh = 62, 98, 78, 78
WONG_BLUE = "#0072B2"

def mstr(p, f, l):  # PSNR / FSIM / LPIPS stacked under each crop
    return f"PSNR {p:.2f}\nFSIM {f:.3f}\nLPIPS {l:.3f}"

# columns: Reference, Noisy, then 4 trials ascending SSIM weight
imgs   = [clean, noisy] + [recons[i] for i in range(len(recons))]
heads  = ["Reference", "Noisy"] + [f"$\\lambda_{{SSIM}}$={ssim[i]:.2f}\n$\\lambda_{{FFL}}$={ffl[i]:.3f}" for i in range(len(recons))]
subs   = ["", mstr(float(d["noisy_psnr"]), float(d["noisy_fsim"]), float(d["noisy_lpips"]))] + \
         [mstr(psnrs[i], fsims[i], lpipss[i]) for i in range(len(recons))]
n = len(imgs)

fig, axes = plt.subplots(2, n, figsize=(2.05*n, 5.3),
                         gridspec_kw=dict(hspace=0.06, wspace=0.05))
for j in range(n):
    # top: full image with ROI box
    ax = axes[0, j]; ax.imshow(imgs[j], cmap="gray", vmin=0, vmax=1)
    rect = mpatches.Rectangle((rx, ry), rw, rh, fill=False, edgecolor=WONG_BLUE, linewidth=1.6)
    ax.add_patch(rect); ax.set_xticks([]); ax.set_yticks([])
    t = ax.set_title(heads[j], fontsize=10.5, pad=4)
    for s in ax.spines.values(): s.set_visible(False)
    # bottom: zoomed crop
    axz = axes[1, j]
    crop = imgs[j][ry:ry+rh, rx:rx+rw]
    axz.imshow(crop, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axz.set_xticks([]); axz.set_yticks([])
    for s in axz.spines.values(): s.set_edgecolor(WONG_BLUE); s.set_linewidth(1.6)
    if subs[j]:
        axz.set_xlabel(subs[j], fontsize=8.0, linespacing=1.35)

axes[0,0].set_ylabel("full field", fontsize=9.5)
axes[1,0].set_ylabel("hilum ROI", fontsize=9.5)
for r in (0,1):
    axes[r,0].yaxis.set_visible(True); axes[r,0].set_yticks([])

fig.savefig("/tmp/lambda_roi_sensitivity.pdf", bbox_inches="tight", dpi=200)
fig.savefig("/tmp/lambda_roi_sensitivity.png", bbox_inches="tight", dpi=160)
print("saved lambda_roi_sensitivity.pdf/.png  columns:", heads)
