import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

d = np.load("/tmp/lambda_roi_recons.npz", allow_pickle=True)
clean, noisy, recons = d["clean"], d["noisy"], d["recons"]
ssim, ffl, psnrs = d["ssim"], d["ffl"], d["psnrs"]
rx, ry, rw, rh = 62, 98, 78, 78
WONG_BLUE = "#0072B2"

# psnr of noisy vs clean
def psnr(a,b):
    m = np.mean((a-b)**2); return 99.0 if m<1e-12 else 20*np.log10(1/np.sqrt(m))

# columns: Reference, Noisy, then 4 trials ascending SSIM weight
imgs   = [clean, noisy] + [recons[i] for i in range(len(recons))]
heads  = ["Reference", "Noisy"] + [f"$\\lambda_{{SSIM}}$={ssim[i]:.2f}\n$\\lambda_{{FFL}}$={ffl[i]:.3f}" for i in range(len(recons))]
subs   = ["", f"{psnr(clean,noisy):.2f} dB"] + [f"{psnrs[i]:.2f} dB" for i in range(len(recons))]
n = len(imgs)

fig, axes = plt.subplots(2, n, figsize=(2.05*n, 4.5),
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
        axz.set_xlabel(subs[j], fontsize=10)

axes[0,0].set_ylabel("full field", fontsize=9.5)
axes[1,0].set_ylabel("hilum ROI", fontsize=9.5)
for r in (0,1):
    axes[r,0].yaxis.set_visible(True); axes[r,0].set_yticks([])

fig.savefig("/tmp/lambda_roi_sensitivity.pdf", bbox_inches="tight", dpi=200)
fig.savefig("/tmp/lambda_roi_sensitivity.png", bbox_inches="tight", dpi=160)
print("saved lambda_roi_sensitivity.pdf/.png  columns:", heads)
