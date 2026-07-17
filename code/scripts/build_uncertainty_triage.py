"""Fig 5.1 v4 (Arm H): ONE reconstruction reporting BOTH uncertainties together.
A single denoised X-ray carries both channels at once, as discrete numbered regions:
  amber = measurement-limited (aleatoric sigma_a) ; blue = model-inferred / verify (epistemic sigma_e).
Left: the clean denoised image; right: the same image with both overlays + a per-scan report.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Patch
from scipy import ndimage

d = np.load("aleatoric_epistemic_H.npz")
mu, sig_a, sig_e, noisy = d["mu"], d["sig_a"], d["sig_e"], d["noisy"]
H, W = mu.shape
C_AL, C_EP = "#E69F00", "#2E86DE"   # amber = aleatoric ; blue = epistemic

def zone(cx, cy):
    row = "upper" if cy < H/3 else ("mid" if cy < 2*H/3 else "lower")
    if 0.40*W <= cx <= 0.60*W: return "mediastinum"
    return f"{'right' if cx < 0.40*W else 'left'} {row}"

def regions(sigma, pct, nmax, min_area=30):
    s = ndimage.gaussian_filter(sigma, 2.0)
    lbl, n = ndimage.label(s >= np.percentile(s, pct)); smax = float(s.max()); out = []
    for i in range(1, n+1):
        m = lbl == i
        if m.sum() < min_area: continue
        ys, xs = np.where(m)
        out.append(dict(x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max(),
                        cx=xs.mean(), cy=ys.mean(), score=float(s[m].sum()),
                        sev="High" if s[m].max() >= 0.75*smax else "Moderate"))
    out.sort(key=lambda r: -r["score"]); return out[:nmax]

al = regions(sig_a, 98.5, 2); ep = regions(sig_e, 97.5, 3)

def pixel_overlay(ax, sigma, rgb, pct, amax=0.62):
    """Translucent colour wash over the high-uncertainty pixels of one channel."""
    s = ndimage.gaussian_filter(sigma, 1.5)
    thr = np.percentile(s, pct); hi = np.percentile(s, 99.7)
    sn = np.clip((s - thr) / (hi - thr + 1e-9), 0, 1)
    rgba = np.zeros((*s.shape, 4))
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = rgb
    rgba[..., 3] = np.where(s >= thr, amax * (0.4 + 0.6 * sn), 0.0)
    ax.imshow(rgba, zorder=2)

def draw_box(ax, r, color, lab):
    w, h, p = r["x1"]-r["x0"], r["y1"]-r["y0"], 5
    ax.add_patch(FancyBboxPatch((r["x0"]-p, r["y0"]-p), w+2*p, h+2*p,
        boxstyle="round,pad=2,rounding_size=5", lw=2.3, ec=color, fc="none", zorder=4))
    cx, cy = r["x0"]-p, r["y0"]-p
    ax.add_patch(Circle((cx, cy), 9, color=color, zorder=5))
    ax.text(cx, cy, lab, color="white", fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=6)

fig = plt.figure(figsize=(13.8, 4.6))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.62], wspace=0.06)

an = fig.add_subplot(gs[0]); an.imshow(noisy, cmap="gray", vmin=0, vmax=1)
an.set_title("Noisy input", fontsize=10); an.set_xticks([]); an.set_yticks([])

a0 = fig.add_subplot(gs[1]); a0.imshow(mu, cmap="gray", vmin=0, vmax=1)
a0.set_title("Denoised reconstruction (Arm H)", fontsize=10); a0.set_xticks([]); a0.set_yticks([])

a1 = fig.add_subplot(gs[2]); a1.imshow(mu, cmap="gray", vmin=0, vmax=1)
pixel_overlay(a1, sig_a, (0.902, 0.624, 0.0), 96)    # amber pixels = data-limited (aleatoric)
pixel_overlay(a1, sig_e, (0.180, 0.525, 0.871), 95)  # blue pixels = model-inferred (epistemic)
for k, r in enumerate(al, 1): draw_box(a1, r, C_AL, f"A{k}")
for k, r in enumerate(ep, 1): draw_box(a1, r, C_EP, f"E{k}")
a1.set_title("Combined reliability overlay", fontsize=10); a1.set_xticks([]); a1.set_yticks([])
a1.legend(handles=[
    Patch(fc=C_AL, ec="k", lw=0.4, label=r"Measurement-limited (aleatoric $\hat{\sigma}_a$)"),
    Patch(fc=C_EP, ec="k", lw=0.4, label=r"Model-inferred, verify (epistemic $\hat{\sigma}_e$)")],
    loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, fontsize=8, frameon=False)

a2 = fig.add_subplot(gs[3]); a2.set_xlim(0, 1); a2.set_ylim(0, 1); a2.axis("off"); T = a2.transAxes
def rpt(x, y, s, **k): a2.text(x, y, s, transform=T, va="top", **k)
rpt(0.0, 0.99, "Per-scan report", fontsize=9.5, fontweight="bold")
rpt(0.0, 0.90, f"{'ID':<4}{'Location':<15}{'Sev.'}", fontsize=8, family="monospace", fontweight="bold")
a2.plot([0, 1], [0.855, 0.855], color="0.7", lw=0.8, transform=T, clip_on=False)
y = 0.82
for k, r in enumerate(ep, 1):
    rpt(0.0, y, f"{'E'+str(k):<4}{zone(r['cx'],r['cy']):<15}{r['sev']}", fontsize=8, family="monospace", color="#1B5E9B"); y -= 0.09
for k, r in enumerate(al, 1):
    rpt(0.0, y, f"{'A'+str(k):<4}{zone(r['cx'],r['cy']):<15}{'—'}", fontsize=8, family="monospace", color="#B37400"); y -= 0.09
rpt(0.0, y-0.02, "Action: cross-check the model-\ninferred (E) regions against the raw\nlow-dose image before reporting.", fontsize=7.6, color="0.15")

OUT = ("/Users/sam/Documents/PPVAE Dissertation Project/PpCNN/dissertation-latex/"
       "dissertation-latex/figures/evaluation_v2/fig_uncertainty_hotspot")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print("saved v4 |", "A:", [zone(r['cx'],r['cy']) for r in al], "E:", [zone(r['cx'],r['cy']) for r in ep])
