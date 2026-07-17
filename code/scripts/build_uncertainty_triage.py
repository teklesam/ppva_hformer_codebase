"""Redesign v3 (Arm H): inferno pixel overlay + non-overlapping annotations.
- Uncertainty overlaid as a translucent INFERNO heatmap: alpha grows with sigma, so
  high-uncertainty pixels glow and the anatomy shows through elsewhere.
- Two channels in separate panels (so labels never collide):
    aleatoric sig_a -> MEASUREMENT-LIMITED (tracks high-contrast bone/edges)
    epistemic sig_e -> MODEL-INFERRED ('verify against source', the actionable signal)
- Numbered markers sit in clear space with short leader lines; details in the report.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from scipy import ndimage

d = np.load("aleatoric_epistemic_H.npz")
mu, sig_a, sig_e = d["mu"], d["sig_a"], d["sig_e"]
H, W = mu.shape
C_EP, C_AL = "#4FC3F7", "#FFD54F"   # marker ring colours (light, readable on inferno)

def zone(cx, cy):
    row = "upper" if cy < H/3 else ("mid" if cy < 2*H/3 else "lower")
    if 0.40*W <= cx <= 0.60*W: return "mediastinum"
    return f"{'right' if cx < 0.40*W else 'left'} {row} zone"

def inferno_overlay(ax, base, sigma, plo=55, phi=99.5):
    ax.imshow(base, cmap="gray", vmin=0, vmax=1)
    s = ndimage.gaussian_filter(sigma, 1.5)
    lo, hi = np.percentile(s, plo), np.percentile(s, phi)
    sn = np.clip((s - lo) / (hi - lo + 1e-9), 0, 1)
    rgba = cm.inferno(sn)
    rgba[..., 3] = (sn ** 0.9) * 0.88          # low uncertainty -> transparent
    ax.imshow(rgba)
    ax.set_xticks([]); ax.set_yticks([])
    return s

def regions(s, pct, nmax, min_area=30):
    lbl, n = ndimage.label(s >= np.percentile(s, pct))
    smax = float(s.max()); out = []
    for i in range(1, n + 1):
        m = lbl == i
        if m.sum() < min_area: continue
        ys, xs = np.where(m)
        out.append(dict(cx=xs.mean(), cy=ys.mean(), score=float(s[m].sum()),
                        sev="High" if s[m].max() >= 0.75*smax else "Moderate"))
    out.sort(key=lambda r: -r["score"])
    return out[:nmax]

def place_labels(ax, regs, ring, prefix):
    """Small numbered marker on each region; nudge apart if two would collide.
    Stays inside the panel, tied to the highlighted pixels."""
    placed = []
    for k, r in enumerate(regs, 1):
        cx, cy = r["cx"], r["cy"]
        # offset the marker just off the bright peak toward the nearest side edge
        mx = min(cx + 22, W - 14) if cx <= W/2 else max(cx - 22, 14)
        my = cy
        for (px, py) in placed:                       # collision nudge
            if abs(mx - px) < 26 and abs(my - py) < 26:
                my = py + 30
        placed.append((mx, my))
        ax.plot([cx, mx], [cy, my], color=ring, lw=1.1, zorder=5)
        ax.add_patch(Circle((mx, my), 8.5, fc=ring, ec="black", lw=1.0, zorder=6))
        ax.text(mx, my, f"{prefix}{k}", color="black", fontsize=7.6, fontweight="bold",
                ha="center", va="center", zorder=7)

s_a = None
fig = plt.figure(figsize=(13.6, 4.5))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.05], wspace=0.28)

a0 = fig.add_subplot(gs[0]); a0.imshow(mu, cmap="gray", vmin=0, vmax=1)
a0.set_title("Denoised reconstruction (Arm H)", fontsize=9); a0.set_xticks([]); a0.set_yticks([])

a1 = fig.add_subplot(gs[1]); s_a = inferno_overlay(a1, mu, sig_a)
reg_a = regions(s_a, 98.5, 2); place_labels(a1, reg_a, C_AL, "M")
a1.set_title("Measurement-limited  $\\hat{\\sigma}_a$", fontsize=9)

a2 = fig.add_subplot(gs[2]); s_e = inferno_overlay(a2, mu, sig_e)
reg_e = regions(s_e, 97.5, 3); place_labels(a2, reg_e, C_EP, "E")
a2.set_title("Model-inferred  $\\hat{\\sigma}_e$  (verify)", fontsize=9)

# shared qualitative inferno colourbar under the two overlays
cax = fig.add_axes([0.30, 0.02, 0.42, 0.03])
ColorbarBase(cax, cmap="inferno", norm=Normalize(0, 1), orientation="horizontal",
             ticks=[0, 1]); cax.set_xticklabels(["lower", "higher"], fontsize=7.5)
cax.set_title("per-pixel uncertainty", fontsize=7.5, pad=2)

# report
a3 = fig.add_subplot(gs[3]); a3.set_xlim(0, 1); a3.set_ylim(0, 1); a3.axis("off")
T = a3.transAxes
def rpt(x, y, s, **k): a3.text(x, y, s, transform=T, va="top", **k)
rpt(0.0, 0.99, "Per-scan reliability report", fontsize=9.5, fontweight="bold")
rpt(0.0, 0.90, f"{len(reg_e)} to verify · {len(reg_a)} measurement-limited", fontsize=8, color="0.3")
rpt(0.0, 0.79, f"{'ID':<4}{'Location':<17}{'Severity'}", fontsize=8, family="monospace", fontweight="bold")
a3.plot([0, 1], [0.745, 0.745], color="0.7", lw=0.8, transform=T, clip_on=False)
y = 0.71
for k, r in enumerate(reg_e, 1):
    rpt(0.0, y, f"{'E'+str(k):<4}{zone(r['cx'],r['cy']):<17}{r['sev']}", fontsize=8, family="monospace", color="#0277BD"); y -= 0.085
for k, r in enumerate(reg_a, 1):
    rpt(0.0, y, f"{'M'+str(k):<4}{zone(r['cx'],r['cy']):<17}{'—'}", fontsize=8, family="monospace", color="#F9A825"); y -= 0.085
rpt(0.0, y-0.03, "Action: cross-check the model-inferred (E)\nregions against the raw low-dose image before\nreporting. Measurement (M) regions are noise-\nlimited, not model uncertainty.", fontsize=7.6, color="0.15")

OUT = ("/Users/sam/Documents/PPVAE Dissertation Project/PpCNN/dissertation-latex/"
       "dissertation-latex/figures/evaluation_v2/fig_uncertainty_hotspot")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print("saved final | E:", [zone(r['cx'],r['cy']) for r in reg_e], "| M:", [zone(r['cx'],r['cy']) for r in reg_a])
