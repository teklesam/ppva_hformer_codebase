"""Fig 4.8 (redesigned): two-stage loss-substitution validation dynamics, made readable.
Top: full validation-PSNR trajectory over 300 epochs (Arm A Stage-1 base, then Arms R/S
Stage-2). Bottom: a zoom on Stage 2 (epochs 200-300) where R and S separate clearly -- R
(L1 base) stays flat, S (NLL base) dips as the variance head re-stabilises, then recovers.
Reads the per-epoch train_log.csv for A/R/S. Colour-blind-safe, no in-figure title."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/tmp/csd3_pull/trainlogs"
def load(arm, off=0):
    r = list(csv.DictReader(open(f"{BASE}/{arm}/train_log.csv")))
    ep, vp = [], []
    for x in r:
        if x["val_psnr"] in ("", "nan"):   # validation logged every 5 epochs; drop gaps
            continue
        ep.append(int(x["epoch"]) + off); vp.append(float(x["val_psnr"]))
    return np.array(ep), np.array(vp)

epA, vA = load("arm_a_l2")            # Stage 1: epochs 0-199
epR, vR = load("arm_r_ft_j", 200)     # Stage 2: 200-299
epS, vS = load("arm_s_ft_d", 200)
A_final = vA[-1]                       # Arm A converged validation level

CB, CR, CS = "#333333", "#0072B2", "#D55E00"   # dark / blue / vermillion
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 7.0),
                               gridspec_kw=dict(height_ratios=[1.0, 1.15], hspace=0.28))

# ---- Top: full trajectory ----
ax1.axvspan(0, 200, color="#f7f7f7"); ax1.axvspan(200, 300, color="#eef5fb", alpha=0.6)
ax1.axvline(200, color="0.4", ls="--", lw=1.1, zorder=1)
ax1.plot(epA, vA, color=CB, lw=1.8, label="Arm A --- Stage 1 base (L2)", zorder=3)
ax1.plot(epR, vR, color=CR, lw=2.2, label="Arm R --- Stage 2 (L1+SSIM+FFL)", zorder=3)
ax1.plot(epS, vS, color=CS, lw=2.2, label="Arm S --- Stage 2 (NLL+SSIM+FFL)", zorder=3)
# legend ABOVE the panel so it never overlaps the curves or stage labels
ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=8.5)
ax1.text(100, 30.0, "Stage 1: MSE", ha="center", fontsize=9, color="0.4")
ax1.text(250, 30.0, "Stage 2: composite loss", ha="center", fontsize=9, color="#2a6f9e")
ax1.set_ylabel("validation PSNR (dB)"); ax1.set_xlim(0, 300); ax1.set_title("")

# ---- Bottom: Stage-2 zoom (direct on-curve labels, no legend box) ----
ax2.axhline(A_final, color=CB, ls=":", lw=1.4)
ax2.text(202, A_final+0.02, f"Arm A converged ({A_final:.2f} dB val)", ha="left", va="bottom", fontsize=8.5, color=CB)
ax2.plot(epR, vR, color=CR, lw=2.0, marker="o", ms=3.2)
ax2.plot(epS, vS, color=CS, lw=2.0, marker="s", ms=3.2)
# label each curve directly, in clear space (R above its flat line, S below in the gap)
ax2.text(258, 34.60, "Arm R (L1 base): stays flat", ha="center", color=CR, fontsize=9, fontweight="bold")
ax2.text(255, 33.62, "Arm S (NLL base): dips, then recovers", ha="center", color=CS, fontsize=9, fontweight="bold")
ax2.annotate("test 34.545 dB", (epR[-1], vR[-1]), xytext=(-4, 7), textcoords="offset points",
             ha="right", color=CR, fontsize=8.5)
ax2.annotate("test 34.608 dB", (epS[-1], vS[-1]), xytext=(-4, -13), textcoords="offset points",
             ha="right", color=CS, fontsize=8.5)
ax2.set_xlabel("epoch"); ax2.set_ylabel("validation PSNR (dB)  [Stage 2 zoom]")
ax2.set_xlim(200, 300)
lo = np.nanmin([np.nanmin(vR), np.nanmin(vS)]) - 0.15
ax2.set_ylim(lo, max(A_final, np.nanmax(vR), np.nanmax(vS)) + 0.28)

for ax in (ax1, ax2):
    ax.grid(True, alpha=0.25, lw=0.5); ax.set_axisbelow(True)
OUT = "results/figures/curriculum_loss_dynamics.pdf"
fig.savefig(OUT, bbox_inches="tight", dpi=200)
fig.savefig(OUT.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
print("saved", OUT, f"| A_final={A_final:.3f} R_final={vR[-1]:.3f} S_final={vS[-1]:.3f}")
