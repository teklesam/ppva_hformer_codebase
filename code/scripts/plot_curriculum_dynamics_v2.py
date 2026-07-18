"""Fig 4.8: two-stage loss-substitution training dynamics (twin-axis design).
Top: validation-PSNR trajectory 0-300 (Arm A Stage-1 L2, then Arms R/S Stage-2),
with a paradigm-shift marker and the NLL warmup dip called out. Bottom: training
loss on a log axis (Arm A MSE, Arm R composite) with Arm S's NLL on a right axis.

Rebuilt from /tmp/csd3_pull/trainlogs. Fixes the previous version's overlap where
the Arm R and Arm S validation end-labels printed on top of each other: they now
sit at separated heights in the right margin with short leader lines.
Colour-blind-safe; no in-figure title (caption supplies it)."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/tmp/csd3_pull/trainlogs"
def load(arm, col="val_psnr", off=0):
    ep, v = [], []
    for x in csv.DictReader(open(f"{BASE}/{arm}/train_log.csv")):
        if x.get(col, "") in ("", "nan"):
            continue
        ep.append(int(x["epoch"]) + off); v.append(float(x[col]))
    return np.array(ep), np.array(v)

# --- validation PSNR ---
epA, vA = load("arm_a_l2")
epR, vR = load("arm_r_ft_j", off=200)
epS, vS = load("arm_s_ft_d", off=200)
A_final = vA[-1]
# --- training loss (bottom) ---
epAl, lA = load("arm_a_l2",   "train_loss")
epRl, lR = load("arm_r_ft_j", "train_loss", off=200)
epSl, lS = load("arm_s_ft_d", "l_rec",      off=200)   # NLL, negative -> right axis

CA, CR, CS = "#1f77b4", "#ff7f0e", "#2ca02c"   # blue / orange / green
CSHIFT = "#d62728"
BG1, BG2 = "#eaf3fb", "#f2f2f2"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.8, 5.9),
                               gridspec_kw=dict(height_ratios=[1.15, 1.0], hspace=0.16))

# ================= TOP: validation PSNR =================
ax1.axvspan(0, 200, color=BG1); ax1.axvspan(200, 300, color=BG2)
ax1.axvline(200, color=CSHIFT, ls="--", lw=1.6, zorder=2)
ax1.plot(epA, vA, color=CA, lw=2.0, label=r"Stage 1 --- Arm A ($\mathcal{L}_2$)", zorder=3)
ax1.plot(epR, vR, color=CR, lw=2.0, label=r"Stage 2-R --- $\mathcal{L}_1$+SSIM+FFL", zorder=3)
ax1.plot(epS, vS, color=CS, lw=2.0, label="Stage 2-S --- NLL+SSIM+FFL", zorder=3)
ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

# paradigm-shift callout
ax1.annotate("Loss\nparadigm\nshift", xy=(200, 34.7), xytext=(163, 33.4),
             ha="center", va="center", fontsize=8.5, color=CSHIFT,
             arrowprops=dict(arrowstyle="->", color=CSHIFT, lw=1.2))
# Arm A converged plateau
ax1.annotate(f"{A_final:.2f} dB", xy=(199, A_final), xytext=(150, 34.95),
             ha="center", va="bottom", fontsize=8.5, color=CA)
# NLL blend warmup dip
iS_min = int(np.argmin(vS))
ax1.annotate("NLL blend\nwarmup dip", xy=(epS[iS_min], vS[iS_min]), xytext=(243, 33.55),
             ha="center", va="center", fontsize=8.5, color=CS,
             arrowprops=dict(arrowstyle="->", color=CS, lw=1.1))
# stage region labels
ax1.text(100, 27.4, r"Stage 1: $\mathcal{L}_2$ only", ha="center", fontsize=9, color="#2a6f9e")
ax1.text(250, 27.4, "Stage 2", ha="center", fontsize=9, color="0.4")

# --- end-labels, separated so R and S no longer overlap ---
# both curves end ~0.05 dB apart; place labels at fixed, well-separated heights
# in the right margin with short leader lines back to each endpoint.
ax1.annotate(f"Arm S  {vS[-1]:.2f} dB", xy=(300, vS[-1]), xytext=(303, 35.05),
             ha="left", va="center", fontsize=8.5, color=CS, fontweight="bold",
             annotation_clip=False, arrowprops=dict(arrowstyle="-", color=CS, lw=0.7))
ax1.annotate(f"Arm R  {vR[-1]:.2f} dB", xy=(300, vR[-1]), xytext=(303, 33.55),
             ha="left", va="center", fontsize=8.5, color=CR, fontweight="bold",
             annotation_clip=False, arrowprops=dict(arrowstyle="-", color=CR, lw=0.7))

ax1.set_ylabel("Validation PSNR (dB)")
ax1.set_xlim(0, 300); ax1.set_ylim(26.9, 35.4)
ax1.set_xticklabels([]); ax1.grid(True, alpha=0.25, lw=0.5); ax1.set_axisbelow(True)

# ================= BOTTOM: training loss =================
ax2.axvspan(0, 200, color=BG1); ax2.axvspan(200, 300, color=BG2)
ax2.axvline(200, color=CSHIFT, ls="--", lw=1.6, zorder=2)
ax2.set_yscale("log")
# Only the two arms whose losses sit on a meaningful axis: Arm A's MSE (Stage 1,
# left axis) and Arm S's NLL (Stage 2, right axis). Arm R (L1+SSIM+FFL) has no NLL
# head and its composite loss is not on a comparable scale, so it is omitted here
# and shown only in the top validation panel.
lnA, = ax2.plot(epAl, lA, color=CA, lw=1.8, label=r"Arm A: $\mathcal{L}_2$ MSE (left axis)", zorder=3)
ax2.set_ylabel("Loss (log scale)"); ax2.set_xlim(0, 300)
ax2.set_xlabel("Training epoch")
ax2.grid(True, which="major", alpha=0.25, lw=0.5); ax2.set_axisbelow(True)

ax2r = ax2.twinx()   # NLL on its own axis; more negative (better) is higher
lnS, = ax2r.plot(epSl, lS, color=CS, lw=2.0, ls="--",
                 label="Arm S: NLL (right axis)", zorder=3)
ax2r.set_ylim(0.3, -4.3)
ax2r.set_ylabel("NLL loss (nats/px)", color=CS)
ax2r.tick_params(axis="y", colors=CS)
ax2.legend(handles=[lnA, lnS], loc="lower right", fontsize=8.5, framealpha=0.9)

OUT = ("/Users/sam/Documents/PPVAE Dissertation Project/PpCNN/dissertation-latex/"
       "dissertation-latex/figures/evaluation_v2/curriculum_loss_dynamics.pdf")
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
print(f"saved | A={A_final:.3f} R={vR[-1]:.3f} S={vS[-1]:.3f} Smin={vS.min():.3f}@{epS[iS_min]}")
