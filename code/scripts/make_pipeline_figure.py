#!/usr/bin/env python3
"""Integrated PP-VAE-Hformer pipeline figure (replaces fig_study_pipeline).
Real CXR -> Foi degradation -> U-Net of HybridBlocks + VAE bottleneck ->
dual uncertainty-aware outputs -> training/evaluation. Uses real image thumbnails
and real encoder/decoder feature maps."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.image as mpimg
import numpy as np, os, sys

FIGDIR = sys.argv[1] if len(sys.argv) > 1 else "."
A = os.path.join(FIGDIR, "arch")
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/fig_study_pipeline.pdf"

# palette
BLUE   = "#2563eb"; LBLUE = "#dbeafe"
GREEN  = "#0f766e"; LGREEN = "#ccfbf1"
AMBER  = "#b45309"; LAMBER = "#fef3c7"
PURPLE = "#6d28d9"; LPUR  = "#ede9fe"
GREY   = "#334155"; LGREY = "#f1f5f9"
EDGE   = "#cbd5e1"

fig = plt.figure(figsize=(16, 8.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def load(name, crop_strip=False):
    im = mpimg.imread(os.path.join(A, name))
    if crop_strip:  # clean/noisy inputs carry a thin right-hand strip
        im = im[8:534, 4:496]
    return im

def tile(rect, name, label=None, lcol=GREY, crop=False, border=EDGE, lw=1.0, cmap=None):
    a = fig.add_axes(rect); a.imshow(load(name, crop), cmap=("gray" if crop else cmap))
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_edgecolor(border); s.set_linewidth(lw)
    if label:
        a.set_title(label, fontsize=8.0, color=lcol, pad=2.5)
    return a

def box(x, y, w, h, fc, ec, lw=1.2, r=0.014):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, mutation_aspect=0.55, zorder=1)
    ax.add_patch(p); return p

def arrow(p0, p1, col=GREY, lw=1.8, style="-|>", ls="-", rad=0.0, mut=14, z=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=mut,
        lw=lw, color=col, ls=ls, shrinkA=1, shrinkB=1, zorder=z,
        connectionstyle=f"arc3,rad={rad}"))

def txt(x, y, s, size=9, col=GREY, w="normal", ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=size, color=col, ha=ha, va=va, fontweight=w,
            fontstyle=style, zorder=8)

# ---- stage banners ----
def banner(x, w, s, col, lcol):
    box(x, 0.945, w, 0.045, lcol, col, lw=1.3)
    txt(x + w/2, 0.9675, s, size=10.5, col=col, w="bold")
banner(0.010, 0.140, "1 · Degradation", AMBER, LAMBER)
banner(0.165, 0.500, "2 · PP-VAE-Hformer", BLUE, LBLUE)
banner(0.680, 0.155, "3 · Uncertainty-aware outputs", PURPLE, LPUR)
banner(0.845, 0.145, "4 · Held-out evaluation", GREEN, LGREEN)

# =========================================================================
# ZONE 1 : degradation
# =========================================================================
box(0.010, 0.300, 0.140, 0.610, "#fffdf7", LAMBER, lw=1.4)
tile([0.028, 0.700, 0.104, 0.150], "arch_input_clean.png", "Clean CXR (Kermany)", AMBER, crop=True)
tile([0.028, 0.345, 0.104, 0.150], "arch_input_noisy.png", "Simulated low-dose CXR", AMBER, crop=True)
arrow((0.058, 0.695), (0.058, 0.500), col=AMBER, lw=2.2)
txt(0.106, 0.607, "Foi noise\n$\\mathrm{Var}{=}a\\,y{+}b$\n3 dose levels",
    size=7.8, col=AMBER, w="bold", ha="center")

# =========================================================================
# ZONE 2 : U-Net architecture with real feature maps
# =========================================================================
box(0.165, 0.300, 0.500, 0.610, "#fbfdff", LBLUE, lw=1.4)
tw, th = 0.052, 0.096
enc = [("featmap_256.png", 0.185, 0.730, "$256^2$·64"),
       ("featmap_128.png", 0.247, 0.620, "$128^2$·128"),
       ("featmap_64.png",  0.309, 0.510, "$64^2$·256"),
       ("featmap_32.png",  0.371, 0.400, "$32^2$·512")]
dec = [("featmap_d64.png",  0.505, 0.510, "$64^2$·256"),
       ("featmap_d128.png", 0.567, 0.620, "$128^2$·128"),
       ("featmap_d256.png", 0.629, 0.730, "$256^2$·64")]
def ctr(x, y): return (x + tw/2, y + th/2)
epos, dpos = [], []
for n, x, y, lab in enc:
    tile([x, y, tw, th], n, None, border=BLUE, lw=1.2)
    txt(x + tw/2, y - 0.022, lab, size=6.6, col=BLUE); epos.append((x, y))
for n, x, y, lab in dec:
    tile([x, y, tw, th], n, None, border=PURPLE, lw=1.2)
    txt(x + tw/2, y - 0.022, lab, size=6.6, col=PURPLE); dpos.append((x, y))

# stem arrow from noisy image into encoder
arrow((0.134, 0.430), (0.184, ctr(*epos[0][:2])[1] - 0.01), col=GREY, lw=1.8, rad=-0.12)
txt(0.183, 0.842, "stem $3{\\times}3$\n$\\to$64 ch", size=6.6, col=GREY)
# encoder chain (down)
for i in range(len(epos) - 1):
    arrow(ctr(epos[i][0], epos[i][1]), ctr(epos[i+1][0], epos[i+1][1]), col=BLUE, lw=1.8)
# VAE bottleneck
vx, vy, vw, vh = 0.398, 0.352, 0.086, 0.058
box(vx, vy, vw, vh, LBLUE, BLUE, lw=1.6)
txt(vx + vw/2, vy + vh/2 + 0.010, "VAE bottleneck", size=7.4, col=BLUE, w="bold")
txt(vx + vw/2, vy + vh/2 - 0.013, "$z=z_\\mu+\\varepsilon\\odot e^{\\frac{1}{2} z_{\\log\\sigma^2}}$", size=6.9, col=BLUE)
arrow(ctr(*epos[-1]), (vx + vw/2, vy + vh), col=BLUE, lw=1.8)
arrow((vx + vw, vy + vh/2), ctr(dpos[0][0], dpos[0][1]), col=PURPLE, lw=1.8)
# decoder chain (up)
for i in range(len(dpos) - 1):
    arrow(ctr(dpos[i][0], dpos[i][1]), ctr(dpos[i+1][0], dpos[i+1][1]), col=PURPLE, lw=1.8)
# skip connections (dashed, encoder->decoder at matching resolution, near-horizontal)
for (ex, ey), (dx, dy) in zip(epos[:3], reversed(dpos)):
    arrow((ex + tw, ey + th/2), (dx, dy + th/2), col="#94a3b8", lw=1.2, ls=(0, (5, 3)),
          style="-|>", rad=-0.06, mut=9)
txt(0.415, 0.792, "skip connections", size=7.0, col="#64748b", style="italic")
# labels
txt(0.255, 0.372, "Encoder  (3 downsampling stages)", size=8.4, col=BLUE, w="bold")
txt(0.575, 0.855, "Decoder + dual head", size=8.4, col=PURPLE, w="bold")
box(0.183, 0.306, 0.300, 0.030, "#ffffff", EDGE, lw=1.0)
txt(0.333, 0.321, "HybridBlock $=$ WinAttn$_{8\\times8}$(ResBlock($\\cdot$))  $\\times2$/stage  ·  GroupNorm+GELU",
    size=7.4, col=GREY)

# =========================================================================
# ZONE 3 : outputs
# =========================================================================
box(0.680, 0.300, 0.155, 0.610, "#fefcff", LPUR, lw=1.4)
oy = [0.700, 0.520, 0.340]
onm = [("arch_output_mu.png", "Denoised mean  $\\hat\\mu_y$", PURPLE, "gray"),
       ("arch_output_aleat.png", "Aleatoric  $\\hat\\sigma^2_a$  (NLL)", PURPLE, "magma"),
       ("arch_output_epist.png", "Epistemic  $\\hat\\sigma^2_e$  ($K{=}20$ MC)", PURPLE, "magma")]
for (n, lab, c, cm), y in zip(onm, oy):
    tile([0.705, y, 0.105, 0.150], n, lab, c, border=PURPLE, lw=1.2, cmap=cm)
txt(0.7575, 0.318, "per-pixel mean $+$ trust map", size=7.6, col=PURPLE, w="bold")

# connect decoder head -> outputs
arrow((0.629 + tw, 0.730 + th/2), (0.705, 0.800), col=PURPLE, lw=2.0, rad=-0.1)

# =========================================================================
# ZONE 4 : evaluation (right column)
# =========================================================================
box(0.845, 0.300, 0.145, 0.610, "#f7fefb", LGREEN, lw=1.4)
txt(0.9175, 0.865, "624 held-out CXRs", size=8.6, col=GREEN, w="bold")
ev = ["Reconstruction:", "  PSNR · SSIM · FSIM · LPIPS", "", "Uncertainty calibration:",
      "  reliability + $\\sigma$-scaling", "", "Subgroup fairness:",
      "  Normal / Bacterial / Viral"]
yy = 0.80
for line in ev:
    b = line.endswith(":")
    txt(0.853, yy, line, size=7.8, col=GREEN if b else GREY, w="bold" if b else "normal", ha="left")
    yy -= 0.045
arrow((0.835, 0.605), (0.845, 0.605), col=GREEN, lw=2.0)

# =========================================================================
# BOTTOM BAND : training / what was tested
# =========================================================================
box(0.010, 0.035, 0.980, 0.225, LGREY, EDGE, lw=1.3)
txt(0.500, 0.238, "Training objective  &  ablation design", size=10.0, col=GREY, w="bold")
# loss box
box(0.024, 0.055, 0.300, 0.155, "#ffffff", BLUE, lw=1.3)
txt(0.174, 0.192, "Composite objective", size=9.0, col=BLUE, w="bold")
txt(0.174, 0.150, "$\\mathcal{L}=\\mathrm{NLL}+\\lambda_S\\,\\mathrm{SSIM}+\\lambda_F\\,\\mathrm{FFL}$", size=9.2, col=GREY)
txt(0.174, 0.115, "$+\\;\\beta(t)\\,\\mathrm{KL}$   (cyclic anneal)", size=9.2, col=GREY)
txt(0.174, 0.075, "AdamW · cosine LR · 200 epochs", size=7.6, col="#64748b")
# ablation box
box(0.348, 0.055, 0.304, 0.155, "#ffffff", AMBER, lw=1.3)
txt(0.500, 0.192, "19-arm ablation  (one factor at a time)", size=9.0, col=AMBER, w="bold")
for i, line in enumerate(["A–E  core cumulative (NLL→SSIM→FFL→VAE)",
                          "F–H  KL schedule    ·   I–K  pixel norm",
                          "L–N  structural reg  ·   O–P  activation",
                          "Q  Charbonnier   ·   R–S  two-stage fine-tune"]):
    txt(0.360, 0.163 - i*0.030, line, size=7.8, col=GREY, ha="left")
txt(0.500, 0.062, "factorial design detailed in Fig.~3.7 (UpSet)", size=7.2, col=AMBER, style="italic")
# baselines box
box(0.676, 0.055, 0.300, 0.155, "#ffffff", GREEN, lw=1.3)
txt(0.826, 0.192, "Benchmarked against", size=9.0, col=GREEN, w="bold")
txt(0.826, 0.150, "KAIR: DnCNN · IRCNN · FFDNet", size=8.2, col=GREY)
txt(0.826, 0.120, "DRUNet · SwinIR", size=8.2, col=GREY)
txt(0.826, 0.088, "SOTA: NAFNet · SCUNet · BM3D", size=8.2, col=GREY)

# flow arrows between top zones (subtle, at banner level handled by inter-box arrows)
for x0, x1 in [(0.150, 0.165), (0.665, 0.680)]:
    arrow((x0, 0.605), (x1, 0.605), col="#94a3b8", lw=2.0, mut=16)

fig.savefig(OUT, dpi=200, bbox_inches=None)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=150)
print("saved", OUT)
