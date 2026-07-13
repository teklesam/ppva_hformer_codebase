#!/usr/bin/env python3
"""PP-VAE-Hformer study-pipeline figure (Fig 3.1), publication style.
Degradation -> U-Net of HybridBlocks as 3D feature volumes + VAE bottleneck ->
uncertainty-aware outputs (epistemic branches from the VAE via MC sampling) ->
held-out evaluation. Muted palette + serif (STIX) type; label haloes and clear
arrow lanes keep it legible."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle
import matplotlib.image as mpimg
import numpy as np, os, sys

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

FIGDIR = sys.argv[1] if len(sys.argv) > 1 else "."
A = os.path.join(FIGDIR, "arch")
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/fig_study_pipeline.pdf"

# ---- muted palette (seaborn-deep family; matches the R statistical figures) ----
INK  = "#2b2b2b"; NET = "#4c72b0"; VAE = "#dd8452"; ALE = "#c44e52"
EVAL = "#55a868"; GREY = "#7f7f7f"; PANEL = "#f5f4f2"; PBORD = "#dcd9d4"

FW, FH = 16.0, 8.6
ASP = FW / FH

def shade(c, f):
    r, g, b = mc.to_rgb(c)
    return (r+(1-r)*f, g+(1-g)*f, b+(1-b)*f) if f >= 0 else (r*(1+f), g*(1+f), b*(1+f))

fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def load(name, crop=False):
    im = mpimg.imread(os.path.join(A, name))
    return im[8:534, 4:496] if crop else im

def txt(x, y, s, size=9, col=INK, w="normal", ha="center", va="center", style="normal",
        halo=False, z=20):
    bb = dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.92) if halo else None
    ax.text(x, y, s, fontsize=size, color=col, ha=ha, va=va, fontweight=w,
            fontstyle=style, bbox=bb, zorder=z)

def panel(x, y, w, h, fc=PANEL, ec=PBORD, lw=1.1, r=0.012, z=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, mutation_aspect=1/ASP, zorder=z))

def arrow(p0, p1, col=INK, lw=1.7, style="-|>", ls="-", rad=0.0, mut=13, z=8, sa=2, sb=2):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=mut, lw=lw, color=col,
                        ls=ls, shrinkA=sa, shrinkB=sb, zorder=z,
                        connectionstyle=f"arc3,rad={rad}")
    a.set_capstyle("round"); a.set_joinstyle("round"); ax.add_patch(a)

def stage(x, s, col):
    txt(x, 0.972, s.upper(), size=10.5, col=col, w="bold", ha="left")
    ax.plot([x, x + 0.052], [0.949, 0.949], color=col, lw=2.2, solid_capstyle="round", zorder=9)

stage(0.018, "1 · Degradation", VAE)
stage(0.150, "2 · PP-VAE-Hformer network", NET)
stage(0.640, "3 · Uncertainty-aware outputs", ALE)
stage(0.858, "4 · Evaluation", EVAL)

def framed(cx, cy, w, name, label, lcol, crop=True, cmap=None, ecol=INK, lw=1.1):
    h = w * ASP
    a = fig.add_axes([cx - w/2, cy - h/2, w, h])
    a.imshow(load(name, crop), cmap=("gray" if crop else cmap)); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_edgecolor(ecol); s.set_linewidth(lw)
    if label: txt(cx, cy + h/2 + 0.026, label, size=8.4, col=lcol, w="bold")
    return h

# =====================================================================
# ZONE 1 : degradation
# =====================================================================
hc = framed(0.070, 0.780, 0.090, "arch_input_clean.png", "Clean CXR (Kermany)", INK, ecol=GREY)
hn = framed(0.070, 0.470, 0.090, "arch_input_noisy.png", "Simulated low-dose CXR", VAE, ecol=VAE)
arrow((0.048, 0.780 - hc/2), (0.048, 0.470 + hn/2), col=VAE, lw=2.3, mut=15)
txt(0.106, 0.625, "Foi noise\n$\\mathrm{Var}=a\\,y+b$\n(3 dose levels)", size=8.0, col=shade(VAE,-0.15),
    w="bold", halo=True)

# =====================================================================
# ZONE 2 : U-Net as 3D feature volumes
# =====================================================================
def cuboid(cx, cy, res, ch, face, cmap="viridis", lab=None, base=NET):
    fw = 0.013 + (res / 256.0) * 0.026
    fh = fw * ASP
    dx = 0.004 + (ch / 512.0) * 0.014
    dy = dx * ASP * 0.7
    x, y = cx - fw/2, cy - fh/2
    ax.add_patch(Polygon([(x, y+fh), (x+fw, y+fh), (x+fw+dx, y+fh+dy), (x+dx, y+fh+dy)],
                 closed=True, fc=shade(base, 0.45), ec=INK, lw=0.7, zorder=4))
    ax.add_patch(Polygon([(x+fw, y), (x+fw+dx, y+dy), (x+fw+dx, y+fh+dy), (x+fw, y+fh)],
                 closed=True, fc=shade(base, -0.22), ec=INK, lw=0.7, zorder=4))
    a = fig.add_axes([x, y, fw, fh]); a.imshow(load(face), cmap=cmap)
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_edgecolor(INK); s.set_linewidth(0.9)
    if lab: txt(cx - 0.006, y - 0.020, lab, size=6.7, col=shade(base, -0.3), halo=True)
    return dict(cx=cx, cy=cy, x=x, y=y, fw=fw, fh=fh, dx=dx, dy=dy)

enc = [(0.176, 0.815, 256, 64, "featmap_256.png", "$256^2$·64"),
       (0.234, 0.705, 128, 128, "featmap_128.png", "$128^2$·128"),
       (0.292, 0.598, 64, 256, "featmap_64.png",  "$64^2$·256"),
       (0.350, 0.502, 32, 512, "featmap_32.png",  "$32^2$·512")]
dec = [(0.452, 0.502, 64, 256, "featmap_d64.png",  "$64^2$·256"),
       (0.510, 0.598, 128, 128, "featmap_d128.png", "$128^2$·128"),
       (0.568, 0.705, 256, 64, "featmap_d256.png", "$256^2$·64")]
E = [cuboid(*e[:4], e[4], lab=e[5]) for e in enc]
D = [cuboid(*d[:4], d[4], lab=d[5]) for d in dec]

# stem
arrow((0.115, 0.500), (E[0]["x"] - 0.003, 0.760), col=INK, lw=1.6, rad=-0.22, sb=3)
txt(0.150, 0.860, "stem $3{\\times}3$", size=7.2, col=GREY, halo=True)
# encoder chain
for i in range(len(E)-1):
    arrow((E[i]["x"]+E[i]["fw"], E[i]["cy"]-0.010), (E[i+1]["x"]-0.002, E[i+1]["cy"]+0.012),
          col=NET, lw=1.7, rad=-0.06)
# VAE bottleneck
vx, vy, vw, vh = 0.360, 0.352, 0.080, 0.070
panel(vx, vy, vw, vh, fc=shade(VAE, 0.74), ec=VAE, lw=1.5)
txt(vx+vw/2, vy+vh-0.019, "VAE bottleneck", size=7.9, col=shade(VAE,-0.25), w="bold")
txt(vx+vw/2, vy+0.023, "$z=z_\\mu+\\varepsilon\\odot e^{\\frac{1}{2}z_{\\log\\sigma^2}}$", size=7.5, col=INK)
arrow((E[3]["cx"], E[3]["y"]), (vx+vw*0.42, vy+vh), col=NET, lw=1.7, rad=0.12, sb=2)
arrow((vx+vw, vy+vh*0.6), (D[0]["x"]-0.002, D[0]["cy"]-0.008), col=NET, lw=1.7, rad=0.12)
# decoder chain
for i in range(len(D)-1):
    arrow((D[i]["x"]+D[i]["fw"], D[i]["cy"]+0.010), (D[i+1]["x"]-0.002, D[i+1]["cy"]-0.012),
          col=NET, lw=1.7, rad=0.06)
# skip connections
for e, d in zip(E[:3], reversed(D)):
    arrow((e["x"]+e["fw"]+e["dx"], e["y"]+e["fh"]+e["dy"]),
          (d["x"]+d["dx"], d["y"]+d["fh"]+d["dy"]),
          col=GREY, lw=1.1, ls=(0, (5, 3)), style="-|>", rad=-0.12, mut=9)
txt(0.372, 0.905, "skip connections (matched resolution)", size=7.4, col=GREY, style="italic", halo=True)
txt(0.234, 0.430, "Encoder — 3 downsampling stages", size=8.3, col=shade(NET,-0.2), w="bold", halo=True)
txt(0.510, 0.790, "Decoder + dual head", size=8.3, col=shade(NET,-0.2), w="bold", halo=True)
panel(0.176, 0.300, 0.216, 0.032, fc="white", ec=PBORD, lw=1.0)
txt(0.284, 0.316, "HybridBlock $=$ WinAttn$_{8\\times8}\\!\\circ$ResBlock,  $\\times2$ per stage",
    size=7.5, col=INK)

# =====================================================================
# ZONE 3 : outputs (mean & aleatoric from head; epistemic from VAE)
# =====================================================================
ow = 0.092
framed(0.700, 0.800, ow, "arch_output_mu.png",    "Denoised mean $\\hat\\mu_y$", INK, crop=False, cmap="gray", ecol=INK)
framed(0.700, 0.560, ow, "arch_output_aleat.png", "Aleatoric $\\hat\\sigma^2_a$  (NLL head)", ALE, crop=False, ecol=ALE)
framed(0.700, 0.320, ow, "arch_output_epist.png", "Epistemic $\\hat\\sigma^2_e$  (VAE sampling)", VAE, crop=False, ecol=VAE)
# dual head split
hx, hy = D[2]["x"] + D[2]["fw"] + D[2]["dx"], D[2]["cy"] + D[2]["fh"]/2
arrow((hx, hy), (0.700 - ow/2, 0.800 - 0.02), col=NET, lw=1.8, rad=-0.10)
arrow((hx, hy - 0.01), (0.700 - ow/2, 0.560 + 0.03), col=ALE, lw=1.8, rad=0.10)
# epistemic from the VAE, in a clear lane below the decoder
arrow((vx + vw*0.5, vy), (0.700 - ow/2, 0.320), col=VAE, lw=2.0, rad=-0.30, mut=15)
txt(0.520, 0.300, "$K{=}20$ MC samples of $z$", size=7.6, col=shade(VAE,-0.2), w="bold",
    style="italic", halo=True)

# =====================================================================
# ZONE 4 : evaluation
# =====================================================================
panel(0.856, 0.360, 0.130, 0.470, fc="white", ec=shade(EVAL,0.35), lw=1.3)
txt(0.921, 0.792, "624 held-out CXRs", size=8.6, col=shade(EVAL,-0.2), w="bold")
ev = [("Reconstruction", "PSNR · SSIM · FSIM · LPIPS"),
      ("Calibration", "reliability + $\\sigma$-scaling"),
      ("Subgroup fairness", "Normal / Bacterial / Viral")]
yy = 0.715
for head, body in ev:
    txt(0.866, yy, head, size=8.0, col=shade(EVAL,-0.25), w="bold", ha="left")
    txt(0.866, yy-0.038, body, size=7.4, col=INK, ha="left"); yy -= 0.115
arrow((0.796, 0.560), (0.856, 0.560), col=EVAL, lw=1.9, mut=15)

# =====================================================================
# FOOTER : objective + baselines (design detailed in the loss table)
# =====================================================================
panel(0.016, 0.045, 0.968, 0.150, fc=PANEL, ec=PBORD, lw=1.1)
txt(0.034, 0.155, "Training objective", size=9.2, col=INK, w="bold", ha="left")
txt(0.034, 0.113, "$\\mathcal{L}=\\mathrm{NLL}+\\lambda_S\\,\\mathrm{SSIM}+\\lambda_F\\,\\mathrm{FFL}+\\beta(t)\\,\\mathrm{KL}$",
    size=10.5, col=INK, ha="left")
txt(0.034, 0.075, "cyclic KL anneal · AdamW · cosine LR · 200 epochs · 19-arm ablation (Table 3.4)",
    size=8.0, col=GREY, ha="left")
ax.plot([0.560, 0.560], [0.062, 0.178], color=PBORD, lw=1.0, zorder=1)
txt(0.582, 0.155, "Benchmarked against", size=9.2, col=INK, w="bold", ha="left")
txt(0.582, 0.117, "KAIR:  DnCNN · IRCNN · FFDNet · DRUNet · SwinIR", size=8.6, col=INK, ha="left")
txt(0.582, 0.081, "SOTA:  NAFNet · SCUNet        classical:  BM3D", size=8.6, col=INK, ha="left")

fig.savefig(OUT, dpi=200)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=150)
print("saved", OUT)
