#!/usr/bin/env python3
"""PP-VAE-Hformer study-pipeline figure (Fig 3.1), publication style.
Four light phase panels; a symmetric 3D U-Net; orthogonal ('zig-zag') routing for
the stem and VAE connectors; horizontal skip lines; epistemic branches from the VAE.
Serif (STIX) type, label haloes so no text is crossed by a line."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle
from matplotlib.path import Path as MPath
import matplotlib.image as mpimg
import numpy as np, os, sys

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})

FIGDIR = sys.argv[1] if len(sys.argv) > 1 else "."
A = os.path.join(FIGDIR, "arch")
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/fig_study_pipeline.pdf"

INK  = "#2b2b2b"; NET = "#4c72b0"; VAE = "#dd8452"; ALE = "#c44e52"
EVAL = "#55a868"; GREY = "#7f7f7f"; PBORD = "#d7d4cf"
FW, FH = 16.0, 8.8
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
        halo=False, z=25):
    bb = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.95) if halo else None
    ax.text(x, y, s, fontsize=size, color=col, ha=ha, va=va, fontweight=w, fontstyle=style,
            bbox=bb, zorder=z)

def rbox(x, y, w, h, fc, ec, lw=1.1, r=0.013, z=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, mutation_aspect=1/ASP, zorder=z))

def arrow(p0, p1, col=INK, lw=1.7, style="-|>", ls="-", rad=0.0, mut=13, z=10, sa=2, sb=2):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=mut, lw=lw, color=col, ls=ls,
                        shrinkA=sa, shrinkB=sb, zorder=z, connectionstyle=f"arc3,rad={rad}")
    a.set_capstyle("round"); a.set_joinstyle("round"); ax.add_patch(a)

def zarrow(pts, col=INK, lw=1.7, mut=13, z=10):
    codes = [MPath.MOVETO] + [MPath.LINETO]*(len(pts)-1)
    a = FancyArrowPatch(path=MPath(pts, codes), arrowstyle="-|>", mutation_scale=mut, lw=lw,
                        color=col, zorder=z, shrinkA=1, shrinkB=1)
    a.set_capstyle("round"); a.set_joinstyle("round"); ax.add_patch(a)

def framed(cx, cy, w, name, label, lcol, crop=True, cmap=None, ecol=INK, lw=1.1, lpos="top"):
    h = w * ASP
    a = fig.add_axes([cx - w/2, cy - h/2, w, h])
    a.imshow(load(name, crop), cmap=("gray" if crop else cmap)); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_edgecolor(ecol); s.set_linewidth(lw)
    if label:
        ly = cy + h/2 + 0.028 if lpos == "top" else cy - h/2 - 0.030
        txt(cx, ly, label, size=8.0, col=lcol, w="bold")
    return h

# =====================================================================
# phase panels (aligned; light fills)
# =====================================================================
yT, yB = 0.905, 0.250
P = {"1": (0.010, 0.150, VAE, "1 · Degradation"),
     "2": (0.160, 0.628, NET, "2 · PP-VAE-Hformer network"),
     "3": (0.638, 0.826, ALE, "3 · Uncertainty-aware outputs"),
     "4": (0.836, 0.990, EVAL, "4 · Evaluation")}
for k, (x0, x1, col, title) in P.items():
    rbox(x0, yB, x1 - x0, yT - yB, fc=shade(col, 0.92), ec=shade(col, 0.30), lw=1.3)
    xc = (x0 + x1) / 2
    tsz = 9.5 if k != "3" else 8.4
    txt(xc, 0.955, title, size=tsz, col=shade(col, -0.35), w="bold")
    ax.plot([xc-0.028, xc+0.028], [0.936, 0.936], color=shade(col, -0.1), lw=1.8,
            solid_capstyle="round", zorder=9)

# =====================================================================
# PHASE 1 : degradation
# =====================================================================
framed(0.080, 0.700, 0.086, "arch_input_clean.png", "Clean CXR\n(Kermany)", INK, ecol=GREY)
framed(0.080, 0.400, 0.086, "arch_input_noisy.png", "Simulated\nlow-dose CXR", VAE, ecol=VAE, lpos="bottom")
arrow((0.058, 0.615), (0.058, 0.485), col=VAE, lw=2.3, mut=15)
txt(0.108, 0.548, "Foi noise\n$\\mathrm{Var}{=}a\\,y{+}b$\n(3 doses)", size=7.6, col=shade(VAE,-0.15),
    w="bold", halo=True)

# =====================================================================
# PHASE 2 : symmetric 3D U-Net
# =====================================================================
LV = {256: 0.812, 128: 0.706, 64: 0.600}          # skip-connected levels
def cuboid(cx, cy, res, ch, face, lab, side):
    fw = 0.012 + (res/256.0)*0.026; fh = fw*ASP
    dx = 0.004 + (ch/512.0)*0.014;  dy = dx*ASP*0.7
    x, y = cx-fw/2, cy-fh/2
    ax.add_patch(Polygon([(x,y+fh),(x+fw,y+fh),(x+fw+dx,y+fh+dy),(x+dx,y+fh+dy)],
                 closed=True, fc=shade(NET,0.45), ec=INK, lw=0.7, zorder=6))
    ax.add_patch(Polygon([(x+fw,y),(x+fw+dx,y+dy),(x+fw+dx,y+fh+dy),(x+fw,y+fh)],
                 closed=True, fc=shade(NET,-0.22), ec=INK, lw=0.7, zorder=6))
    a = fig.add_axes([x, y, fw, fh]); a.imshow(load(face), cmap="viridis")
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_edgecolor(INK); s.set_linewidth(0.9)
    lx = x - 0.006 if side == "L" else x + fw + dx + 0.006
    txt(lx, cy, lab, size=6.6, col=shade(NET,-0.3), ha=("right" if side == "L" else "left"),
        halo=True)
    return dict(cx=cx, cy=cy, x=x, y=y, fw=fw, fh=fh, dx=dx, dy=dy)

E = [cuboid(0.206, LV[256], 256, 64, "featmap_256.png", "$256^2$·64", "L"),
     cuboid(0.254, LV[128], 128, 128, "featmap_128.png", "$128^2$·128", "L"),
     cuboid(0.302, LV[64], 64, 256, "featmap_64.png",  "$64^2$·256", "L"),
     cuboid(0.350, 0.498, 32, 512, "featmap_32.png",  "$32^2$·512", "L")]
D = [cuboid(0.470, LV[64], 64, 256, "featmap_d64.png",  "$64^2$·256", "R"),
     cuboid(0.518, LV[128], 128, 128, "featmap_d128.png", "$128^2$·128", "R"),
     cuboid(0.566, LV[256], 256, 64, "featmap_d256.png", "$256^2$·64", "R")]

# stem: noisy -> E0  (orthogonal riser in the left margin of the panel)
E0L = E[0]["x"]
zarrow([(0.123, 0.400), (0.178, 0.400), (0.178, LV[256]), (E0L, LV[256])], col=INK, lw=1.7)
txt(0.178, 0.862, "stem\n$3{\\times}3$", size=7.0, col=GREY, halo=True)
# encoder staircase
for i in range(3):
    zarrow([(E[i]["x"]+E[i]["fw"], E[i]["cy"]-E[i]["fh"]*0.32),
            (E[i+1]["cx"], E[i]["cy"]-E[i]["fh"]*0.32),
            (E[i+1]["cx"], E[i+1]["cy"]+E[i+1]["fh"]/2)], col=NET, lw=1.6, mut=11)
# VAE bottleneck
vx, vy, vw, vh = 0.372, 0.372, 0.084, 0.068
rbox(vx, vy, vw, vh, fc=shade(VAE,0.72), ec=VAE, lw=1.5, z=6)
txt(vx+vw/2, vy+vh-0.017, "VAE bottleneck", size=7.7, col=shade(VAE,-0.25), w="bold")
txt(vx+vw/2, vy+0.021, "$z=z_\\mu+\\varepsilon\\odot e^{\\frac{1}{2}z_{\\log\\sigma^2}}$", size=7.2, col=INK)
# E3 -> VAE (down, then right into left face) ; VAE -> D0 (right, then up into base)
zarrow([(E[3]["cx"], E[3]["y"]), (E[3]["cx"], vy+vh/2), (vx, vy+vh/2)], col=NET, lw=1.7)
zarrow([(vx+vw, vy+vh/2), (D[0]["cx"], vy+vh/2), (D[0]["cx"], D[0]["y"])], col=NET, lw=1.7)
# decoder staircase
for i in range(2):
    zarrow([(D[i]["x"]+D[i]["fw"]+D[i]["dx"], D[i]["cy"]+D[i]["fh"]*0.30),
            (D[i+1]["cx"], D[i]["cy"]+D[i]["fh"]*0.30),
            (D[i+1]["cx"], D[i+1]["cy"]-D[i+1]["fh"]/2)], col=NET, lw=1.6, mut=11)
# horizontal skip connections at matched levels
for res in (256, 128, 64):
    e = next(b for b in E if abs(b["cy"]-LV[res]) < 1e-6)
    d = next(b for b in D if abs(b["cy"]-LV[res]) < 1e-6)
    zarrow([(e["x"]+e["fw"]+e["dx"], LV[res]), (d["x"], LV[res])],
           col=GREY, lw=1.05, mut=8)
txt(0.386, 0.860, "skip connections", size=7.2, col=GREY, style="italic", halo=True)
txt(0.256, 0.300, "Encoder — 3 downsampling stages", size=8.0, col=shade(NET,-0.2), w="bold", halo=True)
txt(0.516, 0.300, "Decoder + dual head", size=8.0, col=shade(NET,-0.2), w="bold", halo=True)
rbox(0.300, 0.262, 0.190, 0.030, fc="white", ec=PBORD, lw=1.0, z=5)
txt(0.395, 0.277, "HybridBlock $=$ WinAttn$_{8\\times8}\\!\\circ$ResBlock", size=7.2, col=INK)

# =====================================================================
# PHASE 3 : outputs (mean & aleatoric from head; epistemic from VAE)
# =====================================================================
ocx, ow = 0.732, 0.088
framed(ocx, 0.770, ow, "arch_output_mu.png",    "Denoised mean $\\hat\\mu_y$", INK, crop=False, cmap="gray", ecol=INK)
framed(ocx, 0.560, ow, "arch_output_aleat.png", "Aleatoric $\\hat\\sigma^2_a$  (NLL head)", ALE, crop=False, ecol=ALE)
framed(ocx, 0.350, ow, "arch_output_epist.png", "Epistemic $\\hat\\sigma^2_e$  (VAE)", VAE, crop=False, ecol=VAE)
# dual head -> mean (from D2 right) & aleatoric (from D2 base): orthogonal into each output
hx = D[2]["x"] + D[2]["fw"] + D[2]["dx"]
zarrow([(hx, D[2]["cy"]), (0.664, D[2]["cy"]), (0.664, 0.770), (ocx-ow/2, 0.770)], col=NET, lw=1.7)
zarrow([(D[2]["cx"], D[2]["y"]), (D[2]["cx"], 0.560), (ocx-ow/2, 0.560)], col=ALE, lw=1.7)
txt(0.614, 0.690, "dual head", size=7.0, col=shade(NET,-0.2), halo=True)
# epistemic: from the VAE, clear lane below the U-Net
zarrow([(vx+vw/2, vy), (vx+vw/2, 0.350), (ocx-ow/2, 0.350)], col=VAE, lw=1.9, mut=14)
txt(0.545, 0.372, "$K{=}20$ MC samples of $z$", size=7.4, col=shade(VAE,-0.2), w="bold",
    style="italic", halo=True)

# =====================================================================
# PHASE 4 : evaluation
# =====================================================================
txt(0.913, 0.815, "624 held-out CXRs", size=8.4, col=shade(EVAL,-0.2), w="bold")
ev = [("Reconstruction", "PSNR · SSIM", "FSIM · LPIPS"),
      ("Calibration", "reliability +", "$\\sigma$-scaling"),
      ("Subgroup fairness", "Normal / Bacterial", "/ Viral")]
yy = 0.720
for head, l1, l2 in ev:
    txt(0.848, yy, head, size=7.9, col=shade(EVAL,-0.25), w="bold", ha="left")
    txt(0.848, yy-0.035, l1, size=7.2, col=INK, ha="left")
    txt(0.848, yy-0.065, l2, size=7.2, col=INK, ha="left"); yy -= 0.135
arrow((0.826, 0.560), (0.858, 0.560), col=EVAL, lw=1.9, mut=14)

# =====================================================================
# FOOTER : objective + baselines
# =====================================================================
rbox(0.010, 0.045, 0.980, 0.150, fc="#f5f4f2", ec=PBORD, lw=1.1)
txt(0.030, 0.156, "Training objective", size=9.0, col=INK, w="bold", ha="left")
txt(0.030, 0.114, "$\\mathcal{L}=\\mathrm{NLL}+\\lambda_S\\,\\mathrm{SSIM}+\\lambda_F\\,\\mathrm{FFL}+\\beta(t)\\,\\mathrm{KL}$",
    size=10.5, col=INK, ha="left")
txt(0.030, 0.076, "cyclic KL anneal · AdamW · cosine LR · 200 epochs · 19-arm ablation",
    size=7.9, col=GREY, ha="left")
ax.plot([0.560, 0.560], [0.060, 0.180], color=PBORD, lw=1.0, zorder=1)
txt(0.582, 0.156, "Benchmarked against", size=9.0, col=INK, w="bold", ha="left")
txt(0.582, 0.116, "KAIR:  DnCNN · IRCNN · FFDNet · DRUNet · SwinIR", size=8.4, col=INK, ha="left")
txt(0.582, 0.080, "SOTA:  NAFNet · SCUNet         classical:  BM3D", size=8.4, col=INK, ha="left")

fig.savefig(OUT, dpi=200)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=150)
print("saved", OUT)
