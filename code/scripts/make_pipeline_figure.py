#!/usr/bin/env python3
"""Integrated PP-VAE-Hformer study-pipeline figure (Fig 3.1).
Degradation -> U-Net of HybridBlocks drawn as 3D feature volumes + VAE bottleneck
-> uncertainty-aware outputs (epistemic branches from the VAE via MC sampling)
-> held-out evaluation, over a compact 19-arm ablation design matrix.
Muted, print-oriented palette consistent with the R statistical figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle, Circle
import matplotlib.image as mpimg
import numpy as np, os, sys

FIGDIR = sys.argv[1] if len(sys.argv) > 1 else "."
A = os.path.join(FIGDIR, "arch")
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/fig_study_pipeline.pdf"

# ---- muted, professional palette (seaborn-deep family; matches the R figures) ----
INK   = "#2f2f2f"
NET   = "#4c72b0"   # network (encoder+decoder)
VAE   = "#dd8452"   # stochastic VAE bottleneck  ->  epistemic path
ALE   = "#c44e52"   # aleatoric
EVAL  = "#55a868"   # evaluation
GREY  = "#7f7f7f"
FAINT = "#c9c9c9"
PANEL = "#f4f3f1"   # warm paper for panel fills
PBORD = "#d8d5d0"

FW, FH = 16.0, 10.0
ASP = FW / FH  # 1.6 : convert x-fraction depth to equal-looking y-fraction

def shade(c, f):
    r, g, b = mc.to_rgb(c)
    if f >= 0:  return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)
    f = -f;     return (r * (1 - f), g * (1 - f), b * (1 - f))

fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def load(name, crop_strip=False):
    im = mpimg.imread(os.path.join(A, name))
    if crop_strip:
        im = im[8:534, 4:496]
    return im

def txt(x, y, s, size=9, col=INK, w="normal", ha="center", va="center", style="normal", rot=0):
    ax.text(x, y, s, fontsize=size, color=col, ha=ha, va=va, fontweight=w,
            fontstyle=style, rotation=rot, zorder=20)

def panel(x, y, w, h, fc=PANEL, ec=PBORD, lw=1.1, r=0.012, z=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, mutation_aspect=1/ASP, zorder=z))

def arrow(p0, p1, col=INK, lw=1.7, style="-|>", ls="-", rad=0.0, mut=13, z=8, sa=2, sb=2):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=mut, lw=lw, color=col,
                        ls=ls, shrinkA=sa, shrinkB=sb, zorder=z,
                        connectionstyle=f"arc3,rad={rad}")
    a.set_capstyle("round"); a.set_joinstyle("round"); ax.add_patch(a)

def stage(x, s, col):
    txt(x, 0.988, s.upper(), size=10, col=col, w="bold", ha="left")
    ax.plot([x, x + 0.006], [0.976, 0.976], color=col, lw=3, solid_capstyle="round",
            zorder=9, transform=ax.transAxes)

# =====================================================================
# stage captions (thin, no heavy banner boxes)
# =====================================================================
stage(0.020, "1  Degradation", VAE)
stage(0.175, "2  PP-VAE-Hformer network", NET)
stage(0.660, "3  Uncertainty-aware outputs", ALE)
stage(0.870, "4  Evaluation", EVAL)

# =====================================================================
# ZONE 1 : degradation
# =====================================================================
def framed_img(cx, cy, w, name, label, lcol, crop=True, cmap=None, ecol=INK, lw=1.1):
    h = w * ASP
    a = fig.add_axes([cx - w/2, cy - h/2, w, h])
    a.imshow(load(name, crop), cmap=("gray" if crop else cmap)); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_edgecolor(ecol); s.set_linewidth(lw)
    if label: txt(cx, cy + h/2 + 0.017, label, size=8.2, col=lcol, w="bold")
    return h

hc = framed_img(0.072, 0.865, 0.088, "arch_input_clean.png", "Clean CXR (Kermany)", INK, ecol=GREY)
hn = framed_img(0.072, 0.610, 0.088, "arch_input_noisy.png", "Low-dose CXR", INK, ecol=VAE)
arrow((0.072, 0.865 - hc/2), (0.072, 0.610 + hn/2), col=VAE, lw=2.2, mut=15)
txt(0.128, 0.735, "Foi noise\n$\\mathrm{Var}{=}a\\,y{+}b$\n3 dose levels", size=7.6, col=VAE, w="bold")

# =====================================================================
# ZONE 2 : U-Net drawn as 3D feature volumes
# =====================================================================
def cuboid(cx, cy, res, ch, base=NET, face=None, cmap=None, lab=None, labcol=None):
    """Oblique 3D feature volume; front-face size ~ spatial res, depth ~ channels."""
    fw = 0.0165 + (res / 256.0) * 0.030
    fh = fw * ASP
    dx = 0.0045 + (ch / 512.0) * 0.017
    dy = dx * ASP * 0.72
    x, y = cx - fw/2, cy - fh/2
    top  = shade(base, 0.42); side = shade(base, -0.18)
    ax.add_patch(Polygon([(x, y+fh), (x+fw, y+fh), (x+fw+dx, y+fh+dy), (x+dx, y+fh+dy)],
                 closed=True, fc=top, ec=INK, lw=0.7, zorder=4))
    ax.add_patch(Polygon([(x+fw, y), (x+fw+dx, y+dy), (x+fw+dx, y+fh+dy), (x+fw, y+fh)],
                 closed=True, fc=side, ec=INK, lw=0.7, zorder=4))
    if face:
        a = fig.add_axes([x, y, fw, fh]); a.imshow(load(face), cmap=cmap)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_edgecolor(INK); s.set_linewidth(0.9)
    else:
        ax.add_patch(Rectangle((x, y), fw, fh, fc=shade(base, 0.15), ec=INK, lw=0.9, zorder=5))
    if lab: txt(cx, y - 0.016, lab, size=6.6, col=labcol or shade(base, -0.25))
    return dict(cx=cx, cy=cy, x=x, y=y, fw=fw, fh=fh, dx=dx, dy=dy)

enc = [(0.198, 0.855, 256, 64, "featmap_256.png", "$256^2{\\cdot}64$"),
       (0.256, 0.770, 128, 128, "featmap_128.png", "$128^2{\\cdot}128$"),
       (0.314, 0.688, 64, 256, "featmap_64.png",  "$64^2{\\cdot}256$"),
       (0.372, 0.612, 32, 512, "featmap_32.png",  "$32^2{\\cdot}512$")]
dec = [(0.474, 0.612, 64, 256, "featmap_d64.png",  "$64^2{\\cdot}256$"),
       (0.532, 0.688, 128, 128, "featmap_d128.png", "$128^2{\\cdot}128$"),
       (0.590, 0.770, 256, 64, "featmap_d256.png", "$256^2{\\cdot}64$")]
E = [cuboid(x, y, r, c, base=NET, face=f, cmap="viridis", lab=l) for (x, y, r, c, f, l) in enc]
D = [cuboid(x, y, r, c, base=NET, face=f, cmap="viridis", lab=l) for (x, y, r, c, f, l) in dec]

# stem: noisy image -> first encoder block
arrow((0.116, 0.610), (E[0]["x"] - 0.004, 0.780), col=INK, lw=1.7, rad=-0.18, sb=3)
txt(0.176, 0.905, "stem $3{\\times}3$", size=7.0, col=GREY)
# encoder chain
for i in range(len(E)-1):
    arrow((E[i]["x"]+E[i]["fw"], E[i]["cy"]-0.006), (E[i+1]["x"], E[i+1]["cy"]+0.010),
          col=NET, lw=1.7, rad=-0.05)
# VAE bottleneck
vx, vy, vw, vh = 0.398, 0.480, 0.092, 0.062
panel(vx, vy, vw, vh, fc=shade(VAE, 0.72), ec=VAE, lw=1.5)
txt(vx+vw/2, vy+vh-0.017, "VAE bottleneck", size=7.8, col=shade(VAE,-0.25), w="bold")
txt(vx+vw/2, vy+0.021, "$z=z_\\mu+\\varepsilon\\odot e^{\\frac{1}{2}z_{\\log\\sigma^2}}$", size=7.4, col=INK)
arrow((E[3]["cx"], E[3]["y"]), (vx+vw/2, vy+vh), col=NET, lw=1.7, rad=0.0, sb=2)
arrow((vx+vw, vy+vh/2), (D[0]["x"], D[0]["cy"]-0.004), col=NET, lw=1.7, rad=0.0)
# decoder chain
for i in range(len(D)-1):
    arrow((D[i]["x"]+D[i]["fw"], D[i]["cy"]+0.006), (D[i+1]["x"], D[i+1]["cy"]-0.010),
          col=NET, lw=1.7, rad=0.05)
# skip connections (matching resolution, over the top)
for e, d in zip(E[:3], reversed(D)):
    arrow((e["x"]+e["fw"]+e["dx"], e["y"]+e["fh"]+e["dy"]),
          (d["x"]+d["dx"], d["y"]+d["fh"]+d["dy"]),
          col=GREY, lw=1.1, ls=(0, (5, 3)), style="-|>", rad=-0.10, mut=9)
txt(0.394, 0.905, "skip connections (matched resolution)", size=7.2, col=GREY, style="italic")
txt(0.256, 0.556, "Encoder  ·  3 downsampling stages", size=8.2, col=shade(NET,-0.2), w="bold")
txt(0.532, 0.812, "Decoder + dual head", size=8.2, col=shade(NET,-0.2), w="bold")
# HybridBlock descriptor
panel(0.196, 0.470, 0.192, 0.030, fc="white", ec=PBORD, lw=1.0)
txt(0.292, 0.485, "HybridBlock $=$ WinAttn$_{8\\times8}\\!\\circ$ResBlock,  $\\times2$/stage", size=7.2, col=INK)

# =====================================================================
# ZONE 3 : outputs -- mean & aleatoric from the head; epistemic from the VAE
# =====================================================================
mu_cy, al_cy, ep_cy = 0.860, 0.700, 0.520
ow = 0.088
# dual head split node after last decoder block
hx, hy = D[2]["x"] + D[2]["fw"] + 0.018, D[2]["cy"]
txt(hx + 0.004, hy + 0.052, "dual\n$3{\\times}3$ head", size=6.8, col=shade(NET,-0.2))
framed_img(0.712, mu_cy, ow, "arch_output_mu.png",   "Denoised mean $\\hat\\mu_y$", INK, crop=False, cmap="gray", ecol=INK)
framed_img(0.712, al_cy, ow, "arch_output_aleat.png","Aleatoric $\\hat\\sigma^2_a$  (NLL head)", ALE, crop=False, ecol=ALE)
framed_img(0.712, ep_cy, ow, "arch_output_epist.png","Epistemic $\\hat\\sigma^2_e$  (VAE, $K{=}20$ MC)", VAE, crop=False, ecol=VAE)
# head -> mean & aleatoric
arrow((D[2]["x"]+D[2]["fw"]+D[2]["dx"], D[2]["cy"]+D[2]["fh"]/2), (0.712-ow/2, mu_cy-0.01),
      col=NET, lw=1.8, rad=-0.08)
arrow((hx+0.02, hy-0.01), (0.712-ow/2, al_cy+0.02), col=ALE, lw=1.8, rad=-0.10)
# VAE -> epistemic (the point: epistemic originates from stochastic VAE sampling)
arrow((vx+vw/2, vy), (0.712-ow/2, ep_cy), col=VAE, lw=2.0, rad=-0.34, mut=15)
txt(0.560, 0.455, "$K{=}20$ stochastic $z$ samples", size=7.4, col=shade(VAE,-0.2), w="bold", style="italic")

# =====================================================================
# ZONE 4 : evaluation
# =====================================================================
panel(0.868, 0.500, 0.126, 0.40, fc="white", ec=shade(EVAL,0.4), lw=1.3)
txt(0.931, 0.868, "624 held-out CXRs", size=8.4, col=shade(EVAL,-0.15), w="bold")
ev = [("Reconstruction", "PSNR·SSIM·FSIM·LPIPS"),
      ("Calibration", "reliability + $\\sigma$-scaling"),
      ("Fairness", "Normal/Bacterial/Viral")]
yy = 0.800
for head, body in ev:
    txt(0.876, yy, head, size=7.9, col=shade(EVAL,-0.2), w="bold", ha="left")
    txt(0.876, yy-0.033, body, size=7.3, col=INK, ha="left"); yy -= 0.093
arrow((0.806, 0.700), (0.868, 0.700), col=EVAL, lw=1.9, mut=15)

# =====================================================================
# BOTTOM : objective/baselines strip + 19-arm ablation design matrix (creative Fig 3.7)
# =====================================================================
panel(0.014, 0.028, 0.972, 0.395, fc=PANEL, ec=PBORD, lw=1.2)
txt(0.030, 0.392, "Training objective", size=8.6, col=INK, w="bold", ha="left")
txt(0.030, 0.360, "$\\mathcal{L}=\\mathrm{NLL}+\\lambda_S\\mathrm{SSIM}+\\lambda_F\\mathrm{FFL}+\\beta(t)\\,\\mathrm{KL}$  (cyclic anneal)",
    size=8.6, col=INK, ha="left")
txt(0.030, 0.332, "AdamW · cosine LR · 200 epochs", size=7.4, col=GREY, ha="left")
txt(0.660, 0.392, "Benchmarked against", size=8.6, col=INK, w="bold", ha="left")
txt(0.660, 0.360, "KAIR: DnCNN·IRCNN·FFDNet·DRUNet·SwinIR", size=7.9, col=INK, ha="left")
txt(0.660, 0.334, "SOTA: NAFNet·SCUNet   ·   classical: BM3D", size=7.9, col=INK, ha="left")

# --- ablation dot-matrix ---
arms = list("ABCDEFGHIJKLMNOP") + ["Q", "R", "S"]
rows = ["MSE", "L1", "Charb", "NLL", "SSIM", "FFL", "Edge", "Perc", "VAE/KL", "PReLU"]
active = {
 "MSE": set("A"), "L1": set("IJKR"), "Charb": set("Q"),
 "NLL": set("BCDEFGHKLMNOPS"), "SSIM": set("CDEFGHJMNOPRS"),
 "FFL": set("DEFGHJLMNOPRS"), "Edge": set("LMP"), "Perc": set("N"),
 "VAE/KL": set("EFGHP"), "PReLU": set("OP")}
rowcol = {"MSE": GREY, "L1": GREY, "Charb": GREY, "NLL": NET, "SSIM": NET, "FFL": NET,
          "Edge": NET, "Perc": NET, "VAE/KL": VAE, "PReLU": ALE}
mx0, mx1 = 0.120, 0.640
my1, my0 = 0.300, 0.058
cxs = np.linspace(mx0, mx1, len(arms))
rys = np.linspace(my1, my0, len(rows))
txt((mx0+mx1)/2, 0.392, "19-arm ablation  ·  one factor at a time", size=8.6, col=INK, w="bold")
txt((mx0+mx1)/2, 0.366, "(full term-intersection view: Fig.~3.7 UpSet)", size=7.2, col=GREY, style="italic")
rr = (cxs[1]-cxs[0]) * 0.30
for r, ry in zip(rows, rys):
    txt(mx0 - 0.016, ry, r, size=7.0, col=rowcol[r], w="bold", ha="right")
    on_x = [cx for a, cx in zip(arms, cxs) if a in active[r]]
    off_x = [cx for a, cx in zip(arms, cxs) if a not in active[r]]
    ax.scatter(off_x, [ry]*len(off_x), s=26, marker="o", facecolors="white",
               edgecolors=FAINT, linewidths=0.8, zorder=6)
    ax.scatter(on_x, [ry]*len(on_x), s=30, marker="o", facecolors=rowcol[r],
               edgecolors=shade(rowcol[r], -0.2), linewidths=0.6, zorder=7)
# arm labels + group brackets
grp = [("A", "E", "core"), ("F", "H", "KL"), ("I", "K", "pixel"),
       ("L", "N", "struct"), ("O", "P", "act"), ("Q", "Q", "Charb"), ("R", "S", "FT")]
for a, cx in zip(arms, cxs):
    txt(cx, my1 + 0.024, a, size=6.8, col=INK, w="bold")
ai = {a: i for i, a in enumerate(arms)}
for g0, g1, gl in grp:
    x0, x1 = cxs[ai[g0]], cxs[ai[g1]]
    ax.plot([x0-rr, x1+rr], [my0-0.020, my0-0.020], color=GREY, lw=1.0,
            solid_capstyle="round", zorder=6)
    txt((x0+x1)/2, my0-0.036, gl, size=6.6, col=GREY)

# equalise circle aspect (Circle in fraction coords needs aspect fix): draw as ellipses instead
# (handled visually; ASP≈1.6 keeps them near-round at this scale)

fig.savefig(OUT, dpi=200)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=150)
print("saved", OUT)
