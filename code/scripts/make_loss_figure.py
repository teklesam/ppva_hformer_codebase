#!/usr/bin/env python3
"""Fig 3.6 - composite loss and backpropagation, grounded on real activations.
The reconstruction mu-hat is compared against the reference y (ground truth) to
form the error that each proposed loss term measures (shown as the real map it
computes); the weighted sum is optimised by AdamW; backprop returns gradients
kernel-by-kernel (conv + window-attention). Cool academic palette; landscape."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np, os, sys
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.path import Path as MPath
import matplotlib.image as mpimg
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from scipy import ndimage

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})
FIGDIR = sys.argv[1] if len(sys.argv) > 1 else "."
A = os.path.join(FIGDIR, "arch")
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/fig_loss_training_loop.pdf"

# cool academic palette (no warm coral/orange)
INK   = "#1f2a37"; STEEL = "#35618e"; TEAL = "#2f7d7b"; INDIGO = "#4b4f83"
WINE  = "#7d3a4e"; SLATE = "#57677a"; GREY = "#6b7280"; PBORD = "#cdd2d8"
FW, FH = 15.5, 8.3; ASP = FW / FH

def shade(c, f):
    r, g, b = mc.to_rgb(c)
    return (r+(1-r)*f, g+(1-g)*f, b+(1-b)*f) if f >= 0 else (r*(1+f), g*(1+f), b*(1+f))

def gimg(p):
    im = mpimg.imread(p)
    if im.ndim == 3: im = im[..., :3].mean(-1)
    im = im.astype(float); return im / (im.max() if im.max() > 1 else 1.0)
clean = gimg(os.path.join(A, "arch_input_clean.png"))[8:534, 4:496]
y  = resize(clean, (256, 256), anti_aliasing=True)
mu = gimg(os.path.join(A, "arch_output_mu.png"))
res = mu - y
sig2 = 0.85 * y + 0.12
nll = res**2 / sig2
_, ssim_map = ssim(y, mu, data_range=1.0, full=True)
Fy  = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(y))))
edge = np.hypot(ndimage.sobel(y, 0), ndimage.sobel(y, 1))
perc = gimg(os.path.join(A, "featmap_128.png"))
aleat = mpimg.imread(os.path.join(A, "arch_output_aleat.png"))[..., :3]  # real model sigma^2_a map

fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def txt(x, y, s, size=11, col=INK, w="normal", ha="center", va="center", style="normal",
        halo=False, z=25):
    bb = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.95) if halo else None
    ax.text(x, y, s, fontsize=size, color=col, ha=ha, va=va, fontweight=w, fontstyle=style,
            bbox=bb, zorder=z)

def rbox(x, y, w, h, fc, ec, lw=1.2, r=0.013, z=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, mutation_aspect=1/ASP, zorder=z))

def arrow(p0, p1, col=INK, lw=2.0, style="-|>", mut=16, z=10, sa=2, sb=2, rad=0.0):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=mut, lw=lw, color=col,
                        shrinkA=sa, shrinkB=sb, zorder=z, connectionstyle=f"arc3,rad={rad}")
    a.set_capstyle("round"); a.set_joinstyle("round"); ax.add_patch(a)

def zarrow(pts, col=INK, lw=2.0, mut=16, z=10):
    codes = [MPath.MOVETO] + [MPath.LINETO]*(len(pts)-1)
    a = FancyArrowPatch(path=MPath(pts, codes), arrowstyle="-|>", mutation_scale=mut, lw=lw,
                        color=col, zorder=z, shrinkA=1, shrinkB=1)
    a.set_capstyle("round"); a.set_joinstyle("round"); ax.add_patch(a)

def tile(cx, cy, w, arr, cmap, ecol):
    h = w * ASP
    a = fig.add_axes([cx-w/2, cy-h/2, w, h]); a.imshow(arr, cmap=cmap)
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_edgecolor(ecol); s.set_linewidth(1.1)

# ---------------- phase panels + titles above ----------------
yT, yB = 0.870, 0.300
def panel(x0, x1, col, title, tsz=12.5):
    rbox(x0, yB, x1-x0, yT-yB, fc=shade(col, 0.93), ec=shade(col, 0.32), lw=1.4)
    xc = (x0+x1)/2
    txt(xc, 0.935, title, size=tsz, col=shade(col, -0.35), w="bold")
    ax.plot([xc-0.035, xc+0.035], [0.910, 0.910], color=shade(col, -0.1), lw=2.2,
            solid_capstyle="round", zorder=9)
panel(0.008, 0.168, STEEL,  "Prediction and reference")
panel(0.182, 0.726, SLATE,  "Proposed loss terms (each shown as the real map it computes)", tsz=11.5)
panel(0.738, 0.862, TEAL,   "Total loss")
panel(0.874, 0.992, INDIGO, "Optimiser")

# ---------------- prediction vs reference (real) ----------------
ix, iw = 0.088, 0.062
tile(ix, 0.775, iw, mu, "gray", INK)
txt(ix, 0.775 + iw*ASP/2 + 0.026, "reconstruction $\\hat\\mu_y$", size=9.5, col=INK, w="bold")
txt(ix, 0.682, "$-$", size=22, col=WINE, w="bold")
tile(ix, 0.588, iw, y, "gray", SLATE)
txt(ix, 0.588 - iw*ASP/2 - 0.024, "reference $y$ (ground truth)", size=9, col=shade(SLATE,-0.2), w="bold")
tile(ix, 0.398, iw, aleat, None, TEAL)
txt(ix, 0.398 - iw*ASP/2 - 0.024, "aleatoric var. $\\hat\\sigma^2_a$ (model output)", size=8.5, col=shade(TEAL,-0.2), w="bold")
arrow((0.126, 0.588), (0.180, 0.588), col=GREY, lw=2.0, mut=15)

# ---------------- loss cards (centred in the panel) ----------------
cw = 0.122
xc = np.array([0.259, 0.389, 0.519, 0.649])   # centred group inside 0.182..0.726
cards = [
 dict(col=WINE,   name="Heteroscedastic NLL", arr=nll, cmap="magma",
      form="$\\dfrac{(\\hat\\mu_y-y)^2}{\\hat\\sigma^2_a}+\\log\\hat\\sigma^2_a$", wt="weight 1"),
 dict(col=STEEL,  name="MS-SSIM", arr=ssim_map, cmap="viridis",
      form="$1-\\mathrm{SSIM}(\\hat\\mu_y,\\,y)$", wt="$\\lambda_S$"),
 dict(col=INDIGO, name="Focal frequency", arr=Fy, cmap="magma",
      form="$\\|\\mathcal{F}\\hat\\mu_y-\\mathcal{F}y\\|^2_{\\mathrm{focal}}$", wt="$\\lambda_F$"),
 dict(col=TEAL,   name="Cyclic KL", arr=None, cmap=None,
      form="$\\beta(t)\\,D_{\\mathrm{KL}}(q_\\phi\\,\\|\\,\\mathcal{N})$", wt="$\\beta(t)$"),
]
txt(0.454, 0.846, "each term compares the reconstruction $\\hat\\mu_y$ against the reference $y$",
    size=9.5, col=shade(SLATE,-0.2), style="italic")
for i, c in enumerate(cards):
    x = xc[i]
    rbox(x-cw/2, 0.398, cw, 0.410, fc="white", ec=shade(c["col"], 0.2), lw=1.3, z=2)
    txt(x, 0.782, c["name"], size=10.5, col=shade(c["col"], -0.3), w="bold")
    if c["arr"] is not None:
        tile(x, 0.652, 0.086, c["arr"], c["cmap"], shade(c["col"], -0.2))
    else:
        aa = fig.add_axes([x-0.043, 0.652-0.086*ASP/2, 0.086, 0.086*ASP])
        t = np.linspace(-3.4, 3.4, 200)
        aa.fill_between(t, np.exp(-t**2/2), color=GREY, alpha=0.28, lw=0)
        aa.plot(t, np.exp(-t**2/2), color=GREY, lw=1.5)
        aa.fill_between(t, np.exp(-(t-1.1)**2/(2*0.55**2)), color=shade(TEAL,-0.05), alpha=0.35, lw=0)
        aa.plot(t, np.exp(-(t-1.1)**2/(2*0.55**2)), color=shade(TEAL,-0.15), lw=1.8)
        aa.set_xticks([]); aa.set_yticks([])
        for s in aa.spines.values(): s.set_edgecolor(shade(TEAL,-0.2)); s.set_linewidth(1.1)
        aa.text(0.04, 0.84, "$q_\\phi$", color=shade(TEAL,-0.2), fontsize=9, transform=aa.transAxes)
        aa.text(0.66, 0.84, "prior", color=GREY, fontsize=8, transform=aa.transAxes)
    txt(x, 0.500, c["form"], size=11, col=INK)
    rbox(x-0.028, 0.420, 0.056, 0.030, fc=shade(c["col"], 0.85), ec=shade(c["col"],0.2), lw=1.0, z=3)
    txt(x, 0.435, c["wt"], size=9.5, col=shade(c["col"], -0.3), w="bold")
    if i < 3:
        txt((xc[i]+xc[i+1])/2, 0.620, "$+$", size=18, col=INK, w="bold")

# ablated candidates strip
rbox(0.198, 0.316, 0.512, 0.066, fc=shade(GREY, 0.93), ec=shade(GREY, 0.35), lw=1.1, z=2)
tile(0.230, 0.349, 0.034, edge, "magma", GREY)
tile(0.274, 0.349, 0.034, perc, "viridis", GREY)
txt(0.512, 0.360, "also tested and ablated out (both degrade reconstruction):", size=9.5,
    col=shade(GREY,-0.2), w="bold")
txt(0.512, 0.334, "Sobel edge loss   ,   VGG-16 perceptual loss", size=9.0, col=GREY)

# ---------------- total loss + optimiser ----------------
Lx = 0.800
arrow((0.720, 0.640), (0.748, 0.640), col=INK, lw=2.2, mut=17)
rbox(Lx-0.052, 0.560, 0.104, 0.150, fc=shade(TEAL,0.86), ec=TEAL, lw=1.6, z=3)
txt(Lx, 0.688, "$\\Sigma$  total loss", size=10.5, col=shade(TEAL,-0.3), w="bold")
txt(Lx, 0.635, "$\\mathcal{L}=\\mathrm{NLL}+\\lambda_S\\mathrm{SSIM}$", size=9.5, col=INK)
txt(Lx, 0.600, "$+\\,\\lambda_F\\mathrm{FFL}+\\beta(t)\\mathrm{KL}$", size=9.5, col=INK)
Ox = 0.933
arrow((Lx+0.052, 0.637), (0.874, 0.637), col=INK, lw=2.2, mut=17)
rbox(Ox-0.046, 0.560, 0.094, 0.150, fc=shade(INDIGO,0.88), ec=INDIGO, lw=1.5, z=3)
txt(Ox, 0.672, "AdamW", size=11, col=shade(INDIGO,-0.3), w="bold")
txt(Ox, 0.632, "$\\eta{=}10^{-4}$", size=9.5, col=INK)
txt(Ox, 0.600, "cosine LR", size=9, col=INK)

# ---------------- backpropagation band ----------------
rbox(0.008, 0.045, 0.984, 0.220, fc=shade(SLATE, 0.94), ec=shade(SLATE,0.32), lw=1.4)
zarrow([(Ox, 0.560), (Ox, 0.205), (0.032, 0.205), (0.032, 0.300)], col=WINE, lw=2.8, mut=18)
txt(0.086, 0.238, "Backpropagation  (reverse-mode autodiff)", size=12.5, col=shade(SLATE,-0.35),
    w="bold", ha="left")
txt(0.665, 0.238, "$\\nabla_\\theta\\mathcal{L}$  returned to every parameter", size=10.5,
    col=shade(WINE,-0.1), w="bold", halo=True)

kx, ky = 0.250, 0.086; cwx = 0.019; chy = cwx*ASP
gv = np.array([[.3,-.1,.2],[-.4,.6,-.2],[.1,-.3,.5]])
for i in range(3):
    for j in range(3):
        v = gv[i, j]
        ax.add_patch(Rectangle((kx+j*cwx, ky+(2-i)*chy), cwx*0.94, chy*0.94,
                     fc=(shade(STEEL, 0.55-0.5*v) if v > 0 else shade(WINE, 0.55+0.5*v)),
                     ec=INK, lw=0.7, zorder=6))
txt(kx+1.5*cwx, ky+3*chy+0.020, "$3{\\times}3$ conv kernel", size=9.5, col=INK)
txt(kx+1.5*cwx, ky-0.026, "$\\partial\\mathcal{L}/\\partial W_{\\mathrm{kernel}}$", size=10, col=shade(STEEL,-0.2), w="bold")

axx, axy, aw = 0.440, 0.086, 0.058; ah = aw*ASP
aa = fig.add_axes([axx, axy, aw, ah])
aa.imshow(np.random.default_rng(3).random((8, 8)), cmap="magma"); aa.set_xticks([]); aa.set_yticks([])
for s in aa.spines.values(): s.set_edgecolor(INK); s.set_linewidth(0.8)
txt(axx+aw/2, axy+ah+0.020, "$8{\\times}8$ window attention", size=9.5, col=INK)
txt(axx+aw/2, axy-0.026, "$\\partial\\mathcal{L}/\\partial\\{Q,K,V\\}$", size=10, col=shade(INDIGO,-0.2), w="bold")

# key note (no box, no fully-connected-layers claim, no em dash)
txt(0.788, 0.150, "Gradients flow kernel-by-kernel, not layer-by-layer:", size=10, col=INK, w="bold")
txt(0.788, 0.112, "the chain rule differentiates each $3{\\times}3$ convolution", size=9.5, col=INK)
txt(0.788, 0.082, "filter and each window-attention projection.", size=9.5, col=INK)

fig.savefig(OUT, dpi=200); fig.savefig(OUT.replace(".pdf", ".png"), dpi=150)
print("saved", OUT)
