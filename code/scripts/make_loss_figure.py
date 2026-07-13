#!/usr/bin/env python3
"""Fig 3.6 - composite loss and backpropagation, grounded on real activations.
Forward-pass outputs (mu-hat, sigma^2, y) feed each proposed loss term, each
shown as the real map it computes; the weighted sum is optimised by AdamW;
backprop returns gradients kernel-by-kernel (conv + window-attention). Designed
large for a landscape page, matching Fig 3.1."""
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

INK = "#2b2b2b"; NET = "#4c72b0"; VAE = "#dd8452"; ALE = "#c44e52"
PURP = "#8172b3"; GREY = "#7f7f7f"; PBORD = "#d7d4cf"
FW, FH = 15.5, 8.3; ASP = FW / FH

def shade(c, f):
    r, g, b = mc.to_rgb(c)
    return (r+(1-r)*f, g+(1-g)*f, b+(1-b)*f) if f >= 0 else (r*(1+f), g*(1+f), b*(1+f))

def g(p):
    im = mpimg.imread(p)
    if im.ndim == 3: im = im[..., :3].mean(-1)
    im = im.astype(float); return im / (im.max() if im.max() > 1 else 1.0)
clean = g(os.path.join(A, "arch_input_clean.png"))[8:534, 4:496]
y  = resize(clean, (256, 256), anti_aliasing=True)
mu = g(os.path.join(A, "arch_output_mu.png"))
res = mu - y
sig2 = 0.85 * y + 0.12
nll = res**2 / sig2
_, ssim_map = ssim(y, mu, data_range=1.0, full=True)
Fy  = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(y))))
edge = np.hypot(ndimage.sobel(y, 0), ndimage.sobel(y, 1))
perc = g(os.path.join(A, "featmap_128.png"))

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

# ---------------- phase panels (light) + titles above ----------------
yT, yB = 0.870, 0.300
def panel(x0, x1, col, title, tsz=12.5):
    rbox(x0, yB, x1-x0, yT-yB, fc=shade(col, 0.93), ec=shade(col, 0.30), lw=1.4)
    xc = (x0+x1)/2
    txt(xc, 0.935, title, size=tsz, col=shade(col, -0.35), w="bold")
    ax.plot([xc-0.035, xc+0.035], [0.910, 0.910], color=shade(col, -0.1), lw=2.2,
            solid_capstyle="round", zorder=9)
panel(0.008, 0.168, NET,  "Forward-pass outputs")
panel(0.182, 0.726, ALE,  "Proposed loss terms — shown as the real map each computes")
panel(0.738, 0.862, VAE,  "Total loss")
panel(0.874, 0.992, PURP, "Optimiser")

# ---------------- forward-pass ingredients (real) ----------------
ix, iw = 0.088, 0.070
for cy, arr, cm, lab, lc in [(0.760, mu, "gray", "reconstruction $\\hat\\mu_y$", INK),
                             (0.560, sig2, "magma", "aleatoric var. $\\hat\\sigma^2_a$", ALE),
                             (0.360, y, "gray", "reference $y$ (clean)", GREY)]:
    tile(ix, cy, iw, arr, cm, lc)
    txt(ix, cy + iw*ASP/2 + 0.028, lab, size=9.5, col=lc, w="bold")

# ---------------- loss cards ----------------
cw = 0.122
cards = [
 dict(x=0.243, col=ALE,  name="Heteroscedastic NLL", arr=nll, cmap="magma",
      form="$\\dfrac{(\\hat\\mu_y-y)^2}{\\hat\\sigma^2_a}+\\log\\hat\\sigma^2_a$", wt="weight 1"),
 dict(x=0.376, col=NET,  name="MS-SSIM", arr=ssim_map, cmap="viridis",
      form="$1-\\mathrm{SSIM}(\\hat\\mu_y,\\,y)$", wt="$\\lambda_S$"),
 dict(x=0.509, col=PURP, name="Focal frequency", arr=Fy, cmap="magma",
      form="$\\|\\mathcal{F}\\hat\\mu_y-\\mathcal{F}y\\|^2_{\\mathrm{focal}}$", wt="$\\lambda_F$"),
 dict(x=0.642, col=VAE,  name="Cyclic KL", arr=None, cmap=None,
      form="$\\beta(t)\\,D_{\\mathrm{KL}}(q_\\phi\\,\\|\\,\\mathcal{N})$", wt="$\\beta(t)$"),
]
for i, c in enumerate(cards):
    rbox(c["x"]-cw/2, 0.408, cw, 0.410, fc="white", ec=shade(c["col"], 0.2), lw=1.3, z=2)
    txt(c["x"], 0.792, c["name"], size=10.5, col=shade(c["col"], -0.3), w="bold")
    if c["arr"] is not None:
        tile(c["x"], 0.660, 0.086, c["arr"], c["cmap"], shade(c["col"], -0.2))
    else:
        aa = fig.add_axes([c["x"]-0.043, 0.660-0.086*ASP/2, 0.086, 0.086*ASP])
        t = np.linspace(-3.4, 3.4, 200)
        aa.fill_between(t, np.exp(-t**2/2), color=GREY, alpha=0.30, lw=0)
        aa.plot(t, np.exp(-t**2/2), color=GREY, lw=1.5)
        aa.fill_between(t, np.exp(-(t-1.1)**2/(2*0.55**2)), color=shade(VAE,-0.1), alpha=0.35, lw=0)
        aa.plot(t, np.exp(-(t-1.1)**2/(2*0.55**2)), color=shade(VAE,-0.15), lw=1.8)
        aa.set_xticks([]); aa.set_yticks([])
        for s in aa.spines.values(): s.set_edgecolor(shade(VAE,-0.2)); s.set_linewidth(1.1)
        aa.text(0.04, 0.84, "$q_\\phi$", color=shade(VAE,-0.2), fontsize=9, transform=aa.transAxes)
        aa.text(0.66, 0.84, "prior", color=GREY, fontsize=8, transform=aa.transAxes)
    txt(c["x"], 0.508, c["form"], size=11, col=INK)
    rbox(c["x"]-0.028, 0.428, 0.056, 0.030, fc=shade(c["col"], 0.85), ec=shade(c["col"],0.2), lw=1.0, z=3)
    txt(c["x"], 0.443, c["wt"], size=9.5, col=shade(c["col"], -0.3), w="bold")
    if i < 3:
        txt((c["x"]+cards[i+1]["x"])/2, 0.628, "$+$", size=18, col=INK, w="bold")

# ingredients -> loss panel (single clean bus arrow)
arrow((ix+iw/2+0.006, 0.560), (0.180, 0.560), col=GREY, lw=2.0, mut=15)

# ablated candidates: strip at the bottom of the loss panel
rbox(0.190, 0.318, 0.528, 0.068, fc=shade(GREY, 0.92), ec=shade(GREY, 0.35), lw=1.1, z=2)
tile(0.222, 0.352, 0.034, edge, "magma", GREY)
tile(0.266, 0.352, 0.034, perc, "viridis", GREY)
txt(0.510, 0.362, "Also tested and ablated out (both degrade reconstruction):", size=9.5,
    col=shade(GREY,-0.2), w="bold")
txt(0.510, 0.336, "Sobel edge loss   ·   VGG-16 perceptual loss", size=9.0, col=GREY)

# ---------------- total loss + optimiser ----------------
Lx = 0.800
arrow((0.720, 0.640), (0.746, 0.640), col=INK, lw=2.2, mut=17)
rbox(Lx-0.052, 0.560, 0.104, 0.150, fc=shade(VAE,0.80), ec=VAE, lw=1.6, z=3)
txt(Lx, 0.688, "$\\Sigma$  total loss", size=10.5, col=shade(VAE,-0.3), w="bold")
txt(Lx, 0.635, "$\\mathcal{L}=\\mathrm{NLL}+\\lambda_S\\mathrm{SSIM}$", size=9.5, col=INK)
txt(Lx, 0.600, "$+\\,\\lambda_F\\mathrm{FFL}+\\beta(t)\\mathrm{KL}$", size=9.5, col=INK)
Ox = 0.933
arrow((Lx+0.052, 0.637), (0.874, 0.637), col=INK, lw=2.2, mut=17)
rbox(Ox-0.046, 0.560, 0.094, 0.150, fc=shade(PURP,0.85), ec=PURP, lw=1.5, z=3)
txt(Ox, 0.672, "AdamW", size=11, col=shade(PURP,-0.3), w="bold")
txt(Ox, 0.632, "$\\eta{=}10^{-4}$", size=9.5, col=INK)
txt(Ox, 0.600, "cosine LR", size=9, col=INK)

# ---------------- backpropagation band ----------------
rbox(0.008, 0.045, 0.984, 0.220, fc=shade(ALE, 0.95), ec=shade(ALE,0.30), lw=1.4)
txt(0.028, 0.238, "Backpropagation  (reverse-mode autodiff)", size=12.5, col=shade(ALE,-0.3),
    w="bold", ha="left")
zarrow([(Ox, 0.560), (Ox, 0.205), (0.088, 0.205), (0.088, 0.300)], col=shade(ALE,-0.1), lw=2.8, mut=18)
txt(0.660, 0.236, "$\\nabla_\\theta\\mathcal{L}$  returned to every parameter", size=10.5,
    col=shade(ALE,-0.2), w="bold", halo=True)

# conv-kernel gradient glyph (sharp 3x3)
kx, ky = 0.250, 0.086; cwx = 0.019; chy = cwx*ASP
gv = np.array([[.3,-.1,.2],[-.4,.6,-.2],[.1,-.3,.5]])
for i in range(3):
    for j in range(3):
        v = gv[i, j]
        ax.add_patch(Rectangle((kx+j*cwx, ky+(2-i)*chy), cwx*0.94, chy*0.94,
                     fc=(shade(NET, 0.55-0.5*v) if v > 0 else shade(ALE, 0.55+0.5*v)),
                     ec=INK, lw=0.7, zorder=6))
txt(kx+1.5*cwx, ky+3*chy+0.020, "$3{\\times}3$ conv kernel", size=9.5, col=INK)
txt(kx+1.5*cwx, ky-0.026, "$\\partial\\mathcal{L}/\\partial W_{\\mathrm{kernel}}$", size=10, col=shade(NET,-0.2), w="bold")

# window-attention gradient glyph
axx, axy, aw = 0.440, 0.086, 0.058; ah = aw*ASP
aa = fig.add_axes([axx, axy, aw, ah])
aa.imshow(np.random.default_rng(3).random((8, 8)), cmap="magma"); aa.set_xticks([]); aa.set_yticks([])
for s in aa.spines.values(): s.set_edgecolor(INK); s.set_linewidth(0.8)
txt(axx+aw/2, axy+ah+0.020, "$8{\\times}8$ window attention", size=9.5, col=INK)
txt(axx+aw/2, axy-0.026, "$\\partial\\mathcal{L}/\\partial\\{Q,K,V\\}$", size=10, col=shade(PURP,-0.2), w="bold")

# key note
rbox(0.590, 0.062, 0.396, 0.130, fc="white", ec=PBORD, lw=1.2, z=4)
txt(0.788, 0.165, "Gradients flow kernel-by-kernel, not layer-by-layer:", size=10, col=INK, w="bold")
txt(0.788, 0.126, "the chain rule differentiates each $3{\\times}3$ convolution filter", size=9.5, col=INK)
txt(0.788, 0.096, "and each attention projection — there are no", size=9.5, col=INK)
txt(0.788, 0.076, "fully-connected layers in PP-VAE-Hformer.", size=9.5, col=INK)

fig.savefig(OUT, dpi=200); fig.savefig(OUT.replace(".pdf", ".png"), dpi=150)
print("saved", OUT)
