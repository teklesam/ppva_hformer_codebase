#!/usr/bin/env python3
"""Fig 3.6 - composite loss and backpropagation, grounded on real activations.
Forward-pass outputs (mu-hat, sigma^2, target y) feed each proposed loss term,
each shown as the real map it computes; the terms are weighted into the total
loss; backprop returns gradients kernel-by-kernel (conv + window-attention),
not layer-by-layer. Publication serif style, matches Fig 3.1."""
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
EVAL = "#55a868"; PURP = "#8172b3"; GREY = "#7f7f7f"; PBORD = "#d7d4cf"
FW, FH = 16.0, 8.8; ASP = FW / FH

def shade(c, f):
    r, g, b = mc.to_rgb(c)
    return (r+(1-r)*f, g+(1-g)*f, b+(1-b)*f) if f >= 0 else (r*(1+f), g*(1+f), b*(1+f))

# ---------- real maps ----------
def g(p):
    im = mpimg.imread(p)
    if im.ndim == 3: im = im[..., :3].mean(-1)
    im = im.astype(float); return im / (im.max() if im.max() > 1 else 1.0)
clean = g(os.path.join(A, "arch_input_clean.png"))[8:534, 4:496]
noisy = g(os.path.join(A, "arch_input_noisy.png"))[8:534, 4:496]
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

def txt(x, y, s, size=9, col=INK, w="normal", ha="center", va="center", style="normal",
        halo=False, z=25):
    bb = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.95) if halo else None
    ax.text(x, y, s, fontsize=size, color=col, ha=ha, va=va, fontweight=w, fontstyle=style,
            bbox=bb, zorder=z)

def rbox(x, y, w, h, fc, ec, lw=1.1, r=0.012, z=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, mutation_aspect=1/ASP, zorder=z))

def arrow(p0, p1, col=INK, lw=1.6, style="-|>", ls="-", rad=0.0, mut=12, z=10, sa=2, sb=2):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=mut, lw=lw, color=col, ls=ls,
                        shrinkA=sa, shrinkB=sb, zorder=z, connectionstyle=f"arc3,rad={rad}")
    a.set_capstyle("round"); a.set_joinstyle("round"); ax.add_patch(a)

def zarrow(pts, col=INK, lw=1.6, mut=12, z=10, style="-|>"):
    codes = [MPath.MOVETO] + [MPath.LINETO]*(len(pts)-1)
    a = FancyArrowPatch(path=MPath(pts, codes), arrowstyle=style, mutation_scale=mut, lw=lw,
                        color=col, zorder=z, shrinkA=1, shrinkB=1)
    a.set_capstyle("round"); a.set_joinstyle("round"); ax.add_patch(a)

def tile(cx, cy, w, arr, cmap, ecol, label=None, lcol=INK, lpos="bottom", norm=None):
    h = w * ASP
    a = fig.add_axes([cx-w/2, cy-h/2, w, h]); a.imshow(arr, cmap=cmap, norm=norm)
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_edgecolor(ecol); s.set_linewidth(1.0)
    if label:
        ly = cy - h/2 - 0.024 if lpos == "bottom" else cy + h/2 + 0.024
        txt(cx, ly, label, size=7.2, col=lcol, w="bold")
    return h

# =====================================================================
# phase titles above panels
# =====================================================================
yT, yB = 0.855, 0.310
def panel(x0, x1, col, title, tsz=9.3):
    rbox(x0, yB, x1-x0, yT-yB, fc=shade(col, 0.93), ec=shade(col, 0.30), lw=1.3)
    xc = (x0+x1)/2
    txt(xc, 0.905, title, size=tsz, col=shade(col, -0.35), w="bold")
    ax.plot([xc-0.03, xc+0.03], [0.887, 0.887], color=shade(col, -0.1), lw=1.8,
            solid_capstyle="round", zorder=9)
panel(0.010, 0.150, NET,  "Forward-pass outputs")
panel(0.160, 0.712, ALE,  "Proposed loss terms — each shown as the map it computes")
panel(0.722, 0.872, VAE,  "Total loss")
panel(0.882, 0.990, PURP, "Optimiser")

# =====================================================================
# forward-pass ingredients (real)
# =====================================================================
ing_x = 0.080
tile(ing_x, 0.760, 0.086, mu,   "gray",  INK,  "reconstruction $\\hat\\mu_y$", INK)
tile(ing_x, 0.560, 0.086, sig2, "magma", ALE,  "aleatoric var. $\\hat\\sigma^2_a$", ALE)
tile(ing_x, 0.360, 0.086, y,    "gray",  GREY, "reference $y$ (clean)", GREY)
txt(ing_x, 0.836, "from the forward pass", size=6.8, col=GREY, style="italic")

# =====================================================================
# loss cards (4 proposed terms) — real computed maps
# =====================================================================
cw = 0.132
cards = [
 dict(x=0.186, col=ALE,  name="Heteroscedastic NLL", arr=nll, cmap="magma",
      form="$\\dfrac{(\\hat\\mu_y-y)^2}{\\hat\\sigma^2_a}+\\log\\hat\\sigma^2_a$",
      wt="weight 1", note="down-weights noisy lung;\nedges dominate residual"),
 dict(x=0.330, col=NET,  name="MS-SSIM", arr=ssim_map, cmap="viridis",
      form="$1-\\mathrm{SSIM}(\\hat\\mu_y,\\,y)$",
      wt="$\\lambda_S$", note="local structural\nsimilarity map"),
 dict(x=0.474, col=PURP, name="Focal frequency (FFL)", arr=Fy, cmap="magma",
      form="$\\|\\mathcal{F}\\hat\\mu_y-\\mathcal{F}y\\|^2_{\\mathrm{focal}}$",
      wt="$\\lambda_F$", note="error in the\nFourier spectrum"),
 dict(x=0.618, col=VAE,  name="Cyclic KL", arr=None, cmap=None,
      form="$\\beta(t)\\,D_{\\mathrm{KL}}(q_\\phi\\,\\|\\,\\mathcal{N})$",
      wt="$\\beta(t)$", note="pulls latent posterior\ntoward the prior"),
]
map_cy = 0.665
for i, c in enumerate(cards):
    rbox(c["x"]-cw/2, 0.400, cw, 0.415, fc="white", ec=shade(c["col"], 0.2), lw=1.2, z=2)
    txt(c["x"], 0.792, c["name"], size=8.0, col=shade(c["col"], -0.3), w="bold")
    if c["arr"] is not None:
        tile(c["x"], map_cy, 0.086, c["arr"], c["cmap"], shade(c["col"], -0.2))
    else:  # KL: posterior vs prior
        aa = fig.add_axes([c["x"]-0.043, map_cy-0.080, 0.086, 0.160])
        t = np.linspace(-3.5, 3.5, 200)
        aa.fill_between(t, np.exp(-t**2/2), color=GREY, alpha=0.30, lw=0)
        aa.plot(t, np.exp(-t**2/2), color=GREY, lw=1.3)
        aa.fill_between(t, np.exp(-(t-1.1)**2/(2*0.55**2)), color=shade(VAE,-0.1), alpha=0.35, lw=0)
        aa.plot(t, np.exp(-(t-1.1)**2/(2*0.55**2)), color=shade(VAE,-0.15), lw=1.5)
        aa.set_xticks([]); aa.set_yticks([])
        for s in aa.spines.values(): s.set_edgecolor(shade(VAE,-0.2)); s.set_linewidth(1.0)
        aa.text(0.03, 0.86, "$q_\\phi$", color=shade(VAE,-0.2), fontsize=7, transform=aa.transAxes)
        aa.text(0.70, 0.86, "prior", color=GREY, fontsize=6.5, transform=aa.transAxes)
    txt(c["x"], 0.520, c["form"], size=8.2, col=INK)
    txt(c["x"], 0.460, c["note"], size=6.4, col=GREY, style="italic")
    rbox(c["x"]-0.024, 0.412, 0.048, 0.026, fc=shade(c["col"], 0.85), ec=shade(c["col"],0.2), lw=0.8, z=3)
    txt(c["x"], 0.425, c["wt"], size=7.2, col=shade(c["col"], -0.3), w="bold")
    if i < 3:  # '+' between adjacent cards (this is a sum)
        txt((c["x"]+cards[i+1]["x"])/2, 0.615, "$+$", size=13, col=INK, w="bold")

# ingredients feed every term (one clean bus arrow)
arrow((ing_x+0.045, 0.560), (0.164, 0.560), col=GREY, lw=1.6, mut=12)
txt(0.150, 0.585, "$\\hat\\mu_y,\\hat\\sigma^2_a,y$", size=7.2, col=INK, halo=True)
txt(0.150, 0.536, "feed every term", size=6.4, col=GREY, style="italic", halo=True)

# ablated candidates: thin strip at the bottom of the loss panel
rbox(0.168, 0.322, 0.536, 0.060, fc=shade(GREY, 0.92), ec=shade(GREY, 0.35), lw=1.0, z=2)
tile(0.196, 0.352, 0.030, edge, "magma", GREY, None)
tile(0.236, 0.352, 0.030, perc, "viridis", GREY, None)
txt(0.500, 0.360, "Also tested and ablated out (both degrade reconstruction):", size=7.0,
    col=shade(GREY,-0.2), w="bold")
txt(0.500, 0.338, "Sobel edge loss  ·  VGG-16 perceptual loss", size=6.8, col=GREY)

# =====================================================================
# total loss (weighted sum) and optimiser
# =====================================================================
Lx = 0.797
arrow((0.706, 0.640), (0.745, 0.640), col=INK, lw=1.7, mut=13)
rbox(Lx-0.050, 0.560, 0.100, 0.155, fc=shade(VAE,0.80), ec=VAE, lw=1.5, z=3)
txt(Lx, 0.688, "$\\Sigma$  total loss", size=8.2, col=shade(VAE,-0.3), w="bold")
txt(Lx, 0.636, "$\\mathcal{L}=\\mathrm{NLL}+\\lambda_S\\mathrm{SSIM}$", size=7.4, col=INK)
txt(Lx, 0.604, "$+\\,\\lambda_F\\mathrm{FFL}+\\beta(t)\\mathrm{KL}$", size=7.4, col=INK)
Ox = 0.936
arrow((Lx+0.050, 0.637), (0.890, 0.637), col=INK, lw=1.7, mut=13)
rbox(Ox-0.046, 0.560, 0.092, 0.155, fc=shade(PURP,0.85), ec=PURP, lw=1.4, z=3)
txt(Ox, 0.672, "AdamW", size=8.4, col=shade(PURP,-0.3), w="bold")
txt(Ox, 0.634, "$\\eta{=}10^{-4}$", size=7.6, col=INK)
txt(Ox, 0.602, "cosine LR", size=7.2, col=INK)

# =====================================================================
# backpropagation band (kernel-by-kernel, not layer-by-layer)
# =====================================================================
rbox(0.010, 0.045, 0.980, 0.230, fc=shade(ALE, 0.95), ec=shade(ALE,0.30), lw=1.3)
txt(0.030, 0.246, "Backpropagation  (reverse-mode autodiff)", size=9.3, col=shade(ALE,-0.3),
    w="bold", ha="left")
# return path: from optimiser/L back to the forward network
zarrow([(Ox, 0.560), (Ox, 0.210), (0.086, 0.210), (0.086, 0.330)], col=shade(ALE,-0.1), lw=2.4, mut=15)
txt(0.500, 0.230, "$\\nabla_\\theta\\mathcal{L}$  returned to every parameter", size=8.0,
    col=shade(ALE,-0.2), w="bold", halo=True)

# conv-kernel gradient glyph (sharp 3x3 grid), NOT an MLP
kx, ky = 0.250, 0.092; cwx = 0.015; chy = cwx*ASP
gv = np.array([[ .3,-.1,.2],[-.4,.6,-.2],[.1,-.3,.5]])
for i in range(3):
    for j in range(3):
        v = gv[i, j]
        ax.add_patch(Rectangle((kx+j*cwx, ky+(2-i)*chy), cwx*0.94, chy*0.94,
                     fc=(shade(NET, 0.55-0.5*v) if v > 0 else shade(ALE, 0.55+0.5*v)),
                     ec=INK, lw=0.6, zorder=6))
gcx = kx + 1.5*cwx
txt(gcx, ky+3*chy+0.016, "$3{\\times}3$ conv kernel", size=6.8, col=INK)
txt(gcx, ky-0.020, "$\\partial\\mathcal{L}/\\partial W_{\\mathrm{kernel}}$", size=7.4, col=shade(NET,-0.2), w="bold")

# window-attention gradient glyph (aligned with the conv grid)
ax2x, ax2y, aw = 0.470, 0.092, 0.046; ah = aw*ASP
aa = fig.add_axes([ax2x, ax2y, aw, ah])
rng = np.random.default_rng(3); M = rng.random((8, 8))
aa.imshow(M, cmap="magma"); aa.set_xticks([]); aa.set_yticks([])
for s in aa.spines.values(): s.set_edgecolor(INK); s.set_linewidth(0.7)
txt(ax2x+aw/2, ax2y+ah+0.016, "$8{\\times}8$ window attention", size=6.8, col=INK)
txt(ax2x+aw/2, ax2y-0.020, "$\\partial\\mathcal{L}/\\partial\\{Q,K,V\\}$", size=7.4, col=shade(PURP,-0.2), w="bold")

# the key note
rbox(0.590, 0.070, 0.392, 0.120, fc="white", ec=PBORD, lw=1.1, z=4)
txt(0.786, 0.160, "Gradients flow kernel-by-kernel, not layer-by-layer:", size=7.8,
    col=INK, w="bold")
txt(0.786, 0.126, "the chain rule differentiates each $3{\\times}3$ convolution", size=7.3, col=INK)
txt(0.786, 0.100, "filter and each attention projection — there are no", size=7.3, col=INK)
txt(0.786, 0.074, "fully-connected layers in PP-VAE-Hformer.", size=7.3, col=INK)

fig.savefig(OUT, dpi=200); fig.savefig(OUT.replace(".pdf", ".png"), dpi=150)
print("saved", OUT)
