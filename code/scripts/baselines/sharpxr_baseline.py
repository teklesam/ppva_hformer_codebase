#!/usr/bin/env python
"""Train + evaluate SharpXR (Abolade et al. 2025) as a within-study baseline:
our Kermany split, our Foi Poisson-Gaussian noise, SharpXR's own architecture
(DualDecoderHybrid, verbatim from github.com/ileri-oluwa-kiiye/SharpXR) and its
RMSE loss. Evaluated on the same 624-image test set as every other baseline.
"""
import os, sys, glob, math, csv, random, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # one Kermany JPEG is truncated
import piq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = "/rds/user/stm43/hpc-work/chest_xray"
RESULTS = "/rds/user/stm43/hpc-work/ppvae_results/sharpxr"
os.makedirs(RESULTS, exist_ok=True)
PRESETS = [("low", 0.01, 0.002), ("mid", 0.03, 0.005), ("high", 0.08, 0.010)]
MID = (0.03, 0.005)

# ---------------------------------------------------------------- SharpXR model (verbatim)
class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(True),
                                  nn.Conv2d(o, o, 3, padding=1), nn.ReLU(True))
    def forward(self, x): return self.conv(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_list, out_list):
        super().__init__()
        self.upconvs = nn.ModuleList(); self.decoders = nn.ModuleList()
        for ic, oc in zip(in_list, out_list):
            self.upconvs.append(nn.ConvTranspose2d(ic, oc, 2, stride=2))
            self.decoders.append(DoubleConv(oc*2, oc))
    def _lap(self, x):
        k = torch.tensor([[[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]]], dtype=torch.float32, device=x.device)
        k = k.expand(x.size(1), 1, 3, 3)
        return F.conv2d(x, k, padding=1, groups=x.size(1))
    def forward(self, x, skips, laplacian=False):
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x); skip = skips[i]
            if laplacian: skip = skip + self._lap(skip)
            if x.shape != skip.shape: x = TF.resize(x, skip.shape[2:])
            x = torch.cat((skip, x), dim=1); x = self.decoders[i](x)
        return x

class LearnableFusion(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn = nn.Sequential(nn.Conv2d(c*2, c, 3, padding=1), nn.ReLU(True),
                                  nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True),
                                  nn.Conv2d(c, 2, 1), nn.Softmax(dim=1))
    def forward(self, a, b):
        w = self.attn(torch.cat([a, b], dim=1))
        return w[:, 0:1]*a + w[:, 1:2]*b

class DualDecoderHybrid(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = nn.ModuleList(); ic = in_channels
        for f in features: self.encoder.append(DoubleConv(ic, f)); ic = f
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        in_list = [features[-1]*2] + list(reversed(features))[:-1]
        out_list = list(reversed(features))
        self.decoder_denoise = DecoderBlock(in_list, out_list)
        self.decoder_edge = DecoderBlock(in_list, out_list)
        self.learnable_fusion = LearnableFusion(features[0])
        self.final_out = nn.Conv2d(features[0], out_channels, 1)
    def forward(self, x):
        skips = []
        for layer in self.encoder:
            x = layer(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x); skips = skips[::-1]
        d = self.decoder_denoise(x, skips); e = self.decoder_edge(x, skips, laplacian=True)
        return self.final_out(self.learnable_fusion(d, e))

# ---------------------------------------------------------------- data + Foi noise (ours)
def load_img(p): return (np.array(Image.open(p).convert("L").resize((256,256)))/255.0).astype(np.float32)
def add_noise(c, a, b, rng): return np.clip(c + rng.standard_normal(c.shape)*np.sqrt(a*np.clip(c,0,None)+b), 0, 1).astype(np.float32)
def list_split(split):
    fs = []
    for cl in ("NORMAL","PNEUMONIA"):
        fs += sorted(glob.glob(f"{DATA}/{split}/{cl}/*.jpeg")+glob.glob(f"{DATA}/{split}/{cl}/*.png"))
    return fs
def rmse(pred, tgt): return torch.sqrt(F.mse_loss(pred, tgt))

def train():
    files = list_split("train"); random.Random(0).shuffle(files)
    nval = max(1, int(0.1*len(files))); val_files, tr_files = files[:nval], files[nval:]
    print(f"train {len(tr_files)} val {len(val_files)}", flush=True)
    model = DualDecoderHybrid().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    clean_val = [load_img(p) for p in val_files]
    best, bad, BATCH, EPOCHS = -1, 0, 8, 90
    for ep in range(EPOCHS):
        model.train(); random.shuffle(tr_files); rng = np.random.default_rng(ep); tot = 0.0; nb = 0
        for i in range(0, len(tr_files), BATCH):
            batch = tr_files[i:i+BATCH]; cs, ns = [], []
            for p in batch:
                c = load_img(p); _, a, b = random.choice(PRESETS)
                cs.append(c); ns.append(add_noise(c, a, b, rng))
            x = torch.tensor(np.stack(ns))[:, None].to(DEVICE); y = torch.tensor(np.stack(cs))[:, None].to(DEVICE)
            opt.zero_grad(); out = model(x).clamp(0,1); loss = rmse(out, y); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        # val PSNR at mid noise
        model.eval(); vr = np.random.default_rng(123); ps = []
        with torch.no_grad():
            for c in clean_val:
                n = add_noise(c, MID[0], MID[1], vr)
                o = model(torch.tensor(n)[None,None].to(DEVICE)).clamp(0,1).squeeze().cpu().numpy()
                mse = np.mean((o-c)**2); ps.append(99.0 if mse<1e-12 else 20*math.log10(1/math.sqrt(mse)))
        vpsnr = float(np.mean(ps))
        print(f"ep {ep+1}/{EPOCHS} rmse {tot/nb:.4f} val_psnr {vpsnr:.3f}", flush=True)
        if vpsnr > best: best = vpsnr; bad = 0; torch.save(model.state_dict(), f"{RESULTS}/best_model.pth")
        else:
            bad += 1
            if bad >= 12: print("early stop", flush=True); break
    print(f"best val_psnr {best:.3f}", flush=True)

def evaluate():
    model = DualDecoderHybrid().to(DEVICE)
    model.load_state_dict(torch.load(f"{RESULTS}/best_model.pth", map_location=DEVICE)); model.eval()
    lpips_fn = piq.LPIPS(reduction="none").to(DEVICE)
    files = list_split("test"); rng = np.random.default_rng(42); rows = []
    def cls_of(p):
        if "/NORMAL/" in p.upper(): return "Normal"
        return "Bacterial" if os.path.basename(p).upper().startswith("BACTERIA") else "Viral"
    for i, p in enumerate(files):
        c = load_img(p); n = add_noise(c, MID[0], MID[1], np.random.default_rng(i+42))
        with torch.no_grad():
            o = model(torch.tensor(n)[None,None].to(DEVICE)).clamp(0,1)
        y = torch.tensor(c)[None,None].to(DEVICE)
        oc = o.squeeze().cpu().numpy(); mse = np.mean((oc-c)**2)
        psnr = 99.0 if mse<1e-12 else 20*math.log10(1/math.sqrt(mse))
        ss = piq.ssim(o, y, data_range=1.0).item(); fs = piq.fsim(o, y, data_range=1.0, chromatic=False).item()
        lp = lpips_fn(o.repeat(1,3,1,1), y.repeat(1,3,1,1)).item()
        rows.append(("SharpXR", cls_of(p), psnr, ss, fs, lp))
        if i % 100 == 0: print(f"  eval {i}/{len(files)}", flush=True)
    out = f"{RESULTS}/sharpxr_perimage.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model","class","psnr","ssim","fsim","lpips"]); w.writerows(rows)
    arr = np.array([[r[2],r[3],r[4],r[5]] for r in rows])
    print(f"SharpXR overall: PSNR {arr[:,0].mean():.3f} SSIM {arr[:,1].mean():.4f} FSIM {arr[:,2].mean():.4f} LPIPS {arr[:,3].mean():.4f}", flush=True)
    print("wrote", out, flush=True)

if __name__ == "__main__":
    if "--eval-only" not in sys.argv: train()
    evaluate()
