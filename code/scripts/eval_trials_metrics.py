"""Evaluate each arm-D Optuna trial checkpoint on the full test set with
PSNR + SSIM + FSIM + LPIPS. PSNR is confounded with the (SSIM/FFL) objective;
FSIM/SSIM/LPIPS give a less objective-aligned view of how loss-weighting shifts
perceptual quality. Writes per-image rows for R."""
import os, sys, glob, math, csv, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import piq
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from src.models.ppvae_hformer import PPVAEHformer
from src.training.config import ABLATION_ARMS, ExperimentConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = "/rds/user/stm43/hpc-work/chest_xray"
SWEEP = "/rds/user/stm43/hpc-work/ppvae_results/optuna_sweep_d"
OUT = os.path.join(SWEEP, "trials_perimage_metrics.csv")
MID = (0.03, 0.005)

def load_trial_params():
    p = os.path.join(SWEEP, "trials.csv"); m = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            try: n = int(row["number"])
            except: continue
            m[n] = (float(row["params_lambda_ssim"]), float(row["params_lambda_ffl"]))
    return m

def build_arm_d():
    cfg = ExperimentConfig()
    for k,v in ABLATION_ARMS["arm_d_nll_ssim_ffl"].get("model",{}).items(): setattr(cfg.model,k,v)
    return PPVAEHformer(in_channels=cfg.model.in_channels, base_channels=cfg.model.base_channels,
        num_blocks=cfg.model.num_blocks, num_scales=cfg.model.num_scales, num_heads=cfg.model.num_heads,
        window_size=cfg.model.window_size, use_vae=cfg.model.use_vae,
        activation=getattr(cfg.model,"activation","gelu"))

def load_img(p): return (np.array(Image.open(p).convert("L").resize((256,256)))/255.0).astype(np.float32)
def add_noise(c,a,b,rng): return np.clip(c+rng.standard_normal(c.shape)*np.sqrt(a*np.clip(c,0,None)+b),0,1).astype(np.float32)
def cls_of(p):
    if "/NORMAL/" in p.upper(): return "Normal"
    return "Bacterial" if os.path.basename(p).upper().startswith("BACTERIA") else "Viral"

files = sorted(glob.glob(f"{DATA}/test/NORMAL/*.jpeg")+glob.glob(f"{DATA}/test/NORMAL/*.png")) + \
        sorted(glob.glob(f"{DATA}/test/PNEUMONIA/*.jpeg")+glob.glob(f"{DATA}/test/PNEUMONIA/*.png"))
print(f"test images: {len(files)}  device={DEVICE}", flush=True)
params = load_trial_params()
trial_dirs = sorted(glob.glob(os.path.join(SWEEP, "trial_*")))
lpips_fn = piq.LPIPS(reduction="none").to(DEVICE)

# precompute clean + noisy tensors once (shared across trials)
cleans, noisys, clss = [], [], []
for i,p in enumerate(files):
    c = load_img(p); n = add_noise(c, MID[0], MID[1], np.random.default_rng(i+42))
    cleans.append(c); noisys.append(n); clss.append(cls_of(p))

rows = []
for td in trial_dirs:
    tn = int(os.path.basename(td).split("_")[1]); ck = os.path.join(td, "best_model.pth")
    if tn not in params or not os.path.exists(ck): continue
    ss, ff = params[tn]
    m = build_arm_d(); st = torch.load(ck, map_location=DEVICE, weights_only=False)
    m.load_state_dict(st["model"] if isinstance(st,dict) and "model" in st else st); m.to(DEVICE).eval()
    with torch.no_grad():
        for i in range(len(files)):
            c, n = cleans[i], noisys[i]
            x = torch.from_numpy(n)[None,None].to(DEVICE)
            mu,_,_,_ = m(x, deterministic=True); o = mu.clamp(0,1)
            y = torch.from_numpy(c)[None,None].to(DEVICE)
            oc = o.squeeze().cpu().numpy(); mse = np.mean((oc-c)**2)
            psnr = 99.0 if mse<1e-12 else 20*math.log10(1/math.sqrt(mse))
            sv = piq.ssim(o,y,data_range=1.0).item(); fv = piq.fsim(o,y,data_range=1.0,chromatic=False).item()
            lv = lpips_fn(o.repeat(1,3,1,1), y.repeat(1,3,1,1)).item()
            rows.append((tn, ss, ff, i, clss[i], psnr, sv, fv, lv))
    print(f"trial {tn:2d} ssim={ss:.3f} ffl={ff:.3f} done", flush=True)
with open(OUT,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["trial","lambda_ssim","lambda_ffl","image","class","psnr","ssim","fsim","lpips"]); w.writerows(rows)
print("wrote", OUT, len(rows), "rows", flush=True)
