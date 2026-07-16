"""Extended re-eval: NAFNet + SCUNet + SharpXR at all 3 Foi presets with all 5 metrics
(PSNR, SSIM, MS-SSIM, FSIM, LPIPS), so the appendix full tables are consistent with the
unified main-text numbers. Same rng(i+42) seed, 256x256, [0,1] as the mid re-eval."""
import os, sys, glob, math, csv, numpy as np, torch
SCR = "/rds/user/stm43/hpc-work/ppvae_hformer/scripts"
RES = "/rds/user/stm43/hpc-work/ppvae_results"
_roi = open(f"{SCR}/generate_nafnet_scunet_roi.py").read().split("\n")
g = {"__name__": "sota_eval"}; exec(compile("\n".join(_roi[:246]), "roi", "exec"), g)
scunet, DEVICE = g["scunet"], g["DEVICE"]
_tn = open(f"{SCR}/train_nafnet.py").read().split("\n")
gn = {"__name__": "_tn", "__file__": f"{SCR}/train_nafnet.py"}
exec(compile("\n".join(_tn[:187]), "tn", "exec"), gn)
nafnet = gn["NAFNet"](in_ch=1, width=64, middle_blk_num=12, enc_blks=[2,2,4,8], dec_blks=[2,2,2,2])
_sd = torch.load(f"{RES}/baselines/nafnet/best_model.pth", map_location=DEVICE, weights_only=False)
nafnet.load_state_dict(_sd["model"] if (isinstance(_sd,dict) and "model" in _sd) else _sd); nafnet.to(DEVICE).eval()
_sx = open(f"{SCR}/sharpxr_baseline.py").read().split("\n")
gs = {"__name__": "_sx"}; exec(compile("\n".join(_sx[:78]), "sx", "exec"), gs)
sharpxr = gs["DualDecoderHybrid"]().to(DEVICE)
sharpxr.load_state_dict(torch.load(f"{RES}/sharpxr/best_model.pth", map_location=DEVICE)); sharpxr.eval()
print("loaded nafnet + scunet + sharpxr", flush=True)
import piq
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
DATA = "/rds/user/stm43/hpc-work/chest_xray/test"
PRESETS = [("low",0.01,0.002),("mid",0.03,0.005),("high",0.08,0.010)]
def load_img(p): return (np.array(Image.open(p).convert("L").resize((256,256)))/255.0).astype(np.float32)
def cls_of(p):
    if "/NORMAL/" in p.upper(): return "Normal"
    return "Bacterial" if os.path.basename(p).upper().startswith("BACTERIA") else "Viral"
files = sorted(glob.glob(f"{DATA}/NORMAL/*.jpeg")+glob.glob(f"{DATA}/NORMAL/*.png")) + \
        sorted(glob.glob(f"{DATA}/PNEUMONIA/*.jpeg")+glob.glob(f"{DATA}/PNEUMONIA/*.png"))
lpips_fn = piq.LPIPS(reduction="none").to(DEVICE); rows=[]
for lvl,A,B in PRESETS:
    for name, model in [("nafnet",nafnet),("scunet",scunet),("sharpxr",sharpxr)]:
        for i,p in enumerate(files):
            c=load_img(p); rng=np.random.default_rng(i+42)
            n=np.clip(c+rng.standard_normal(c.shape)*np.sqrt(A*np.clip(c,0,None)+B),0,1).astype(np.float32)
            x=torch.from_numpy(n)[None,None].to(DEVICE); y=torch.from_numpy(c)[None,None].to(DEVICE)
            with torch.no_grad(): o=model(x).clamp(0,1)
            oc=o.squeeze().cpu().numpy(); mse=float(np.mean((oc-c)**2))
            psnr=99.0 if mse<1e-12 else 20*math.log10(1/math.sqrt(mse))
            ss=piq.ssim(o,y,data_range=1.0).item()
            mss=piq.multi_scale_ssim(o,y,data_range=1.0).item()
            fs=piq.fsim(o,y,data_range=1.0,chromatic=False).item()
            lp=lpips_fn(o.repeat(1,3,1,1),y.repeat(1,3,1,1)).item()
            rows.append((name,lvl,cls_of(p),i,psnr,ss,mss,fs,lp))
        arr=np.array([[r[4],r[5],r[6],r[7],r[8]] for r in rows if r[0]==name and r[1]==lvl])
        print(f"{lvl:4s} {name:8s} PSNR {arr[:,0].mean():.3f} SSIM {arr[:,1].mean():.4f} MSSSIM {arr[:,2].mean():.4f} FSIM {arr[:,3].mean():.4f} LPIPS {arr[:,4].mean():.4f}", flush=True)
OUT=f"{RES}/evaluation/sota_extended_perimage.csv"
with open(OUT,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["model","noise_level","class","image","psnr","ssim","ms_ssim","fsim","lpips"]); w.writerows(rows)
print("wrote", OUT, flush=True)
