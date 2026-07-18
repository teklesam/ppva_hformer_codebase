"""Generate mu-hat reconstructions for the 6 gallery conditions missing from
figures_v2 (arms Q, R, S and SOTA baselines NAFNet, SCUNet, SharpXR), on the
EXACT same shared Bacterial case and noisy input the existing 21 panels used
(generate_figures_v2's Poisson noise, seed 42). Saves clean 256x256 grayscale
mu PNGs so the appendix gallery can show all conditions on one identical input.
"""
import os, sys, numpy as np, torch
SCR = "/rds/user/stm43/hpc-work/ppvae_hformer/scripts"
RES = "/rds/user/stm43/hpc-work/ppvae_results"
OUT = "/rds/user/stm43/hpc-work/ppvae_results/gallery_missing"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, SCR)
from PIL import Image

# --- reproduce the shared Bacterial case + noisy input (generate_figures_v2 logic) ---
import generate_figures_v2 as G
ALL = G.get_test_files()
pool = [(p, l) for p, l in ALL if l == "Bacterial"]
step = max(1, len(pool) // 4)
case_path = pool[step * 1][0]            # top row (ri=0) of the Bacterial panel
clean = G.load_img(case_path).astype(np.float32)
noisy = G.add_noise(clean, seed=42)       # Poisson(clean*200)/200 + N(0,0.01)  -- ri=0 -> seed 42
noisy_t = torch.tensor(noisy).float().unsqueeze(0)     # (1,256,256) as infer() expects
print("CASE:", os.path.basename(case_path), "| shared noisy input reproduced", flush=True)


def save_mu(name, mu):
    mu = np.clip(mu, 0, 1)
    Image.fromarray((mu * 255).round().astype(np.uint8)).save(f"{OUT}/mu_{name}.png")
    print("  wrote", name, "mu range", round(float(mu.min()), 3), round(float(mu.max()), 3), flush=True)

# also persist the exact shared input so the gallery can be verified against it
save_mu("_shared_noisy", noisy)
save_mu("_shared_clean", clean)

# --- Arms Q, R, S via the PP-VAE-Hformer loader/infer (same as the 21) ---
arm_models = G.load_models(["arm_q_charb", "arm_r_ft_j", "arm_s_ft_d"])
for arm, m in arm_models.items():
    recon, _, _ = G.infer(arm, m, noisy_t)
    save_mu(arm, recon)

# --- SOTA baselines: build via the authoritative eval prefixes ---
DEVICE = G.DEVICE
# SCUNet (from ROI-script prefix) + NAFNet (train class) + SharpXR (baseline class)
_roi = open(f"{SCR}/generate_nafnet_scunet_roi.py").read().split("\n")
g = {"__name__": "gal"}
exec(compile("\n".join(_roi[:246]), "roi", "exec"), g)
scunet = g["scunet"]; DEVICE = g["DEVICE"]

_tn = open(f"{SCR}/train_nafnet.py").read().split("\n")
gn = {"__name__": "_tn", "__file__": f"{SCR}/train_nafnet.py"}
exec(compile("\n".join(_tn[:187]), "tn", "exec"), gn)
nafnet = gn["NAFNet"](in_ch=1, width=64, middle_blk_num=12, enc_blks=[2, 2, 4, 8], dec_blks=[2, 2, 2, 2])
_sd = torch.load(f"{RES}/baselines/nafnet/best_model.pth", map_location=DEVICE, weights_only=False)
nafnet.load_state_dict(_sd["model"] if (isinstance(_sd, dict) and "model" in _sd) else _sd)
nafnet.to(DEVICE).eval()

_sx = open(f"{SCR}/sharpxr_baseline.py").read().split("\n")
gs = {"__name__": "_sx"}
exec(compile("\n".join(_sx[:78]), "sx", "exec"), gs)
sharpxr = gs["DualDecoderHybrid"]().to(DEVICE)
sharpxr.load_state_dict(torch.load(f"{RES}/sharpxr/best_model.pth", map_location=DEVICE))
sharpxr.eval()
print("loaded nafnet + scunet + sharpxr", flush=True)

x = noisy_t.unsqueeze(0).to(DEVICE)      # (1,1,256,256)
for name, model in [("nafnet", nafnet), ("scunet", scunet), ("sharpxr", sharpxr)]:
    with torch.no_grad():
        o = model(x)
        if isinstance(o, (tuple, list)):
            o = o[0]
        o = o.clamp(0, 1)
    save_mu(name, o.squeeze().cpu().numpy())
print("DONE", flush=True)
