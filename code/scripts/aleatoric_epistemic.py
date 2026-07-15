"""Do the NLL-head aleatoric map and the VAE Monte-Carlo epistemic map flag the same lung
regions or different ones? Compute both for a VAE arm (H) on a pathology case, then their
spatial correlation and top-decile overlap."""
import os, sys, numpy as np, torch
SCR="/rds/user/stm43/hpc-work/ppvae_hformer/scripts"; sys.path.insert(0, SCR)
import generate_roi_panels_v2 as G
from src.models.ppvae_hformer import PPVAEHformer
from src.training.config import ABLATION_ARMS, ExperimentConfig
RES="/rds/user/stm43/hpc-work/ppvae_results"
def build(arm):
    cfg=ExperimentConfig()
    for k,v in ABLATION_ARMS[arm].get("model",{}).items(): setattr(cfg.model,k,v)
    return PPVAEHformer(in_channels=cfg.model.in_channels, base_channels=cfg.model.base_channels,
        num_blocks=cfg.model.num_blocks, num_scales=cfg.model.num_scales, num_heads=cfg.model.num_heads,
        window_size=cfg.model.window_size, use_vae=cfg.model.use_vae, activation=getattr(cfg.model,"activation","gelu"))
m=build("arm_h_kl_cyc_fb"); m.load_state_dict(G._load_ckpt(f"{RES}/arm_h_kl_cyc_fb/best_model.pth")); m.to(G.DEVICE).eval()
clean=G.load_image("PNEUMONIA", 10); noisy=G.add_noise(clean, 200, seed=42)
x=torch.from_numpy(noisy)[None,None].float().to(G.DEVICE)
with torch.no_grad():
    mu,lsa,zmu,zlv = m(x, deterministic=True)
sig_a = torch.exp(0.5*lsa).squeeze().cpu().numpy()
K=20; recs=[]
with torch.no_grad():
    for _ in range(K):
        mk = m(x, deterministic=False)[0]; recs.append(mk.squeeze().cpu().numpy())
sig_e = np.stack(recs).std(0)
a=sig_a.ravel(); e=sig_e.ravel()
r=np.corrcoef(a,e)[0,1]
ta,te=np.percentile(a,90),np.percentile(e,90); ma,me=a>=ta,e>=te
iou=(ma&me).sum()/max(1,(ma|me).sum())
print(f"aleatoric mean={a.mean():.4f}  epistemic mean={e.mean():.4f}", flush=True)
print(f"Pearson(sig_a, sig_e) = {r:.3f}", flush=True)
print(f"top-10% high-uncertainty overlap (IoU) = {iou:.3f}", flush=True)
# where each dominates: fraction of pixels aleatoric>epistemic (after z-scoring each)
az=(sig_a-a.mean())/a.std(); ez=(sig_e-e.mean())/e.std()
print(f"pixels where aleatoric dominates: {(az>ez).mean()*100:.0f}%  epistemic dominates: {(ez>az).mean()*100:.0f}%", flush=True)
np.savez(f"{RES}/aleatoric_epistemic_H.npz", clean=clean, noisy=noisy,
         mu=mu.squeeze().cpu().numpy(), sig_a=sig_a, sig_e=sig_e)
print("saved maps", flush=True)
