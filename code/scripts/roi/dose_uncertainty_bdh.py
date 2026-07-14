# Dose-uncertainty comparison for Arms B, D, H: noisy inputs + sigma_a maps at low/mid/high noise, one CXR, with a shared ROI.
import os, sys, glob, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
PROJECT="/rds/user/stm43/hpc-work/ppvae_hformer"; os.chdir(PROJECT); sys.path.insert(0,PROJECT)
from src.models.ppvae_hformer import PPVAEHformer
from src.training.config import ExperimentConfig, ABLATION_ARMS
RESULTS="/rds/user/stm43/hpc-work/ppvae_results"; DATA="/rds/user/stm43/hpc-work/chest_xray/test"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRESETS=[("Low",0.01,0.002),("Mid",0.03,0.005),("High",0.08,0.010)]
ARMS=[("arm_b_nll","B (NLL)"),("arm_d_nll_ssim_ffl","D (NLL+SSIM+FFL)"),("arm_h_kl_cyc_fb","H (PP-VAE)")]
ROI=(92,104,64,64)   # perihilar vessels

def load_img(p): return np.array(Image.open(p).convert("L").resize((256,256)))/255.0
def add_noise(c,a,b,seed=7):
    rng=np.random.default_rng(seed); return np.clip(c+rng.standard_normal(c.shape)*np.sqrt(a*np.clip(c,0,None)+b),0,1).astype(np.float32)
def psnr(a,b):
    m=np.mean((a-b)**2); return 99.0 if m<1e-12 else 20*np.log10(1.0/np.sqrt(m))
def load_model(arm):
    cfg=ExperimentConfig()
    for k,v in ABLATION_ARMS[arm].get("model",{}).items(): setattr(cfg.model,k,v)
    m=PPVAEHformer(in_channels=cfg.model.in_channels,base_channels=cfg.model.base_channels,num_blocks=cfg.model.num_blocks,
      num_scales=cfg.model.num_scales,num_heads=cfg.model.num_heads,window_size=cfg.model.window_size,
      use_vae=cfg.model.use_vae,activation=getattr(cfg.model,"activation","gelu")).to(DEVICE)
    ck=torch.load(os.path.join(RESULTS,arm,"best_model.pth"),map_location=DEVICE,weights_only=False)
    m.load_state_dict(ck["model"] if isinstance(ck,dict) and "model" in ck else ck); return m.eval()
def box(ax,colr="cyan"):
    x,y,w,h=ROI; ax.add_patch(Rectangle((x,y),w,h,ec=colr,fc="none",lw=1.4))
def bare(ax): ax.set_xticks([]); ax.set_yticks([])

clean=load_img(sorted(glob.glob(os.path.join(DATA,"NORMAL","*")))[9]).astype(np.float32)
noisy={nl:add_noise(clean,a,b) for nl,a,b in PRESETS}

nrow=1+len(ARMS); ncol=1+len(PRESETS)
fig,axes=plt.subplots(nrow,ncol,figsize=(2.35*ncol,2.35*nrow)); vmax=0.22

# -- top row: inputs (clean + noisy at each dose) --
axes[0,0].imshow(clean,cmap="gray",vmin=0,vmax=1); box(axes[0,0],"red"); bare(axes[0,0])
axes[0,0].set_title("clean reference",fontsize=9); axes[0,0].set_ylabel("Inputs",fontsize=10,fontweight="bold")
for c,(nl,a,b) in enumerate(PRESETS):
    ax=axes[0,c+1]; ax.imshow(noisy[nl],cmap="gray",vmin=0,vmax=1); box(ax,"red"); bare(ax)
    ax.set_title(f"{nl} noise",fontsize=9)
    ax.text(0.03,0.05,f"noisy {psnr(noisy[nl],clean):.1f} dB",transform=ax.transAxes,color="w",fontsize=7,
            bbox=dict(boxstyle="round,pad=0.15",fc="black",alpha=0.6))

# -- arm rows: reconstruction (col0) + sigma_a maps at each dose --
im=None
for r,(arm,albl) in enumerate(ARMS):
    m=load_model(arm)
    with torch.no_grad():
        mu_mid,_,_,_=m(torch.tensor(noisy["Mid"])[None,None].to(DEVICE),deterministic=True)
    ax0=axes[r+1,0]; ax0.imshow(mu_mid.squeeze().cpu().numpy(),cmap="gray",vmin=0,vmax=1); box(ax0,"red"); bare(ax0)
    ax0.set_ylabel(albl,fontsize=9,fontweight="bold")
    if r==0: ax0.set_title("$\\hat\\mu$ (Mid)",fontsize=9)
    for c,(nl,a,b) in enumerate(PRESETS):
        with torch.no_grad():
            _,lsa,_,_=m(torch.tensor(noisy[nl])[None,None].to(DEVICE),deterministic=True)
        sig=np.exp(0.5*lsa.squeeze().cpu().numpy())
        ax=axes[r+1,c+1]; im=ax.imshow(sig,cmap="inferno",vmin=0,vmax=vmax); box(ax,"cyan"); bare(ax)
        ax.text(0.03,0.05,f"$\\bar\\sigma_a$={sig.mean():.3f}",transform=ax.transAxes,color="w",fontsize=7,
                bbox=dict(boxstyle="round,pad=0.15",fc="black",alpha=0.6))
    del m; torch.cuda.empty_cache()
fig.colorbar(im,ax=axes,fraction=0.02,pad=0.01,label="aleatoric $\\hat\\sigma_a$")
out=os.path.join(RESULTS,"roi_panels_v2","fig_dose_uncertainty_bdh")
fig.savefig(out+".png",dpi=200,bbox_inches="tight"); fig.savefig(out+".pdf",bbox_inches="tight")
print("wrote",out+".pdf",flush=True)
