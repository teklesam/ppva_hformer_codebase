"""Append NAFNet + SCUNet reconstructions to the arms' ROI npz so the main ROI montage
includes them. They denoise the identical noisy image the arms saw (from the npz)."""
import os, sys, numpy as np, torch, math
SCR = "/rds/user/stm43/hpc-work/ppvae_hformer/scripts"; sys.path.insert(0, SCR)
RES = "/rds/user/stm43/hpc-work/ppvae_results"; NPZ = f"{RES}/roi_dump_normal9.npz"
# SCUNet + DEVICE via ROI prefix (verified correct); NAFNet via train_nafnet class (correct 34.4)
_roi = open(f"{SCR}/generate_nafnet_scunet_roi.py").read().split("\n")
g={"__name__":"sd"}; exec(compile("\n".join(_roi[:246]),"roi","exec"), g)
scunet, DEVICE = g["scunet"], g["DEVICE"]
_tn = open(f"{SCR}/train_nafnet.py").read().split("\n")
gn={"__name__":"_tn","__file__":f"{SCR}/train_nafnet.py"}; exec(compile("\n".join(_tn[:187]),"tn","exec"), gn)
nafnet = gn["NAFNet"](in_ch=1, width=64, middle_blk_num=12, enc_blks=[2,2,4,8], dec_blks=[2,2,2,2])
sd = torch.load(f"{RES}/baselines/nafnet/best_model.pth", map_location=DEVICE, weights_only=False)
nafnet.load_state_dict(sd["model"] if isinstance(sd,dict) and "model" in sd else sd); nafnet.to(DEVICE).eval()
d = dict(np.load(NPZ, allow_pickle=True))
clean, noisy = d["clean"], d["noisy"]
x = torch.from_numpy(noisy.astype(np.float32))[None,None].to(DEVICE)
def psnr(a,b):
    m=float(np.mean((a-b)**2)); return 99.0 if m<1e-12 else 20*math.log10(1/math.sqrt(m))
with torch.no_grad():
    naf = nafnet(x).clamp(0,1).squeeze().cpu().numpy().astype(np.float32)
    scu = scunet(x).clamp(0,1).squeeze().cpu().numpy().astype(np.float32)
print(f"NAFNet PSNR={psnr(clean,naf):.2f}  SCUNet PSNR={psnr(clean,scu):.2f}  (sanity: ~34)", flush=True)
d["recons"] = np.concatenate([d["recons"], naf[None], scu[None]], 0)
d["labels"] = np.append(d["labels"], ["NAFNet","SCUNet"])
if "keys" in d:  d["keys"]  = np.append(d["keys"],  ["nafnet","scunet"])
if "psnrs" in d: d["psnrs"] = np.append(d["psnrs"], [psnr(clean,naf), psnr(clean,scu)])
np.savez_compressed(NPZ, **d)
print("appended NAFNet+SCUNet; recons now", d["recons"].shape[0], "labels", list(map(str,d["labels"]))[-4:], flush=True)
