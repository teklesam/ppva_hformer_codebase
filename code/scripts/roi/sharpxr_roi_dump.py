"""Append the SharpXR reconstruction to the arms' ROI npz so the main ROI montage
includes it alongside NAFNet + SCUNet. It denoises the identical noisy image the
arms saw (read from the npz). Idempotent: skips if SharpXR is already present."""
import sys, math, numpy as np, torch
SCR = "/rds/user/stm43/hpc-work/ppvae_hformer/scripts"; sys.path.insert(0, SCR)
RES = "/rds/user/stm43/hpc-work/ppvae_results"; NPZ = f"{RES}/roi_dump_normal9.npz"

# Build SharpXR's DualDecoderHybrid by exec'ing only the class defs (before training code)
_sx = open(f"{SCR}/sharpxr_baseline.py").read().split("\n")
g = {"__name__": "_sx"}; exec(compile("\n".join(_sx[:78]), "sx", "exec"), g)
DualDecoderHybrid, DEVICE = g["DualDecoderHybrid"], g["DEVICE"]

model = DualDecoderHybrid().to(DEVICE)
model.load_state_dict(torch.load(f"{RES}/sharpxr/best_model.pth", map_location=DEVICE))
model.eval()

d = dict(np.load(NPZ, allow_pickle=True))
labels = [str(x) for x in d["labels"]]
if "SharpXR" in labels:
    print("SharpXR already in npz; nothing to do."); sys.exit(0)

clean, noisy = d["clean"], d["noisy"]
x = torch.from_numpy(noisy.astype(np.float32))[None, None].to(DEVICE)
def psnr(a, b):
    m = float(np.mean((a - b) ** 2)); return 99.0 if m < 1e-12 else 20 * math.log10(1 / math.sqrt(m))
with torch.no_grad():
    sx = model(x).clamp(0, 1).squeeze().cpu().numpy().astype(np.float32)
print(f"SharpXR PSNR={psnr(clean, sx):.2f}  (test-set mean was 33.56)", flush=True)

d["recons"] = np.concatenate([d["recons"], sx[None]], 0)
d["labels"] = np.append(d["labels"], ["SharpXR"])
if "keys" in d:  d["keys"]  = np.append(d["keys"],  ["sharpxr"])
if "psnrs" in d: d["psnrs"] = np.append(d["psnrs"], [psnr(clean, sx)])
np.savez_compressed(NPZ, **d)
print("appended SharpXR; recons now", d["recons"].shape[0], "labels", list(map(str, d["labels"]))[-4:], flush=True)
