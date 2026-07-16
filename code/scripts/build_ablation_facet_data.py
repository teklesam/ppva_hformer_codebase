"""Assemble the per-image data for Figure 4.6 (all conditions at a glance) with all
27 conditions (19 arms + KAIR x5 + NAFNet + SCUNet + SharpXR) on CONSISTENT scales:

  psnr, ssim  <- per_image_metrics.csv     (arms + KAIR)
  fsim, lpips <- per_image_fsim.csv         (arms + KAIR; piq/VGG-16, as Table 4.2)
  all four    <- sota_unified_perimage.csv  (NAFNet, SCUNet) / sharpxr_perimage.csv (SharpXR)

Writes results/ablation_facet_data.csv with columns arm,noise_level,img_idx,psnr,ssim,fsim,lpips.
LPIPS is piq throughout (avoids the AlexNet-scale mix in per_image_metrics.csv).
"""
import csv, os
RES = os.path.join(os.path.dirname(__file__), "..", "..", "results")

def load(path):
    return list(csv.DictReader(open(os.path.join(RES, path))))

# arms + KAIR: join per_image_metrics (psnr,ssim) with per_image_fsim (fsim,lpips) on (arm,img_idx)
pm = {(r["arm"], r["img_idx"]): r for r in load("per_image_metrics.csv") if r["noise_level"] == "mid"}
pf = {(r["arm"], r["img_idx"]): r for r in load("per_image_fsim.csv") if r["noise_level"] == "mid"}
rows = []
for k, m in pm.items():
    f = pf.get(k)
    if not f:
        continue
    rows.append({"arm": m["arm"], "noise_level": "mid", "img_idx": m["img_idx"],
                 "psnr": m["psnr"], "ssim": m["ssim"], "fsim": f["fsim"], "lpips": f["lpips"]})

# new baselines: all four metrics from their own piq CSVs
for path, col in [("sota_unified_perimage.csv", None), ("sharpxr_perimage.csv", None)]:
    for i, r in enumerate(load(path)):
        name = r["model"].lower()
        rows.append({"arm": name, "noise_level": "mid", "img_idx": str(i),
                     "psnr": r["psnr"], "ssim": r["ssim"], "fsim": r["fsim"], "lpips": r["lpips"]})

OUT = os.path.join(RES, "ablation_facet_data.csv")
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["arm", "noise_level", "img_idx", "psnr", "ssim", "fsim", "lpips"])
    w.writeheader(); w.writerows(rows)
conds = sorted(set(r["arm"] for r in rows))
print(f"wrote {OUT}: {len(rows)} rows, {len(conds)} conditions")
print(conds)
