"""Regenerate Table E.1 (tab:results:recon:mid:full) so every metric column shows
mean +/- SD [95% CI], consistently, for all 28 models. Preserves each row's
Arm / Loss-Architecture / VAE columns and its existing medal (\\first/\\second/\\third)
by parsing them out of the current row and only re-formatting the four metric cells.

Data: per_image_metrics.csv (arms+KAIR), sota_extended_perimage.csv (NAFNet/SCUNet/
SharpXR), bm3d_results.csv (BM3D summary; MS-SSIM absent -> '--'). CI = bootstrap for
per-image models, normal approximation (mean +/- 1.96 SD/sqrt(n)) for BM3D.
"""
import re, csv, statistics as st, random, sys
random.seed(42)
RES = "results"
N = 624

# ---- load per-image data ----
data = {}
def add(fname, keyfield):
    for r in csv.DictReader(open(f"{RES}/{fname}")):
        if r.get("noise_level", "mid") != "mid":
            continue
        d = data.setdefault(r[keyfield], {})
        for m in ("psnr", "ssim", "ms_ssim", "fsim"):
            if r.get(m, "") not in ("", "nan"):
                d.setdefault(m, []).append(float(r[m]))
add("per_image_metrics.csv", "arm")
add("sota_extended_perimage.csv", "model")
bm = next(csv.DictReader(open(f"{RES}/bm3d_results.csv")))
BM3D = {"psnr": (float(bm["psnr_mean"]), float(bm["psnr_sd"])),
        "ssim": (float(bm["ssim_mean"]), float(bm["ssim_sd"])),
        "fsim": (float(bm["fsim_mean"]), float(bm["fsim_sd"]))}

def boot_ci(v, B=2000):
    ms = sorted(st.mean(random.choices(v, k=len(v))) for _ in range(B))
    return ms[int(.025 * B)], ms[int(.975 * B)]

def stats(key, metric):
    if key == "bm3d":
        if metric not in BM3D:
            return None
        m, sd = BM3D[metric]; half = 1.96 * sd / (N ** 0.5); return m, sd, m - half, m + half
    d = data.get(key, {})
    if metric not in d or not d[metric]:
        return None
    v = d[metric]; m = st.mean(v); sd = st.pstdev(v); lo, hi = boot_ci(v); return m, sd, lo, hi

LAB2KEY = {"A":"arm_a_l2","I":"arm_i_l1","J":"arm_j_l1_ssim_ffl","B":"arm_b_nll","D":"arm_d_nll_ssim_ffl",
    "K":"arm_k_nll_l1","L":"arm_l_nll_edge_ffl","N":"arm_n_perc","C":"arm_c_nll_ssim","M":"arm_m_full_det",
    "H":"arm_h_kl_cyc_fb","F":"arm_f_kl_cyc","G":"arm_g_kl_fb","P":"arm_p_best","E":"arm_e_ppvae",
    "O":"arm_o_prelu","Q":"arm_q_charb","R":"arm_r_ft_j","S":"arm_s_ft_d","BM3D":"bm3d",
    "DnCNN":"dncnn_baseline","IRCNN":"ircnn","FFDNet":"ffdnet","DRUNet":"drunet","SwinIR":"swinir",
    "NAFNet":"nafnet","SCUNet":"scunet","SharpXR":"sharpxr"}
METRICS = ["psnr", "ssim", "ms_ssim", "fsim"]

def fmt_cell(orig_cell, key, metric):
    """Re-format one metric cell, preserving its medal macro if any."""
    medal = None
    mm = re.search(r'\\(first|second|third)\{', orig_cell)
    if mm:
        medal = mm.group(1)
    s = stats(key, metric)
    if s is None:
        return " -- "
    m, sd, lo, hi = s
    if metric == "psnr":
        body = f"{m:.3f} \\pm {sd:.3f}"; ci = f"[{lo:.2f},\\,{hi:.2f}]"
    else:
        body = f"{m:.4f} \\pm {sd:.4f}"; ci = f"[{lo:.4f},\\,{hi:.4f}]"
    inner = f"\\{medal}{{{body}}}" if medal else body
    return f" ${inner}$ {{\\scriptsize {ci}}} "

tex_path = "../dissertation-latex/dissertation-latex/main_submission.tex"
lines = open(tex_path).read().split("\n")
# locate the target longtable by its label
start = next(i for i, l in enumerate(lines) if "\\label{tab:results:recon:mid:full}" in l)
body_lo = next(i for i in range(start, len(lines)) if "\\endlastfoot" in lines[i])
body_hi = next(i for i in range(body_lo, len(lines)) if "\\end{longtable}" in lines[i])

changed = 0
for i in range(body_lo + 1, body_hi):
    ln = lines[i]
    if "\\multicolumn" in ln or ln.strip() in ("\\midrule", "\\bottomrule", "") or "\\cmidrule" in ln:
        continue
    if "&" not in ln or "\\\\" not in ln:
        continue
    cells = ln.split("&")
    if len(cells) != 7:
        continue
    label = cells[0].strip()
    key = LAB2KEY.get(label)
    if not key:
        continue
    tail = cells[6].split("\\\\")[1] if "\\\\" in cells[6] else ""
    cells[3] = fmt_cell(cells[3], key, "psnr")
    cells[4] = fmt_cell(cells[4], key, "ssim")
    cells[5] = fmt_cell(cells[5], key, "ms_ssim")
    cells[6] = fmt_cell(cells[6], key, "fsim") + "\\\\" + tail
    lines[i] = "&".join(cells)
    changed += 1

open(tex_path, "w").write("\n".join(lines))
print(f"reformatted {changed} data rows in Table E.1")
