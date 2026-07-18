"""Rebuild Fig 4.7 (pairwise Cohen's d, 4 metrics) with a BINNED diverging palette
whose bands are the Methods' Cohen's-d categories (|d|<0.2 negligible, 0.2-0.5 small,
0.5-0.8 moderate, >=0.8 large), so a cell's colour == its effect-size category.
Validates the computed d against results/pairwise_stats_metrics.csv."""
import csv, math, numpy as np, sys
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

R = "results/"
PM = R + "per_image_metrics.csv"      # arms + KAIR: psnr, ssim, fsim
FS = R + "per_image_fsim.csv"          # arms + KAIR: fsim, lpips (piq scale)
SE = R + "sota_extended_perimage.csv"   # SOTA (NAFNet/SCUNet/SharpXR) all metrics

ORDER = ["arm_a_l2","arm_i_l1","arm_j_l1_ssim_ffl","arm_k_nll_l1","arm_b_nll","arm_c_nll_ssim",
         "arm_d_nll_ssim_ffl","arm_l_nll_edge_ffl","arm_n_perc","arm_m_full_det","arm_e_ppvae",
         "arm_f_kl_cyc","arm_g_kl_fb","arm_h_kl_cyc_fb","arm_p_best","arm_o_prelu",
         "dncnn_baseline","ffdnet","ircnn","drunet","swinir","nafnet","scunet","sharpxr"]
LAB = {"arm_a_l2":"A","arm_i_l1":"I","arm_j_l1_ssim_ffl":"J","arm_k_nll_l1":"K","arm_b_nll":"B",
       "arm_c_nll_ssim":"C","arm_d_nll_ssim_ffl":"D","arm_l_nll_edge_ffl":"L","arm_n_perc":"N",
       "arm_m_full_det":"M","arm_e_ppvae":"E","arm_f_kl_cyc":"F","arm_g_kl_fb":"G","arm_h_kl_cyc_fb":"H",
       "arm_p_best":"P","arm_o_prelu":"O","dncnn_baseline":"DnCNN","ffdnet":"FFDNet","ircnn":"IRCNN",
       "drunet":"DRUNet","swinir":"SwinIR","nafnet":"NAFNet","scunet":"SCUNet","sharpxr":"SharpXR"}
HIGHER = {"psnr":True,"ssim":True,"fsim":True,"lpips":False}   # lpips lower=better
ALPHA = 0.05/276

# --- load per-image metric series: data[metric][cond] = np.array over 624 imgs ---
data = {m:{} for m in HIGHER}
def add(cond, metric, vals): data[metric][cond]=np.array(vals,float)
# arms + KAIR from PM (psnr,ssim,fsim) and FS (fsim,lpips)
pm={};
for r in csv.DictReader(open(PM)):
    if r["noise_level"]!="mid": continue
    pm.setdefault(r["arm"],{}).setdefault("psnr",[]).append(float(r["psnr"]))
    pm[r["arm"]].setdefault("ssim",[]).append(float(r["ssim"]))
fs={}
for r in csv.DictReader(open(FS)):
    if r["noise_level"]!="mid": continue
    fs.setdefault(r["arm"],{}).setdefault("fsim",[]).append(float(r["fsim"]))
    fs[r["arm"]].setdefault("lpips",[]).append(float(r["lpips"]))
for cond in pm:
    if cond in LAB:
        add(cond,"psnr",pm[cond]["psnr"]); add(cond,"ssim",pm[cond]["ssim"])
        add(cond,"fsim",fs[cond]["fsim"]); add(cond,"lpips",fs[cond]["lpips"])
# SOTA from sota_extended (mid)
se={}
for r in csv.DictReader(open(SE)):
    if r.get("noise_level")!="mid": continue
    m=r["model"]
    for k in ("psnr","ssim","fsim","lpips"):
        se.setdefault(m,{}).setdefault(k,[]).append(float(r[k]))
for m in ("nafnet","scunet","sharpxr"):
    for k in HIGHER: add(m,k,se[m][k])

# --- pairwise pooled-SD Cohen's d (positive = row better) + Bonferroni sig ---
def cohen(metric,row,col):
    a=data[metric][row]; b=data[metric][col]
    n=len(a); sp=math.sqrt(((np.var(a,ddof=1)+np.var(b,ddof=1))/2))
    d=(a.mean()-b.mean())/sp
    if not HIGHER[metric]: d=-d                      # flip so + = better
    t,p=stats.ttest_ind(a,b,equal_var=True)
    return d, (p<ALPHA)

# --- validate ssim/fsim/lpips against the precomputed CSV ---
csvd={}
for r in csv.DictReader(open(R+"pairwise_stats_metrics.csv")):
    csvd[(r["metric"],r["arm1"],r["arm2"])]=float(r["cohens_d"])
errs=[]
for (met,a1,a2),dv in csvd.items():
    if a1 in data[met] and a2 in data[met]:
        mine,_=cohen(met,a1,a2)
        # CSV sign convention unknown -> compare |d|
        errs.append(abs(abs(mine)-abs(dv)))
print(f"VALIDATION vs CSV (ssim/fsim/lpips): n={len(errs)} mean|Δ|d|={np.mean(errs):.4f} max={np.max(errs):.4f}")

# --- build matrices ---
N=len(ORDER)
mats={m:np.full((N,N),np.nan) for m in HIGHER}
sigs={m:np.zeros((N,N),bool) for m in HIGHER}
for i,row in enumerate(ORDER):
    for j,col in enumerate(ORDER):
        if i>j:  # lower triangle
            d,sg=cohen_wrap=cohen(list(HIGHER)[0],row,col) if False else (None,None)
for met in HIGHER:
    for i,row in enumerate(ORDER):
        for j,col in enumerate(ORDER):
            if i>j:
                d,sg=cohen(met,row,col); mats[met][i,j]=d; sigs[met][i,j]=sg

# --- binned diverging palette at Cohen thresholds ---
bounds=[-3.001,-0.8,-0.5,-0.2,0.2,0.5,0.8,3.001]
# scico-vik-like: blue(neg)->white(0)->red(pos); 7 bins
colors=["#2166AC","#67A9CF","#D1E5F0","#F7F7F7","#FDDBC7","#EF8A62","#B2182B"]
cmap=ListedColormap(colors); norm=BoundaryNorm(bounds,cmap.N)

TITLES={"psnr":"PSNR","ssim":"SSIM","fsim":"FSIM","lpips":"LPIPS (lower better; sign-flipped)"}
fig,axes=plt.subplots(2,2,figsize=(16.5,15))
for ax,met in zip(axes.ravel(),["psnr","ssim","fsim","lpips"]):
    M=np.ma.masked_invalid(mats[met])
    ax.imshow(M,cmap=cmap,norm=norm,aspect="equal")
    for i in range(N):
        for j in range(N):
            if i>j and not sigs[met][i,j]:
                ax.plot(j,i,marker=".",color="black",ms=4)
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels([LAB[c] for c in ORDER],rotation=90,fontsize=6.5)
    ax.set_yticklabels([LAB[c] for c in ORDER],fontsize=6.5)
    # gridlines separating every cell (aid tracing a box to its row/column labels)
    ax.set_xticks(np.arange(-0.5,N,1),minor=True); ax.set_yticks(np.arange(-0.5,N,1),minor=True)
    ax.grid(which="minor",color="0.78",linewidth=0.5)
    ax.set_axisbelow(False)                 # draw grid above the cells
    ax.tick_params(which="minor",length=0)
    ax.tick_params(which="major",length=0)
    ax.set_title(TITLES[met],fontsize=12,fontweight="bold"); ax.set_xlim(-0.5,N-1.5); ax.set_ylim(N-0.5,0.5)
    for s in ax.spines.values(): s.set_visible(False)
# shared legend
import matplotlib.patches as mpatches
cats=["large (row worse)","moderate","small","negligible","small","moderate","large (row better)"]
handles=[mpatches.Patch(facecolor=colors[k],edgecolor="0.5",label=cats[k]) for k in range(7)]
fig.subplots_adjust(left=0.05,right=0.82,top=0.97,bottom=0.05,wspace=0.18,hspace=0.12)
fig.legend(handles=handles,title="Effect size (Cohen's $d$, row $-$ col)",loc="center left",
           bbox_to_anchor=(0.835,0.5),fontsize=9.5,title_fontsize=10,frameon=True)
OUT="figures/R/cohend_matrix_facets"
fig.savefig(OUT+".pdf",bbox_inches="tight"); fig.savefig(OUT+".png",dpi=130,bbox_inches="tight")
print("wrote",OUT)
