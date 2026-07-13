# Bootstrap 95% CIs and paired statistical comparison for latent ARI / Silhouette (VAE arms).
# Input: per-arm npz (Z=[N,64], y=[N]) from dump_latent_embeddings.py.
# Paired bootstrap: the same resampled image indices are used for every arm on each replicate,
# so pairwise differences are tested on matched samples. Outputs a CSV, a comparison table, and a figure.
import argparse, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

def metrics(Z, y):
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Z)
    return adjusted_rand_score(y, km.labels_), (silhouette_score(Z, y) if len(set(y.tolist())) > 1 else 0.0)

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--B", type=int, default=2000)
ap.add_argument("--out", default="/tmp/latent_ci")
a = ap.parse_args()
order = ["arm_e_ppvae","arm_f_kl_cyc","arm_g_kl_fb","arm_h_kl_cyc_fb","arm_p_best"]
lab = {"arm_e_ppvae":"E","arm_f_kl_cyc":"F","arm_g_kl_fb":"G","arm_h_kl_cyc_fb":"H","arm_p_best":"P"}
data = {}
for arm in order:
    f = os.path.join(a.dir, arm + ".npz")
    if os.path.exists(f):
        d = np.load(f, allow_pickle=True); data[arm] = (d["Z"].astype(float), np.asarray(d["y"]))
arms = [x for x in order if x in data]
assert arms, "no embeddings found"
n = len(data[arms[0]][1])
point = {arm: metrics(*data[arm]) for arm in arms}
rng = np.random.default_rng(0)
boot = {arm: {"ari": [], "sil": []} for arm in arms}
for _ in range(a.B):
    idx = rng.integers(0, n, n)                     # shared across arms -> paired
    for arm in arms:
        Z, y = data[arm]; ari, sil = metrics(Z[idx], y[idx])
        boot[arm]["ari"].append(ari); boot[arm]["sil"].append(sil)
ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

os.makedirs(a.out, exist_ok=True)
import csv
with open(os.path.join(a.out, "latent_ci.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["arm","ARI","ARI_lo","ARI_hi","Sil","Sil_lo","Sil_hi"])
    print("Per-arm point [95% CI]:")
    for arm in arms:
        al, ah = ci(boot[arm]["ari"]); sl, sh = ci(boot[arm]["sil"])
        w.writerow([lab[arm], f"{point[arm][0]:.3f}", f"{al:.3f}", f"{ah:.3f}", f"{point[arm][1]:.3f}", f"{sl:.3f}", f"{sh:.3f}"])
        print(f"  {lab[arm]}: ARI {point[arm][0]:.3f} [{al:.3f}, {ah:.3f}]   Sil {point[arm][1]:.3f} [{sl:.3f}, {sh:.3f}]")

print("\nPaired ARI difference F - X (95% CI; excludes 0 => significant):")
best = "arm_f_kl_cyc"
for arm in arms:
    if arm == best: continue
    for met in ("ari", "sil"):
        diff = np.array(boot[best][met]) - np.array(boot[arm][met])
        lo, hi = ci(diff); sig = "significant" if (lo > 0 or hi < 0) else "n.s."
        print(f"  F - {lab[arm]}  {met.upper():4s}: {np.mean(diff):+.3f} [{lo:+.3f}, {hi:+.3f}]  {sig}")

# figure: point + 95% CI per arm, ARI and Silhouette
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
for ax, met, ttl in zip(axes, ("ari", "sil"), ("Adjusted Rand Index", "Silhouette")):
    ys = np.arange(len(arms))[::-1]
    pts = [point[arm][0 if met == "ari" else 1] for arm in arms]
    los = [ci(boot[arm][met])[0] for arm in arms]; his = [ci(boot[arm][met])[1] for arm in arms]
    ax.errorbar(pts, ys, xerr=[np.array(pts)-np.array(los), np.array(his)-np.array(pts)],
                fmt="o", color="#4c72b0", ecolor="#4c72b0", capsize=3, ms=5)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(ys); ax.set_yticklabels([lab[arm] for arm in arms]); ax.set_title(ttl, fontsize=10)
    ax.set_xlabel("score (95% bootstrap CI)")
fig.suptitle("")  # caption carries the title (house rule)
fig.tight_layout()
fig.savefig(os.path.join(a.out, "latent_ci.pdf")); fig.savefig(os.path.join(a.out, "latent_ci.png"), dpi=150)
print("\nwrote", a.out)
