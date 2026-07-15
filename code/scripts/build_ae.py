import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=np.load("/tmp/ae_H.npz")
mu, sa, se = d["mu"], d["sig_a"], d["sig_e"]
r=np.corrcoef(sa.ravel(), se.ravel())[0,1]
fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
ax[0].imshow(mu, cmap="gray"); ax[0].set_title("Reconstruction $\\hat\\mu$ (Arm H)", fontsize=10.5)
im1=ax[1].imshow(sa, cmap="inferno"); ax[1].set_title("Aleatoric $\\hat\\sigma_a$ (NLL head)", fontsize=10.5)
plt.colorbar(im1, ax=ax[1], fraction=0.046)
im2=ax[2].imshow(se, cmap="inferno"); ax[2].set_title("Epistemic $\\hat\\sigma_e$ (VAE, K=20)", fontsize=10.5)
plt.colorbar(im2, ax=ax[2], fraction=0.046)
for a in ax[:3]: a.set_xticks([]); a.set_yticks([])
# scatter: per-pixel aleatoric vs epistemic (subsample), z-scored
az=(sa-sa.mean())/sa.std(); ez=(se-se.mean())/se.std()
idx=np.random.default_rng(0).choice(az.size, 4000, replace=False)
ax[3].scatter(az.ravel()[idx], ez.ravel()[idx], s=3, alpha=0.25, c="#4c72b0", edgecolors="none")
ax[3].set_xlabel("aleatoric (z-scored)", fontsize=9.5); ax[3].set_ylabel("epistemic (z-scored)", fontsize=9.5)
ax[3].set_title(f"per-pixel relation ($r={r:.2f}$)", fontsize=10.5)
ax[3].axhline(0, color="grey", lw=0.5); ax[3].axvline(0, color="grey", lw=0.5)
fig.tight_layout()
fig.savefig("/tmp/ae_uncertainty.pdf", bbox_inches="tight", dpi=200)
fig.savefig("/tmp/ae_uncertainty.png", bbox_inches="tight", dpi=130)
print("saved  r=%.3f  aleatoric_mean=%.4f epistemic_mean=%.4f"%(r, sa.mean(), se.mean()))
