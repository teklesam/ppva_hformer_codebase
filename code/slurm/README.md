# HPC Job Scripts (Cambridge CSD3) — PP-VAE-Hformer

Every result, table, and figure in the dissertation was produced by the SLURM batch jobs
in this directory, run on the **Cambridge Service for Data Driven Discovery (CSD3)**. This
index lists each job, its resources, what it runs, and what it produces, grouped by pipeline
stage. It is a companion to the top-level [`README.md`](../../README.md) reproduction guide.

## Submission environment

| Setting | Value |
|---|---|
| GPU partition / account | `ampere` (NVIDIA A100 80 GB) / `mrc-bsu2-sl2-gpu` |
| CPU partition / account | `icelake` / `mrc-bsu2-sl2-cpu` (light post-processing only) |
| Modules | `rhel8/default-amp`, `cuda/12.1` |
| Environment | `conda activate ppvae` (Python 3.12, PyTorch 2.5.1) |
| Project root | `/rds/user/stm43/hpc-work/ppvae_hformer` |
| Data / results | `…/chest_xray` · `…/ppvae_results` |

Submit with `sbatch <script>.sh` from the project root; logs are written to `…/ppvae_results/logs/`
(or `…/ppvae_hformer/logs/`). Job arrays fan one task out per arm/baseline — see the `--array`
range in each header.

## Pipeline order

`1 train arms → 2 fine-tune R/S → 3 train baselines → 4 evaluate → 5 sweeps (offline) → 6 figures`.
Stages 1–3 are independent and were launched together; 4 depends on 1–3; 6 depends on 4.

---

### 1 · Training — 19 ablation arms (A–S)

| Script | Resources | Wall | Runs | Purpose / output |
|---|---|---|---|---|
| `array_all_arms.sh` | 1×A100 (array 0–15) | 12 h | `train_proposed.py` | The 16 core arms A–P, one array task each → `<arm>/best_model.pth` |
| `array_new_arms.sh` | 1×A100 (array 0–10) | 12 h | `train_proposed.py` | Re-run subset of arms after config changes |
| `train_arm_q_charb.sh` | 1×A100 | 12 h | `train_proposed.py` | Arm Q (Charbonnier pixel norm), from scratch |
| `armb.sh` | 8c CPU (icelake) | 20 m | `_arm_b_uncertainty_006.py` | Arm B aleatoric-map regeneration (post-hoc) |

### 2 · Two-stage loss-substitution fine-tuning (Arms R, S)

Depend on Arm A's converged checkpoint (stage 1).

| Script | Resources | Wall | Runs | Purpose / output |
|---|---|---|---|---|
| `finetune_r.sh` | 1×A100 | 8 h | `finetune_from_l2.py` | Arm R = L1+SSIM+FFL fine-tuned from A (best perceptual) |
| `finetune_s.sh` | 1×A100 | 8 h | `finetune_from_l2.py` | Arm S = NLL+SSIM+FFL fine-tuned from A (calibrated, ~0 PSNR cost) |

### 3 · Retrained baselines (8)

| Script | Resources | Wall | Runs | Purpose / output |
|---|---|---|---|---|
| `array_baselines.sh` | 1×A100 (array) | 35 h | `train_baseline.py` | KAIR CNNs: DnCNN, IRCNN, FFDNet, DRUNet, SwinIR |
| `train_dncnn.sh` | 1×A100 | 6 h | `train_dncnn.py` | DnCNN (depth-20, 667 k params) standalone retrain |
| `nafnet.sh` | 1×A100 | 6 h | `train_nafnet.py` | NAFNet (116 M) via BasicSR |
| `scunet.sh` | 1×A100 | 12 h | `train_scunet.py` | SCUNet (57 M), strongest baseline (34.15 dB) |
| `sharpxr.sh` | 1×A100 | 8 h | `sharpxr_baseline.py` | SharpXR paediatric-CXR denoiser |
| `train_ddpm.sh` | 1×A100 | 12 h | `train_ddpm.py` | Palette DDPM (reported as an honest negative result) |

### 4 · Evaluation, metrics & calibration

| Script | Resources | Wall | Runs | Purpose / output |
|---|---|---|---|---|
| `evaluate_all.sh` | 1×A100 | 4 h | `evaluate_all.py` | Main pass → `per_image_metrics.csv` (PSNR/SSIM/NLL, 3 noise levels, 1 000-resample CIs) |
| `eval_qrs.sh` | 1×A100 | 1.5 h | `evaluate_all.py` | Same for the supplementary arms Q, R, S |
| `eval_baselines.sh` | 1×A100 | 2 h | `eval_baselines.py` | The 8 retrained baselines on the unified pipeline |
| `sota_eval.sh` | 1×A100 | 40 m | `eval_sota_unified.py` | NAFNet/SCUNet/SharpXR on the identical unified pass (Table 4.2/4.3) |
| `compute_fsim.sh` | 1×A100 | 2 h | `compute_fsim.py` | Held-out FSIM + LPIPS (`piq`) for all conditions |
| `compute_subgroup.sh` | 1×A100 | 3 h | `compute_subgroup_metrics.py` | Normal / Bacterial / Viral subgroup metrics |
| `bm3d_eval.sh` | 1×A100 | 30 m | `eval_bm3d.py` | Classical BM3D baseline (oracle-noise) |
| `sigma_recal.sh` | 1×A100 | 1 h | `sigma_recal.py` | σ-scaling calibration fit (s* within 1 % of unity) |
| `eval_trials.sh` | 1×A100 | 45 m | `eval_trials_metrics.py` | Per-image metrics for the Optuna trial models |

### 5 · Hyperparameter sweeps (offline; inform λ-sensitivity discussion)

| Script | Resources | Wall | Runs | Purpose / output |
|---|---|---|---|---|
| `optuna_sweep.sh` | 1×A100 | 34 h | `optuna_sweep.py` | Loss-weight search around Arm D (14 trials, 40 epochs) |
| `optuna_edge.sh` | 1×A100 | 36 h | `optuna_sweep_edge.py` | Edge/SSIM-weight sweep probing the Arm M interference |

### 6 · Figures & qualitative dumps

| Script | Resources | Wall | Runs | Purpose / output |
|---|---|---|---|---|
| `gen_figures_v2.sh` | 1×A100 | 2 h | `generate_figures_v2.py` | Ablation facets, dose curves, per-arm qualitative panels, latent UMAP |
| `roi_panels.sh` | 1×A100 | 1 h | `generate_roi_panels_v2.py` | Region-of-interest comparison grids |
| `roi_nafnet_scunet.sh` | 1×A100 | 45 m | `generate_nafnet_scunet_roi.py` | SOTA-baseline ROI crops |
| `bm3d_roi.sh` | 1×A100 | 30 m | `generate_bm3d_roi_panel.py` | BM3D ROI panel |
| `gen_vae_qual.sh` | 1×A100 | 1 h | `gen_vae_qual_only.py` | VAE epistemic-map qualitative panels |
| `hotspot.sh` | 1×A100 | 20 m | `generate_uncertainty_hotspot.py` | Prototype reliability-triage overlay (Fig. 5.1) |
| `dose_bdh.sh` | 1×A100 | 30 m | `dose_uncertainty_bdh.py` | Dose-varying uncertainty figure (Arms B, D, H) |
| `dump_roi.sh` | 1×A100 | 20 m | `dump_roi_recons.py` | Dump ROI reconstruction tensors for offline montage |
| `dump_latent.sh` | 1×A100 | 40 m | `dump_latent_embeddings.py` | Latent codes for ARI / silhouette / UMAP |
| `gen_latent_clean.sh` | 1×A100 | 45 m | `generate_latent_clean.py` | Latent-space projection figures |

### Helpers (not SLURM jobs)

| Script | Purpose |
|---|---|
| `deploy_csd3.sh` | Sync local code → CSD3, verify imports, submit jobs (run from the laptop) |
| `monitor.sh` | Live training monitor: `watch -n 60 bash monitor.sh` reads val-PSNR/KL from each arm's CSV |

---

*All jobs were run under the same data split, Foi noise model, and 624-image held-out test set,
so every comparison is controlled to a single variable (see dissertation §3).*
