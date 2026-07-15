# Optuna edge x SSIM sensitivity sweep for arm M (NLL+SSIM+Edge+FFL).
# Purpose: resolve the C2 confound. Arm M underperforms arm D at the literature-default
# weights (lambda_ssim=0.5, lambda_edge=0.1). This sweep searches (lambda_ssim, lambda_edge)
# jointly to test whether a better weighting recovers arm D's PSNR (=> the deficit was a
# weight artefact) or whether even the best edge/SSIM mix still lags (=> the SSIM-edge
# interaction is fundamental). lambda_ffl held at its default 0.1. Resumable via SQLite.
import os, sys, subprocess, csv, math
import optuna

PROJECT = "/rds/user/stm43/hpc-work/ppvae_hformer"
SWEEP   = "/rds/user/stm43/hpc-work/ppvae_results/optuna_sweep_edge_m"
os.makedirs(SWEEP, exist_ok=True)
DATA = "/home/stm43/chest_xray" if os.path.isdir("/home/stm43/chest_xray") else "/rds/user/stm43/hpc-work/chest_xray"
EPOCHS  = int(os.environ.get("SWEEP_EPOCHS", "40"))
NTRIALS = int(os.environ.get("SWEEP_TRIALS", "14"))
ARM     = "arm_m_full_det"

def read_best_val(run_dir):
    p = os.path.join(run_dir, "train_log.csv")
    if not os.path.exists(p): return None
    best = None
    with open(p) as f:
        for row in csv.DictReader(f):
            try: v = float(row["val_psnr"])
            except (ValueError, KeyError): continue
            if not math.isnan(v) and (best is None or v > best): best = v
    return best

def objective(trial):
    ss = trial.suggest_float("lambda_ssim", 0.1, 1.0)
    ed = trial.suggest_float("lambda_edge", 0.005, 0.5, log=True)
    run = f"trial_{trial.number:03d}"
    cmd = [sys.executable, "-u", "scripts/train_proposed.py",
           "--arm", ARM, "--data_dir", DATA, "--output_dir", SWEEP,
           "--epochs", str(EPOCHS), "--noise_level", "mid",
           "--batch_size", "16", "--num_workers", "8",
           "--lambda_ssim", f"{ss:.4f}", "--lambda_edge", f"{ed:.4f}",
           "--lambda_ffl", "0.1",
           "--run_name", run]
    print(f"\n=== TRIAL {trial.number}: lambda_ssim={ss:.3f} lambda_edge={ed:.4f} ===", flush=True)
    r = subprocess.run(cmd, cwd=PROJECT)
    if r.returncode != 0:
        print(f"TRIAL {trial.number} FAILED (rc={r.returncode})", flush=True)
        raise optuna.TrialPruned()
    val = read_best_val(os.path.join(SWEEP, run))
    if val is None:
        raise optuna.TrialPruned()
    print(f"TRIAL {trial.number} val_psnr={val:.3f}", flush=True)
    return val

if __name__ == "__main__":
    storage = f"sqlite:///{SWEEP}/study.db"
    study = optuna.create_study(direction="maximize", study_name="edge_ssim_m",
                                storage=storage, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42))
    # arm M literature-default weights enqueued as trial 0 for reference
    if len(study.trials) == 0:
        study.enqueue_trial({"lambda_ssim": 0.5, "lambda_edge": 0.1})
    study.optimize(objective, n_trials=NTRIALS)
    print("\nBEST:", study.best_params, f"val_psnr={study.best_value:.3f}", flush=True)
    try:
        study.trials_dataframe().to_csv(os.path.join(SWEEP, "trials.csv"), index=False)
        print("wrote", os.path.join(SWEEP, "trials.csv"))
    except Exception as e:
        print("trials_dataframe failed:", e)
