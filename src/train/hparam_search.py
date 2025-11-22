import os, argparse, copy, yaml, optuna, math, subprocess, tempfile, json
import subprocess, sys
    
def run_one(cfg_path, max_steps=1500):
    # Launch your existing train.py for a short run and read the final JSON summary
    # Modify your train loop to write a tiny summary JSON after fit() ends or every N steps.
    # Example assumes train.py writes 'summary.json' into ckpt_dir.
    env = os.environ.copy()

    proc = subprocess.Popen(
        ["python", "-u", "train.py", "--config", cfg_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
    )

    # Stream stdout line by line to see step progress
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()

    proc.wait()

    # Read the summary the Trainer wrote
    ckpt_dir = yaml.safe_load(open(cfg_path))["train"]["ckpt_dir"]
    summ_path = os.path.join(ckpt_dir, "summary.json")
    with open(summ_path, "r", encoding="utf-8") as f:
        summ = json.load(f)
    return float(summ["score"]), summ  # (objective, anything else you logged)

def suggest_and_dump_cfg(base_cfg, trial, tmp_dir):
    cfg = copy.deepcopy(base_cfg)

    # --- Search space (adjust bounds to your taste) ---
    # Global scale and warmup
    cfg["loss_weights"]["priors"]   = trial.suggest_float("lw.priors", 0.01, 0.08, log=True)
    cfg["train"]["priors_warmup_steps"] = trial.suggest_int("train.priors_warmup_steps", 2000, 8000, step=1000)

    # Individual prior scales
    cfg["loss_weights"]["membrane"] = trial.suggest_float("lw.membrane", 0.01, 0.05)
    cfg["loss_weights"]["interface"]= trial.suggest_float("lw.interface",0.01, 0.05)
    cfg["loss_weights"]["pore"]     = trial.suggest_float("lw.pore",     0.01, 0.05)

    # Pore/Interface shape
    cfg.setdefault("priors", {})
    cfg["priors"]["pore_target_A"]  = trial.suggest_float("priors.pore_target_A", 1.0, 4.0)
    cfg["priors"]["intf_cutoff"]    = trial.suggest_float("priors.intf_cutoff", 3.5, 9.5)
    cfg["priors"]["cap_A"]          = trial.suggest_float("priors.cap_A", 1.0, 4.5)

    # Optimizer side (optional)
    cfg["train"]["lr"]              = trial.suggest_float("train.lr", 1e-4, 2e-3, log=True)
    cfg["train"]["grad_clip"]       = trial.suggest_float("train.grad_clip", 0.5, 5.0)

    # Short-run budget for search
    cfg["train"]["steps"]           = trial.suggest_int("train.steps", 1200, 2400, step=300)
    cfg["train"]["eval_every"]      = 10
    cfg["train"]["log_every"]       = 10
    cfg["train"]["ckpt_dir"]        = os.path.join(tmp_dir, f"trial_{trial.number:04d}")

    # Write temp YAML
    cfg_path = os.path.join(tmp_dir, f"trial_{trial.number:04d}.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f: yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="path to base YAML (e.g., configs/recommended.yaml)")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--study", default="viroporin_search")
    args = ap.parse_args()

    base_cfg = yaml.safe_load(open(args.base, "r", encoding="utf-8"))
    os.makedirs("search_runs", exist_ok=True)

    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=300, reduction_factor=3)
    sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=8)

    def objective(trial):
        with tempfile.TemporaryDirectory(dir="search_runs") as tmp:
            cfg_path = suggest_and_dump_cfg(base_cfg, trial, tmp)
            print(f"\n[trial {trial.number}] Starting run with parameters:")
            for k, v in trial.params.items():
                print(f"   {k}: {v}")
            try:
                score, summ = run_one(cfg_path)
                print(f"[trial {trial.number}] Finished  → score={score:.4f}")
                print(f"  ema_loss={summ.get('ema_loss'):.4f}, "
                    f"mem_ratio={summ.get('mem_ratio'):.3f}, "
                    f"pore_ratio={summ.get('pore_ratio'):.3f}")
            except Exception as e:
                print(f"[trial {trial.number}] Failed → {e}")
                return float("inf")

            trial.report(score, step=1)
            if trial.should_prune():
                raise optuna.TrialPruned()
            return score

    study = optuna.create_study(direction="minimize", study_name=args.study, sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print("[best]", study.best_trial.number, study.best_trial.value)
    print("[params]", study.best_trial.params)

if __name__ == "__main__":
    main()
