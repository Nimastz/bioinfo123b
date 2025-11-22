# train.py
# Main training script for the ViroporinAFMini model.
# Loads configuration settings, prepares data loaders, initializes the model, optimizer, and scheduler,
# and launches the training loop.
# - Automatically selects CPU or GPU and configures PyTorch performance options.
# - Optionally enables torch.compile for supported GPUs to accelerate training.
# - Saves checkpoints and logs progress throughout training.
# Used to train the viroporin 3D structure prediction model from sequence data using the settings in a YAML config file.

import argparse, os, warnings, yaml, torch
from src.utils.gpu import pick_device, gpu_caps, configure_backends, should_compile, disable_dynamo_globally
from src.train.loop import Trainer
from src.train.optim import make_optim
from src.data.dataset import make_loaders
from src.model.viroporin_net import ViroporinAFMini
import yaml
from src.optim import create_optimizer, create_scheduler  
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

def load_full_checkpoint(path, model, opt=None, sched=None, trainer=None, device="cuda", cfg=None):
    """
    Robust checkpoint loader:
    - Upgrades old model keys (fills evo.pair_proj from older checkpoints)
    - Loads model with strict=False to allow benign diffs
    - Tries to load optimizer/scheduler; if groups mismatch, rebuild from cfg
    - Restores scaler and global step if present
    Returns: (opt, sched, step)
    """
    import copy, torch

    raw = torch.load(path, map_location=device)
    sd = copy.deepcopy(raw["model"])

    # ---- UPGRADE SHIM for older checkpoints (no evo.pair_proj.*) ----
    need_ln = ("evo.pair_proj.0.weight" not in sd) or ("evo.pair_proj.0.bias" not in sd)
    need_fc = ("evo.pair_proj.1.weight" not in sd) or ("evo.pair_proj.1.bias" not in sd)
    src_w = sd.get("evo.blocks.0.pair_from_single.proj.weight", None)
    src_b = sd.get("evo.blocks.0.pair_from_single.proj.bias", None)
    if (need_ln or need_fc) and (src_w is not None) and (src_b is not None):
        ds2 = src_w.shape[1]  # 2*ds
        sd["evo.pair_proj.0.weight"] = torch.ones(ds2, device=device)
        sd["evo.pair_proj.0.bias"]   = torch.zeros(ds2, device=device)
        sd["evo.pair_proj.1.weight"] = src_w.clone()
        sd["evo.pair_proj.1.bias"]   = src_b.clone()
        print("[upgrade] filled evo.pair_proj from evo.blocks.0.pair_from_single.proj")

    # ---- Load model (allow benign diffs) ----
    info = model.load_state_dict(sd, strict=False)
    if getattr(info, "missing_keys", None) or getattr(info, "unexpected_keys", None):
        print(f"[warn] relaxed load: missing={list(info.missing_keys)} unexpected={list(info.unexpected_keys)}")

    # ---- Restore step early (useful for sched) ----
    step = raw.get("step", 0)

    # ---- Optimizer + Scheduler restore with safety net ----
    def _rebuild_optimizer_and_scheduler(_cfg, _model, _device):
        # Recreate optimizer/scheduler from your project helper(s).
        # Adjust imports to your project structure if needed.
        new_opt = create_optimizer(_cfg, _model)                
        new_sched = create_scheduler(_cfg, new_opt, start_step=step)
        return new_opt, new_sched

    if opt is not None and raw.get("opt") is not None:
        try:
            opt.load_state_dict(raw["opt"])
        except ValueError as e:
            print(f"[warn] optimizer state not compatible with current model ({e}). Rebuilding optimizer.")
            if cfg is None:
                # Minimal fallback: clear state/groups to keep object reference valid
                opt.state = {}
                opt.param_groups[:] = [{"params": [p for p in model.parameters() if p.requires_grad]}]
            else:
                # Rebuild from cfg and swap into the existing reference (so callers keep 'opt')
                new_opt, _ = _rebuild_optimizer_and_scheduler(cfg, model, device)
                opt.__dict__.update(new_opt.__dict__)

    if sched is not None:
        raw_sched = raw.get("sched", None)
        if raw_sched is not None:
            try:
                sched.load_state_dict(raw_sched)
                # many schedulers track last_epoch/step; force align if available
                if hasattr(sched, "last_epoch"):
                    sched.last_epoch = step
            except Exception as e:
                print(f"[warn] scheduler state not compatible ({e}). Rebuilding scheduler.")
                if cfg is not None:
                    _, new_sched = _rebuild_optimizer_and_scheduler(cfg, model, device)
                    sched.__dict__.update(new_sched.__dict__)

    # ---- AMP scaler (if present) ----
    if trainer is not None:
        if raw.get("scaler") and hasattr(trainer, "scaler"):
            try:
                trainer.scaler.load_state_dict(raw["scaler"])
            except Exception as e:
                print(f"[warn] GradScaler not compatible ({e}); keeping fresh scaler.")

        # Keep the trainer's global step in sync with checkpoint
        trainer.global_step = step

    print(f"[info] Resumed model at step {step}")
    return opt, sched, step

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="Path to checkpoint to resume from (optional)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    torch.manual_seed(cfg["seed"])

    device = pick_device(cfg.get("device", "auto"))
    caps = gpu_caps(device)
    configure_backends(caps)

    use_compile, compile_mode = should_compile(cfg, caps)
    if not use_compile:
        disable_dynamo_globally()

    # Data / Model
    train_loader, val_loader = make_loaders(cfg["data"], device=device)
    model = ViroporinAFMini(cfg["model"]).to(device)

    if use_compile:
        try:
            model = torch.compile(model, mode=compile_mode)
            print(f"[info] using torch.compile ({compile_mode})")
        except Exception as e:
            print(f"[warn] torch.compile disabled: {e}")

    opt, sched = make_optim(model, cfg["train"])
    os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True)

    start_step = 0
    if args.ckpt and os.path.exists(args.ckpt):
        print(f"[info] Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        start_step = int(ckpt.get("step", 0))
    elif args.ckpt:
        warnings.warn(f"[warn] Checkpoint not found: {args.ckpt}")

    trainer = Trainer(cfg, model, opt, sched, device, start_step=start_step)

    if args.ckpt and os.path.exists(args.ckpt):
        opt, sched, start_step = load_full_checkpoint(
            args.ckpt, model, opt, sched, trainer, device=device, cfg=cfg
        )
        trainer.start_step = start_step
        trainer.global_step = start_step

    trainer.fit(train_loader, val_loader)

if __name__ == "__main__":
    main()
