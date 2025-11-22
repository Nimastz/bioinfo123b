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

def load_full_checkpoint(path, model, opt=None, sched=None, trainer=None, device="cuda"):
    import copy, torch
    raw = torch.load(path, map_location=device)
    sd = copy.deepcopy(raw["model"])

    # ---- UPGRADE SHIM for older checkpoints (no evo.pair_proj.*) ----
    need_ln = ("evo.pair_proj.0.weight" not in sd) or ("evo.pair_proj.0.bias" not in sd)
    need_fc = ("evo.pair_proj.1.weight" not in sd) or ("evo.pair_proj.1.bias" not in sd)
    src_w = sd.get("evo.blocks.0.pair_from_single.proj.weight", None)
    src_b = sd.get("evo.blocks.0.pair_from_single.proj.bias", None)
    if (need_ln or need_fc) and (src_w is not None) and (src_b is not None):
        # Initialize LN to identity / zero (matches nn.LayerNorm defaults)
        ds2 = src_w.shape[1]     # 2*ds
        sd["evo.pair_proj.0.weight"] = torch.ones(ds2, device=device)
        sd["evo.pair_proj.0.bias"]   = torch.zeros(ds2, device=device)
        # Copy the old linear into pair_proj[1]
        sd["evo.pair_proj.1.weight"] = src_w.clone()
        sd["evo.pair_proj.1.bias"]   = src_b.clone()
        print("[upgrade] filled evo.pair_proj from evo.blocks.0.pair_from_single.proj")

    # ---- now load (non-strict still allows other benign diffs) ----
    info = model.load_state_dict(sd, strict=False)
    if getattr(info, "missing_keys", None) or getattr(info, "unexpected_keys", None):
        print(f"[warn] relaxed load: missing={list(info.missing_keys)} unexpected={list(info.unexpected_keys)}")

    # Optimizer / sched / scaler / step as before
    if opt is not None and raw.get("opt"):   opt.load_state_dict(raw["opt"])
    if sched is not None and raw.get("sched"): sched.load_state_dict(raw["sched"])
    if trainer is not None:
        if raw.get("scaler") and hasattr(trainer, "scaler"): trainer.scaler.load_state_dict(raw["scaler"])
        trainer.global_step = raw.get("step", 0)

    print(f"[info] Resumed full training state from step {raw.get('step', 0)}")

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
        load_full_checkpoint(args.ckpt, model, opt, sched, trainer, device=device)
        trainer.start_step = start_step 
        trainer.global_step = start_step  

    trainer.fit(train_loader, val_loader)

if __name__ == "__main__":
    main()
