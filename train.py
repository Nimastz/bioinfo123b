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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="Path to checkpoint to resume from (optional)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    torch.manual_seed(cfg["seed"])

    # Device + capability
    device = pick_device(cfg.get("device", "auto"))
    caps = gpu_caps(device)
    configure_backends(caps)

    # Respect YAML compile flag; auto-skip on unsupported GPUs (e.g., GTX 1060 cc=6.1)
    use_compile, compile_mode = should_compile(cfg, caps)
    if not use_compile:
        disable_dynamo_globally()  # avoid Triton/Inductor try on old GPUs

    # Data / Model
    train_loader, val_loader = make_loaders(cfg["data"], device=device)
    model = ViroporinAFMini(cfg["model"]).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[info] Model parameters: {num_params:,}  (~{num_params/1e6:.2f}M)")

    # Optional: load checkpoint if provided
    if args.ckpt and os.path.exists(args.ckpt):
        print(f"[info] Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
    elif args.ckpt:
        warnings.warn(f"[warn] Checkpoint not found: {args.ckpt}")

    # Optional: torch.compile (Volta/Turing/Ampere+ only, and only if enabled in YAML)
    if use_compile:
        try:
            model = torch.compile(model, mode=compile_mode)
            print(f"[info] using torch.compile ({compile_mode})")
        except Exception as e:
            print(f"[warn] torch.compile disabled: {e}")

    # Optim & sched
    opt, sched = make_optim(model, cfg["train"])
    os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True)

    trainer = Trainer(cfg, model, opt, sched, device)
    trainer.fit(train_loader, val_loader)

if __name__ == "__main__":
    main()
