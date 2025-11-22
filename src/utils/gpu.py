# src/utils/gpu.py
# Utility functions for GPU detection, configuration, and optimization in PyTorch.
# - pick_device(): chooses between CPU or CUDA based on system availability or config setting.
# - gpu_caps(): queries GPU hardware capabilities (compute capability, bf16/tf32 support, etc.).
# - configure_backends(): enables performance optimizations like cuDNN benchmarking and TF32 math.
# - should_compile(): checks if torch.compile can be safely used given the GPU and YAML settings.
# - disable_dynamo_globally(): disables PyTorch Dynamo/Inductor compilation on unsupported GPUs.
# Used to automatically configure the best GPU settings for training and evaluation.

import os, torch

def pick_device(cfg_device: str | None = None) -> torch.device:
    want = (cfg_device or "auto").lower()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if want == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(want)

def gpu_caps(device: torch.device) -> dict:
    caps = {
        "is_cuda": (device.type == "cuda"),
        "cc_major": None, "cc_minor": None,
        "bf16": False, "tf32": False,
        "compile_ok": False,
    }
    if not caps["is_cuda"]:
        return caps
    maj, minr = torch.cuda.get_device_capability()
    caps["cc_major"], caps["cc_minor"] = maj, minr
    caps["bf16"] = torch.cuda.is_bf16_supported()
    # TF32 available (Ampere+)
    caps["tf32"] = (maj >= 8)
    # torch.compile (Inductor/Triton) needs cc >= 7.0
    caps["compile_ok"] = (maj >= 7)
    return caps

def configure_backends(caps: dict):
    import torch, os
    if not caps.get("is_cuda", False):
        return

    # cuDNN autotune for speed on fixed shapes
    torch.backends.cudnn.benchmark = True

    # Use *new* TF32 controls (deprecates allow_tf32 flags)
    # If the GPU supports TF32 (Ampere+), enable TF32 for convolutions,
    # and keep matmul in IEEE unless explicitly want TF32 matmul too.
    if caps.get("tf32", False):
        try:
            # Convolution kernels (cuDNN)
            torch.backends.cudnn.conv.fp32_precision = "tf32" 
            # Matmul kernels (CUDA)
            torch.backends.cuda.matmul.fp32_precision = "ieee" 
        except Exception:
            pass

    # Guard against Dynamo on unsupported stacks unless explicitly requested
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def should_compile(cfg: dict, caps: dict) -> tuple[bool, str]:
    """Respect YAML flag and GPU capability; return (use_compile, mode)."""
    train = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    want = bool(train.get("compile", False))
    mode = str(train.get("compile_mode", "default"))
    use = want and caps["is_cuda"] and caps["compile_ok"] and hasattr(torch, "compile")
    return use, mode

def disable_dynamo_globally():
    # Prevent accidental Dynamo/Inductor activation on old GPUs
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
