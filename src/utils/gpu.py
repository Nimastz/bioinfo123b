# src/utils/gpu.py
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
    if not caps["is_cuda"]:
        return
    torch.backends.cudnn.benchmark = True
    if caps["tf32"]:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # torch >= 2.0
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass

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
