# src/data/dataset.py
# Handles dataset loading and preprocessing for viroporin sequence and feature data.
# - JsonlSet: loads sequence entries from a JSONL index file and retrieves their feature .npz files.
# - make_loaders(): builds PyTorch DataLoader objects for training and validation sets.
# - Helper functions (_resolve_features_path, _safe_name, etc.) ensure cross-platform
#   compatibility and can locate or sanitize feature file paths (useful on Windows/OneDrive).
# This module provides a reliable way to load sequence features and prepare them for model input.

import json, os, re
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.featurize import load_npz
import torch
_ILLEGAL = re.compile(r'[<>:"/\\|?*]+')
def _safe_name(s: str) -> str:
    return _ILLEGAL.sub("_", s)

def _safe_features_path(raw_path: str) -> str:
    p = Path(raw_path)
    if p.is_absolute():
        parts = list(p.parts)
        head, tail = parts[0], parts[1:]
        tail = [_safe_name(x) for x in tail]
        return str(Path(head).joinpath(*tail))
    else:
        parts = [_safe_name(x) for x in p.parts]
        return str(Path(*parts))
    
def _normalize_pat(stem: str) -> re.Pattern:
    # Turn 'AAC40516.1_HCV_p7_SRC:NCBI' into a regex like 'AAC40516.*1.*HCV.*p7.*SRC.*NCBI'
    pat = re.sub(r'[^A-Za-z0-9]+', '.*', stem)
    return re.compile(pat + r'\.npz$', re.IGNORECASE)

def _candidate_roots(p: Path):
    # roots we’ll search under
    roots = []
    env = os.environ.get("VIROPORIN_FEATURES_ROOT")
    if env:
        roots.append(Path(env))
    roots += [Path("data") / "features", Path("features")]
    # de-dup while preserving order
    seen, uniq = set(), []
    for r in roots:
        rp = r.resolve()
        if rp not in seen:
            seen.add(rp); uniq.append(rp)
    return [r for r in uniq if r.exists()]

def _fuzzy_find_feature(raw_path: str) -> Path | None:
    p = Path(raw_path)
    family = p.parts[1] if (len(p.parts) > 1 and p.parts[0].lower() == "features") else None
    stem = p.stem
    rx = _normalize_pat(stem)

    for root in _candidate_roots(p):
        search_dir = (root / family) if family and (root / family).exists() else root
        hits = []
        for f in search_dir.rglob("*.npz"):
            if rx.search(f.name):
                hits.append(f)
        if hits:
            # pick shortest filename (usually the intended sanitized variant)
            hits.sort(key=lambda f: (len(f.name), str(f)))
            return hits[0]
    return None

class JsonlSet(Dataset):
    def __init__(self, index_path, max_len=None):
        with open(index_path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(l) for l in f if l.strip()]
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        feats_path = _resolve_features_path(row["features"])
        seq_idx, emb = load_npz(feats_path)

        # optional length cap
        if self.max_len and len(seq_idx) > self.max_len:
            seq_idx = seq_idx[:self.max_len]
            if emb is not None:
                emb = emb[:self.max_len]

        return {"id": row.get("id", str(i)), "seq_idx": seq_idx, "emb": emb}

def collate(batch):
    import torch

    # Find batch and sequence dimensions
    Ls = [len(b["seq_idx"]) for b in batch]
    L = max(Ls)
    B = len(batch)

    # Create padded sequence tensor (PAD token = 20 for 21-token vocab)
    seq = torch.full((B, L), 20, dtype=torch.long)

    emb = None
    if batch[0].get("emb") is not None and len(batch[0]["emb"]) > 0:
        D = len(batch[0]["emb"][0])
        emb = torch.full((B, L, D), float('nan'), dtype=torch.float32)

    # Fill tensors with data
    for i, b in enumerate(batch):
        l = len(b["seq_idx"])
        seq[i, :l] = torch.as_tensor(b["seq_idx"], dtype=torch.long)
        if emb is not None:
            eb = torch.as_tensor(b["emb"], dtype=torch.float32)
            emb[i, :l] = eb

    # --- Sanitize embeddings for NaN/Inf values and log if any were found ---
    if emb is not None:
        nan_mask = torch.isnan(emb) | torch.isinf(emb)
        num_bad = nan_mask.sum().item()
        if num_bad > 0:
            print(f"[collate] Warning: found {num_bad} non-finite values in batch embeddings — sanitized.")
        emb = torch.nan_to_num(emb, nan=0.0, posinf=1e4, neginf=-1e4)

    return {"seq_idx": seq, "emb": emb}

def make_loaders(dc, device=None):
    train = JsonlSet(dc["train_index"], dc.get("max_len"))
    val   = JsonlSet(dc["val_index"],   dc.get("max_len"))
    pin = torch.cuda.is_available()
    nw  = min(int(dc.get("num_workers", 2)), 2)
    train_loader = DataLoader(
        train, batch_size=dc["batch_size"], shuffle=dc["shuffle"],
        num_workers=nw, pin_memory=pin, persistent_workers=(nw>0),
        prefetch_factor=(2 if nw>0 else None), collate_fn=collate
    )
    val_loader = DataLoader(
        val, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=pin, collate_fn=collate
    )
    return train_loader, val_loader

def _resolve_features_path(raw_path: str) -> Path:
    p = Path(raw_path)
    cands = [p]

    # try data/<path> if relative
    if not p.is_absolute():
        cands.append(Path("data") / p)

    # sanitized variants (Windows/OneDrive-safe)
    sanitize = lambda s: re.sub(r'[<>:"/\\|?*]+', "_", s)
    clean = Path(*[sanitize(part) for part in p.parts])
    cands += [clean]
    if not clean.is_absolute():
        cands.append(Path("data") / clean)

    # env root (e.g., VIROPORIN_FEATURES_ROOT=C:\...\data\features)
    root = os.environ.get("VIROPORIN_FEATURES_ROOT")
    if root:
        tail = Path(*p.parts[1:]) if (p.parts and p.parts[0].lower() == "features") else p
        tail_clean = Path(*clean.parts[1:]) if (clean.parts and clean.parts[0].lower() == "features") else clean
        cands += [Path(root) / tail, Path(root) / tail_clean]

    for c in cands:
        if c.exists():
            return c

    # fuzzy fallback: ignore punctuation differences and search known roots
    fuzzy = _fuzzy_find_feature(raw_path)
    if fuzzy is not None:
        return fuzzy

    tried = " | ".join(str(c) for c in cands)
    raise FileNotFoundError(f"features file not found. Tried: {tried}")