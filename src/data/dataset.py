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
import numpy as np 
TEACHER_DIR = Path(os.environ.get("VIROPORIN_TEACHER_ROOT", "alphafold/npz"))
    
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

        # ---- NEW: load AlphaFold teacher if available ----
        xyz_ref = None
        tors_ref = None

        if TEACHER_DIR is not None and TEACHER_DIR.exists():
            sid = row.get("id", str(i))
            tpath = TEACHER_DIR / f"{sid}.npz"
            if tpath.exists():
                tnpz = np.load(tpath)
                # produced by AF_teacher.py
                xyz_ref = tnpz.get("xyz_ref")   # (L,3)
                tors_ref = tnpz.get("tors_ref") # (L,3) [phi, psi, omega]
                if xyz_ref is not None:
                    xyz_ref = xyz_ref.astype("float32")
                if tors_ref is not None:
                    tors_ref = tors_ref.astype("float32")

                # match any max_len truncation
                L = len(seq_idx)
                if xyz_ref is not None and xyz_ref.shape[0] > L:
                    xyz_ref = xyz_ref[:L]
                if tors_ref is not None and tors_ref.shape[0] > L:
                    tors_ref = tors_ref[:L]

        return {
            "id": row.get("id", str(i)),
            "seq_idx": seq_idx,
            "emb": emb,
            "xyz_ref": xyz_ref,
            "tors_ref": tors_ref,
        }

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

    # ---- NEW: teacher tensors (if present) ----
    have_xyz_ref = any(b.get("xyz_ref") is not None for b in batch)
    have_tors_ref = any(b.get("tors_ref") is not None for b in batch)

    xyz_ref = None
    tors_ref = None
    if have_xyz_ref:
        xyz_ref = torch.full((B, L, 3), float('nan'), dtype=torch.float32)
    if have_tors_ref:
        tors_ref = torch.full((B, L, 3), float('nan'), dtype=torch.float32)

    # Fill tensors with data
    for i, b in enumerate(batch):
        l = len(b["seq_idx"])
        seq[i, :l] = torch.as_tensor(b["seq_idx"], dtype=torch.long)

        if emb is not None and b.get("emb") is not None:
            eb = torch.as_tensor(b["emb"], dtype=torch.float32)
            emb[i, :l] = eb

        if xyz_ref is not None and b.get("xyz_ref") is not None:
            xr = torch.as_tensor(b["xyz_ref"], dtype=torch.float32)
            xr = xr[:l]
            xyz_ref[i, :xr.shape[0]] = xr

        if tors_ref is not None and b.get("tors_ref") is not None:
            tr = torch.as_tensor(b["tors_ref"], dtype=torch.float32)
            tr = tr[:l]
            tors_ref[i, :tr.shape[0]] = tr

    # --- Sanitize embeddings for NaN/Inf values and log if any were found ---
    if emb is not None:
        nan_mask = torch.isnan(emb) | torch.isinf(emb)
        num_bad = nan_mask.sum().item()
        if num_bad > 0:
            print(f"[collate] Warning: found {num_bad} non-finite values in batch embeddings — sanitized.")
        emb = torch.nan_to_num(emb, nan=0.0, posinf=1e4, neginf=-1e4)

    out = {"seq_idx": seq, "emb": emb}
    if xyz_ref is not None:
        out["xyz_ref"] = xyz_ref
    if tors_ref is not None:
        out["tors_ref"] = tors_ref
    return out

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