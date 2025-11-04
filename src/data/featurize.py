# src/data/featurize.py
import argparse, json, sys, re
from pathlib import Path
import numpy as np

# ---------------- helpers ----------------

# 20 AAs + X
_AA = b"ACDEFGHIKLMNPQRSTVWY"
_AA2I = {c: i for i, c in enumerate(_AA)}
_AA2I.update({ord("X"): 20, ord("x"): 20})

def _seq_bytes_to_idx(seq_bytes: np.ndarray) -> np.ndarray:
    out = np.empty(seq_bytes.shape[0], dtype=np.int64)
    for i, b in enumerate(seq_bytes.tolist()):
        out[i] = _AA2I.get(b, 20)
    return out

def load_npz(path: str):
    """Return (seq_idx:int64[L], emb:None) for dataset.py."""
    d = np.load(path, allow_pickle=False)
    seq_bytes = d["seq"].astype(np.uint8)
    seq_idx = _seq_bytes_to_idx(seq_bytes)
    return seq_idx, None

_ILLEGAL = re.compile(r'[<>:"/\\|?*]+')
def _safe_name(s: str) -> str:
    return _ILLEGAL.sub("_", s)

def _safe_rel_path(features_field: str, out_root: Path) -> Path:
    """
    Map JSONL 'features' (e.g., 'features/HCV_p7/ID.npz') to a path under out_root,
    and sanitize every component for Windows.
    """
    p = Path(features_field)
    parts = list(p.parts)
    # drop leading 'features' so we re-root under out_root
    if parts and parts[0].lower() == "features":
        parts = parts[1:]
    parts = [_safe_name(x) for x in parts]
    return out_root.joinpath(*parts)

def _write_dummy_features(out_path: Path, seq: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             seq=np.frombuffer(seq.encode("ascii"), dtype=np.uint8),
             length=len(seq))

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Write per-sequence feature .npz files directly from index JSONL (uses 'seq')."
    )
    ap.add_argument("--index", required=True, help="data/train.jsonl or data/val.jsonl (must contain 'seq')")
    ap.add_argument("--out_dir", required=True, help="Root dir to place NPZs (e.g., data/features)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_dir)

    with open(args.index, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    # fail fast if seq missing
    bad = [r.get("id") for r in rows if not r.get("seq")]
    if bad:
        print(f"[error] {len(bad)} rows missing 'seq' (e.g., id={bad[0]!r}). This featurizer is index-only.", file=sys.stderr)
        sys.exit(2)

    # group by family for tidy logs
    by_fam = {}
    for r in rows:
        fam = r.get("family", "_")
        by_fam.setdefault(fam, []).append(r)

    total_written = total_skipped = 0

    for fam, group in by_fam.items():
        fam_written = fam_skipped = 0
        for r in group:
            seq = r["seq"]
            # honor JSONL 'features' path, but re-root under out_dir and sanitize
            feats = r.get("features", f"features/{fam}/{_safe_name(r['id'])}.npz")
            outp = _safe_rel_path(feats, out_root)

            if outp.exists() and not args.overwrite:
                fam_skipped += 1
                continue

            _write_dummy_features(outp, seq)
            fam_written += 1
            if args.debug:
                print(f"[ok] wrote {outp} (L={len(seq)})")

        total_written += fam_written
        total_skipped += fam_skipped
        print(f"[family] {fam}: wrote={fam_written}  skipped={fam_skipped}  (root: {out_root})")

    print(f"[summary] wrote={total_written}  skipped={total_skipped}  out_dir={out_root}")

if __name__ == "__main__":
    main()
