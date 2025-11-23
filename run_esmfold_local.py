#!/usr/bin/env python3
import os, json, re, sys, torch
from pathlib import Path

INDEX = r"/mnt/data/test_diverse_subset.jsonl"
OUT_DIR = r"/mnt/data/results_esm"
os.makedirs(OUT_DIR, exist_ok=True)

def sanitize_id(s):
    return re.sub(r'[^A-Za-z0-9_.:\-]+', '_', s)

# Load ESMFold model
try:
    import esm
except ImportError:
    print("[error] Please install esm: pip install fair-esm")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = esm.pretrained.esmfold_v1()
model = model.eval().to(device)

def fold_one(pid: str, seq: str, out_pdb: str):
    with torch.no_grad():
        output = model.infer_pdb(seq)
    with open(out_pdb, "w") as fw:
        fw.write(output)

# Read index and fold
with open(INDEX, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        pid = rec.get("id")
        seq = rec.get("seq") or rec.get("sequence") or rec.get("aa") or ""
        if not pid or not seq:
            continue
        pid_s = sanitize_id(pid)
        out_pdb = os.path.join(OUT_DIR, f"{pid_s}.pdb")
        if os.path.exists(out_pdb):
            print(f"[skip] exists: {out_pdb}")
            continue
        print(f"[fold] {pid_s} -> {out_pdb}")
        fold_one(pid_s, seq, out_pdb)

print(f"[done] ESMFold PDBs in: {OUT_DIR}")
