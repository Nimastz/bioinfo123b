# build_indices_from_families.py
# Clusters per-family FASTA sequences with CD-HIT and builds train/val JSONL indices.
# - load_fasta_map()/parse_fasta_ids(): read FASTA headers and sequences.
# - run_cdhit(): runs CD-HIT to produce representative sequences at a given identity/coverage.
# - parse_clstr()/write_map_csv(): parse CD-HIT .clstr files and write rep→member cluster maps (CSV).
# - build_indices(): splits representatives into train/val (stratified per family), and writes
#   train.jsonl / val.jsonl rows that embed the sequence and point to a per-sequence features path.
# Supports an --index_only mode to skip clustering and index existing *_90.fasta files.

import argparse, subprocess, sys, csv, json, random
from pathlib import Path

random.seed(1337)

def load_fasta_map(fasta_path: Path):
    """
    Return dict: { header_token (first whitespace-stripped token) -> sequence }
    """
    seqs = {}
    if not fasta_path.exists():
        return seqs
    cur_id, cur_seq = None, []
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(cur_seq)
                cur_id = line[1:].strip().split()[0]
                cur_seq = []
            else:
                cur_seq.append(line.strip())
    if cur_id is not None:
        seqs[cur_id] = "".join(cur_seq)
    return seqs

def parse_fasta_ids(fasta_path: Path):
    return list(load_fasta_map(fasta_path).keys())

def run_cdhit(in_fa: Path, out_fa: Path, cid: float, cov: float, threads: int, mem_mb: int = 16000):
    out_fa.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cd-hit", "-i", str(in_fa), "-o", str(out_fa),
        "-c", str(cid), "-n", "5", "-aL", str(cov), "-aS", str(cov),
        "-d", "0", "-T", str(threads), "-M", str(mem_mb),
    ]
    print("[cd-hit]", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit(f"cd-hit failed on {in_fa.name}: {r.returncode}")
    return out_fa, out_fa.with_suffix(out_fa.suffix + ".clstr")

def parse_clstr(clstr_path: Path):
    clusters, cur = [], {"rep": None, "members": []}
    with open(clstr_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                if cur["members"] or cur["rep"]:
                    clusters.append(cur)
                cur = {"rep": None, "members": []}
                continue
            if ">" in line and "," in line:
                hdr = line.split(">", 1)[1].split("...", 1)[0]
                if line.endswith("*"):
                    cur["rep"] = hdr
                cur["members"].append(hdr)
    if cur["members"] or cur["rep"]:
        clusters.append(cur)
    return clusters

def write_map_csv(clusters, map_csv: Path):
    with open(map_csv, "w", newline="", encoding="utf-8") as w:
        cw = csv.DictWriter(w, fieldnames=["rep_id", "member_id", "cluster_size"])
        cw.writeheader()
        for c in clusters:
            size = len(c["members"])
            rep = c["rep"]
            for m in c["members"]:
                cw.writerow({"rep_id": rep, "member_id": m, "cluster_size": size})

def build_indices(reps_by_family, out_train_jsonl: Path, out_val_jsonl: Path, val_frac: float):
    rng = random.Random(1337)
    train_rows, val_rows = [], []
    cap = None  # set to e.g. 1000 if you want to cap per-family train size

    for family in sorted(reps_by_family):
        rep_fasta = reps_by_family[family]
        seqs = load_fasta_map(rep_fasta)  # header_token -> sequence
        rep_ids = list(seqs.keys())
        rng.shuffle(rep_ids)

        n = len(rep_ids)
        if n <= 2:
            n_val = 0
        elif n <= 9:
            n_val = 1
        else:
            n_val = int(n * val_frac)
        if n_val >= n:
            n_val = max(0, n - 1)

        val_ids = rep_ids[:n_val]
        train_ids = rep_ids[n_val:]
        if cap is not None and len(train_ids) > cap:
            train_ids = train_ids[:cap]

        print(f"[split] {family}: total={n}  train={len(train_ids)}  val={len(val_ids)}")

        def row_for(rid):
            seq = seqs[rid]
            rid_safe = rid.replace("|", "_")
            return {
                "id": rid_safe,
                "seq_header": rid,
                "seq": seq,                  # <-- embed the sequence directly
                "seq_len": len(seq),         # <-- handy metadata
                "features": f"features/{family}/{rid_safe}.npz",
                "family": family
            }

        train_rows.extend(row_for(rid) for rid in train_ids)
        val_rows.extend(row_for(rid) for rid in val_ids)

    out_train_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_train_jsonl, "w", encoding="utf-8") as w:
        for r in train_rows:
            w.write(json.dumps(r) + "\n")
    with open(out_val_jsonl, "w", encoding="utf-8") as w:
        for r in val_rows:
            w.write(json.dumps(r) + "\n")
    print(f"[index] train:{len(train_rows)}  val:{len(val_rows)}")

def main():
    ap = argparse.ArgumentParser(description="Cluster per-family FASTAs with CD-HIT and build train/val indices that include sequences.")
    ap.add_argument("--families_dir", required=True, help="Directory with per-family FASTAs")
    ap.add_argument("--out_dir", required=True, help="Output dir for clustered reps and maps")
    ap.add_argument("--id", type=float, default=0.9, help="CD-HIT identity (0..1)")
    ap.add_argument("--cov", type=float, default=0.8, help="Coverage for both longer/shorter (-aL/-aS)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--index_only", action="store_true",
                    help="Skip CD-HIT; just index existing *_90.fasta files in --families_dir")
    args = ap.parse_args()

    fam_dir = Path(args.families_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reps_by_family = {}

    if args.index_only:
        # Use already-clustered representative FASTAs
        for fa in sorted(fam_dir.glob("*_90.fasta")):
            family = fa.stem.replace("_90", "")
            reps_by_family[family] = fa
        if not reps_by_family:
            raise SystemExit("index_only requested but no *_90.fasta found in --families_dir")
    else:
        # Run CD-HIT on raw/merged per-family FASTAs
        for fa in sorted(fam_dir.glob("*.fasta")):
            name = fa.name
            if name.startswith("_"):
                continue
            if any(tok in name for tok in (".raw.", ".ncbi", ".uniprot", ".tagged", ".filt")):
                continue
            family = fa.stem
            out_fa = out_dir / f"{family}_90.fasta"
            out_fa, clstr = run_cdhit(fa, out_fa, args.id, args.cov, args.threads)
            clusters = parse_clstr(clstr)
            write_map_csv(clusters, out_dir / f"{family}_90_map.csv")
            reps_by_family[family] = out_fa
            print(f"[done] {family}: reps={len(parse_fasta_ids(out_fa))}, clusters={len(clusters)}")

    build_indices(
        reps_by_family,
        out_train_jsonl=out_dir.parent / "train.jsonl",
        out_val_jsonl=out_dir.parent / "test.jsonl",
        val_frac=args.val_frac,
    )

if __name__ == "__main__":
    main()
