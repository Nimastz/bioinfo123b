# dataset_report.py
# read data/train.jsonl and data/test.jsonl, then print a short summary of dataset
import json, re, csv
from pathlib import Path
from collections import Counter

def parse_fasta_stream(lines):
    h, seq = None, []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if h is not None:
                yield h, "".join(seq)
            h, seq = line[1:], []
        else:
            seq.append(line)
    if h is not None:
        yield h, "".join(seq)

def fam_from_header(h):
    """Extract family name from FASTA header (ACC|FAMILY|...)."""
    tok = h.split()[0]
    parts = tok.split("|")
    return parts[1] if len(parts) >= 2 else "unknown"

def count_fasta(path: Path):
    fam = Counter()
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for h, s in parse_fasta_stream(f):
            fam[fam_from_header(h)] += 1
            total += 1
    return total, fam

def count_fasta_dir(dirpath: Path):
    fam = Counter()
    total = 0
    for fa in sorted(dirpath.glob("*.fasta")):
        with open(fa, "r", encoding="utf-8") as f:
            n = 0
            for h, s in parse_fasta_stream(f):
                fam_tag = fam_from_header(h)
                if fam_tag == "unknown":
                    fam_tag = re.sub(r"\.(raw|tagged|filt|ncbi|uniprot)$", "", fa.stem)
                fam[fam_tag] += 1
                n += 1
            total += n
    return total, fam

def load_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def summarize(train_jsonl="data/train.jsonl",
              test_jsonl="data/test.jsonl",
              merged_fasta=None,
              families_dir=None):
    print("\n===== SUMMARY REPORT =====")

    # --- Part A: Toy model dataset (JSONL)
    train = load_jsonl(train_jsonl)
    test = load_jsonl(test_jsonl)
    total = len(train) + len(test)

    csv_rows = []  # for dataset_report.csv

    if total > 0:
        pct_train = (len(train) / total) * 100
        pct_test = (len(test) / total) * 100
        print(f"[Toy model dataset]")
        print(f"  Total: {total}")
        print(f"  Training:  {len(train)} ({pct_train:.1f}%)")
        print(f"  Test:{len(test)} ({pct_test:.1f}%)")

        fam_train = Counter(r.get("family", "unknown") for r in train)
        fam_test = Counter(r.get("family", "unknown") for r in test)
        all_fams = sorted(set(fam_train) | set(fam_test))

        print("\n  Per-family (toy dataset):")
        print(f"  {'Family':20s} {'Train':>8s} {'test':>8s} {'Total':>8s}")
        print("  " + "-" * 46)
        for fam in all_fams:
            tr, va = fam_train.get(fam, 0), fam_test.get(fam, 0)
            total_fam = tr + va
            print(f"  {fam:20s} {tr:8d} {va:8d} {total_fam:8d}")
            csv_rows.append({"family": fam, "train": tr, "test": va, "total": total_fam})

        # write dataset report CSV
        out_csv = Path("data/dataset_report.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as w:
            cw = csv.DictWriter(w, fieldnames=["family", "train", "test", "total"])
            cw.writeheader()
            cw.writerows(csv_rows)
        print(f"\n[Saved] {out_csv}")

    # --- Part B: Full merged FASTA dataset
    if merged_fasta:
        mf = Path(merged_fasta)
        if mf.exists():
            total_full, fam_full = count_fasta(mf)
            print("\n[Full merged FASTA dataset]")
            print(f"  Total sequences: {total_full}")
            print(f"  {'Family':20s} {'Count':>8s}")
            print("  " + "-" * 30)
            csv_rows_full = []
            for fam, c in fam_full.most_common():
                print(f"  {fam:20s} {c:8d}")
                csv_rows_full.append({"family": fam, "count": c})
            out_csv = Path("merged_fasta_summary.csv")
            with open(out_csv, "w", newline="", encoding="utf-8") as w:
                cw = csv.DictWriter(w, fieldnames=["family", "count"])
                cw.writeheader()
                cw.writerows(csv_rows_full)
            print(f"[Saved] {out_csv}")
        else:
            print(f"\n[warn] FASTA file {merged_fasta} not found.")

    # --- Part C: Per-family directory
    if families_dir:
        fd = Path(families_dir)
        if fd.exists():
            total_dir, fam_dir = count_fasta_dir(fd)
            print("\n[Per-family FASTA directory dataset]")
            print(f"  Total sequences: {total_dir}")
            print(f"  {'Family':20s} {'Count':>8s}")
            print("  " + "-" * 30)
            csv_rows_dir = []
            for fam, c in fam_dir.most_common():
                print(f"  {fam:20s} {c:8d}")
                csv_rows_dir.append({"family": fam, "count": c})
            out_csv = Path("fasta_summary.csv")
            with open(out_csv, "w", newline="", encoding="utf-8") as w:
                cw = csv.DictWriter(w, fieldnames=["family", "count"])
                cw.writeheader()
                cw.writerows(csv_rows_dir)
            print(f"[Saved] {out_csv}")
        else:
            print(f"\n[warn] Directory {families_dir} not found.")

    print("============================\n")

if __name__ == "__main__":
    summarize(
        train_jsonl="data/train.jsonl",
        test_jsonl="data/Test.jsonl",
        merged_fasta=None,
        families_dir="data/families_clustered"
    )