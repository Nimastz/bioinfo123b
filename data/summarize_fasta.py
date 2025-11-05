# summarize_fasta.py
# Summarizes a FASTA file by sequence length and ambiguity content.
# - Reads all sequences and writes a CSV summary with accession, length, and a flag for ambiguous amino acids (B, J, O, U, X, Z).
# - Prints totals and the number of sequences containing ambiguous letters.
# - Also computes and displays a simple length histogram for quick inspection of sequence distribution.
# Usage:
#   python summarize_fasta.py path/to/input.fasta
# Produces: input.summary.csv with per-sequence statistics.

import sys, re, csv
from pathlib import Path

fa = Path(sys.argv[1])
out_csv = fa.with_suffix(".summary.csv")

def it_fasta(p):
    h, seq = None, []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            if line.startswith(">"):
                if h is not None:
                    yield h, "".join(seq)
                h, seq = line[1:], []
            else:
                seq.append(line)
        if h is not None:
            yield h, "".join(seq)

rows=[]
ambig = re.compile(r"[BJOUXZ]")
n=0; bad=0
for h, s in it_fasta(fa):
    n+=1
    acc = h.split()[0]
    amb = bool(ambig.search(s))
    if amb: bad+=1
    rows.append(dict(accession=acc, length=len(s), has_ambiguous=int(amb)))

print(f"Total: {n}  | Ambiguous-letter sequences: {bad}")
print(f"Writing {out_csv}")
with open(out_csv, "w", newline="", encoding="utf-8") as w:
    w = csv.DictWriter(w, fieldnames=["accession","length","has_ambiguous"])
    w.writeheader(); w.writerows(rows)

# quick length distro
bins = [40,60,80,100,120,140,160,180,200]
hist = {f"{bins[i]}-{bins[i+1]-1}":0 for i in range(len(bins)-1)}
for r in rows:
    L=r["length"]
    for i in range(len(bins)-1):
        if bins[i] <= L < bins[i+1]:
            hist[f"{bins[i]}-{bins[i+1]-1}"]+=1; break
print("Length histogram:")
for k,v in hist.items():
    print(f"{k}: {v}")
