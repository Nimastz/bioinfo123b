# merge_viroporins_families.py
# Merges and deduplicates per-family FASTA files from UniProt and NCBI.
# - parse_fasta_stream()/read_fasta(): stream and load FASTA entries.
# - first_token_id(): normalizes headers to an accession-like ID (leftmost token).
# - choose_best_file(): selects the preferred FASTA for a family from each source directory.
# - merge_family_familyname(): builds a unique-by-sequence set, preferring UniProt headers,
#   and writes headers as ACC|FAMILY|SRC:<UniProt|NCBI>.
# - Optionally writes a global merged FASTA and always writes a per-family manifest CSV.
# CLI takes --uniprot_dir, --ncbi_dir, and --out_dir; can also emit a single merged file with --all_merged.

import argparse
from pathlib import Path
import csv
import sys
from typing import Dict, Tuple, Iterable, List
import re

def parse_fasta_stream(lines: Iterable[str]):
    h, seq = None, []
    for line in lines:
        line = line.rstrip("\n")
        if not line:
            continue
        if line.startswith(">"):
            if h is not None:
                yield h, "".join(seq)
            h = line[1:]
            seq = []
        else:
            seq.append(line.strip())
    if h is not None:
        yield h, "".join(seq)

def read_fasta(path: Path) -> List[Tuple[str, str]]:
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(parse_fasta_stream(f))

def first_token_id(header: str) -> str:
    # Take the first whitespace-delimited token, then split off any trailing pipes (keep leftmost as accession-like id)
    tok = header.split()[0]
    # If header already in ACC|FAM|TAG format, keep ACC only
    if "|" in tok:
        tok = tok.split("|")[0]
    return tok

def wrap_fasta_write(out, header: str, seq: str):
    out.write(f">{header}\n")
    for i in range(0, len(seq), 60):
        out.write(seq[i:i+60] + "\n")

def choose_best_file(d: Path, family: str) -> Path:
    # Preference order per source directory
    candidates = [
        d / f"{family}.fasta",
        d / f"{family}.ncbi.tagged.fasta",
        d / f"{family}.uniprot.tagged.fasta",
        d / f"{family}.filt.fasta",
        d / f"{family}.raw.fasta",
        # fallbacks: any file starting with family
    ]
    for c in candidates:
        if c.exists():
            return c
    # fallback: first matching pattern
    hits = list(d.glob(f"{family}*.fasta"))
    return hits[0] if hits else None


def merge_family_familyname(family: str, uniprot_dir: Path, ncbi_dir: Path, out_dir: Path) -> Tuple[int, int, int, Path, Path, Path]:
    uni_path = choose_best_file(uniprot_dir, family)
    ncb_path = choose_best_file(ncbi_dir, family)
    uni = read_fasta(uni_path)
    ncb = read_fasta(ncb_path)

    # Build unique-by-sequence dictionary, prefer UniProt header if duplicate between sources
    seen: Dict[str, Tuple[str, str]] = {}
    for src, recs in (("UniProt", uni), ("NCBI", ncb)):
        for h, s in recs:
            if not s:
                continue
            if s in seen:
                # Keep existing provenance list if needed
                prev_h, prev_src = seen[s]
                if src not in prev_src:
                    seen[s] = (prev_h, prev_src + f";{src}")
                continue
            acc = first_token_id(h)
            merged_header = f"{acc}|{family}|SRC:{src}"
            seen[s] = (merged_header, src)

    out_path = out_dir / f"{family}.fasta"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as w:
        for seq, (hdr, srcs) in seen.items():
            wrap_fasta_write(w, hdr, seq)

    return len(uni), len(ncb), len(seen), out_path, uni_path, ncb_path

def main():
    ap = argparse.ArgumentParser(description="Merge & dedupe per-family FASTAs from UniProt and NCBI.")
    ap.add_argument("--uniprot_dir", required=True, help="Directory with per-family FASTAs from UniProt")
    ap.add_argument("--ncbi_dir", required=True, help="Directory with per-family FASTAs from NCBI")
    ap.add_argument("--out_dir", default="families", help="Output directory for merged per-family FASTAs")
    ap.add_argument("--all_merged", action="store_true", help="Also write a global merged FASTA of all families")
    args = ap.parse_args()

    uni_dir = Path(args.uniprot_dir)
    ncb_dir = Path(args.ncbi_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Infer family names from filenames present in either directory
    def families_from_dir(d: Path) -> List[str]:
        if not d.exists():
            return []
        fams = []
        for p in d.glob("*.fasta"):
            # strip common suffixes like .ncbi.tagged or .raw to get base family name
            stem = p.stem
            # handle double extensions like name.ncbi.tagged.fasta -> stem becomes name.ncbi.tagged
            stem = re.sub(r"\.(ncbi|uniprot)(\.tagged)?$", "", stem)
            stem = re.sub(r"\.(raw|filt)$", "", stem)
            fams.append(stem)
        return fams

    families = sorted(set(families_from_dir(uni_dir)) | set(families_from_dir(ncb_dir)))
    if not families:
        print("[error] No *.fasta files found in either input directory.", file=sys.stderr)
        sys.exit(2)

    manifest_rows = []
    merged_all_path = out_dir / "_merged_all.fasta" if args.all_merged else None
    if merged_all_path:
        merged_all_path.unlink(missing_ok=True)
        merged_all_out = merged_all_path.open("w", encoding="utf-8")
    else:
        merged_all_out = None

    total_uni = total_ncb = total_unique = 0

    for fam in families:
        uni_n, ncb_n, uniq_n, out_path, uni_used, ncb_used = merge_family_familyname(fam, uni_dir, ncb_dir, out_dir)
        manifest_rows.append({
            "family": fam,
            "uniprot_raw": uni_n,
            "ncbi_raw": ncb_n,
            "unique_merged": uniq_n,
            "uniprot_file": str(uni_used) if uni_used else "",
            "ncbi_file": str(ncb_used) if ncb_used else "",
            "out_path": str(out_path)
        })
        total_uni += uni_n
        total_ncb += ncb_n
        total_unique += uniq_n

        # Append to global merged, if requested
        if merged_all_out:
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    merged_all_out.write(line)

    if merged_all_out:
        merged_all_out.close()

    # Write manifest CSV
    manifest_path = out_dir / "_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as w:
        cw = csv.DictWriter(w, fieldnames=["family", "uniprot_raw", "ncbi_raw", "unique_merged", "uniprot_file", "ncbi_file", "out_path"])
        cw.writeheader()
        cw.writerows(manifest_rows)

    # Print a short summary
    print(f"[done] families processed: {len(families)}")
    print(f"[done] UniProt raw total: {total_uni}")
    print(f"[done] NCBI raw total:    {total_ncb}")
    print(f"[done] unique merged:     {total_unique}")
    print(f"[done] manifest:          {manifest_path}")
    if merged_all_path:
        print(f"[done] global merged:     {merged_all_path}")

if __name__ == "__main__":
    main()
