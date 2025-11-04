# build_viroporins_fasta.py
# Max-recall viroporin downloader that outputs clean FASTA only.
# Entry-mode: coronavirus E, influenza M2, HIV Vpu (+ ORF3a, SH, 6K, NSP4)
# Feature-mode: HCV p7, Polyomavirus VP4 (+ Picornavirus 2B); slices sub-seqs from polyproteins.
#
# Usage (PowerShell):
#   python build_viroporins_fasta.py `
#     --out data/viroporins_balanced_uniprot.fasta `
#     --dir data/viroporin_families `
#     --per_family_max 20000 `
#     --pe_max 4

import argparse, sys, time, re, requests
from pathlib import Path
import csv

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"

# ---------------- Synonyms (broad, recall-first) ----------------

ENTRY_SYNONYMS = {
    "coronavirus_E": [
        "E","envelope protein","small envelope protein","envelope small membrane protein",
        "E protein","viroporin","ion channel","pannexin-like","envelope viroporin"
    ],
    "influenza_M2": [
        "M2","Matrix protein 2","matrix protein 2","AM2","BM2","CM2",
        "ion channel protein M2","proton channel","M2 protein"
    ],
    "HIV_Vpu": [
        "Vpu","viral protein U","accessory protein Vpu","Vpu protein","ion channel","viroporin"
    ],
    # Added entry-mode families
    "coronavirus_ORF3a": [
        "ORF3a","3a","3a protein","protein 3a","SARS 3a","SARS-CoV 3a","SARS-CoV-2 3a",
        "accessory protein 3a","viroporin","ion channel"
    ],
    "paramyxo_SH": [
        "SH","small hydrophobic","small hydrophobic protein","SH protein","viroporin","ion channel"
    ],
    "alphavirus_6K": [
        "6K","6K2","protein 6K","6K protein","small hydrophobic","viroporin","ion channel"
    ],
    "rotavirus_NSP4": [
        "NSP4","nonstructural protein 4","enterotoxin","viroporin","ion channel"
    ],
}

FEATURE_SYNONYMS = {
    "HCV_p7": [
        "p7","protein 7","p7 protein","viroporin p7","p7 viroporin","p7 (viroporin)",
        "peptide p7","7b","p7 protein","p7a","p7b","ion channel p7","channel-forming p7","membrane protein p7"
    ],
    "polyoma_VP4": [
        "VP4","late protein VP4","virion-associated protein VP4","lytic protein VP4",
        "lytic protein","apoptosis-inducing protein","perforating protein","viroporin VP4",
        "agoprotein","agnoprotein","viroporin","viroporin-like","permeabilization protein"
    ],
    # Added feature-mode family
    "picorna_2B": [
        "2B","protein 2B","viroporin 2B","2B viroporin","membrane permeabilization protein",
        "ion channel 2B","permeabilization protein 2B"
    ],
}

# ---------------- Families ----------------
# mode: "entry" (whole-entry FASTA) vs "feature" (slice sub-seq, output FASTA)
FAMILIES = [
    dict(name="coronavirus_E",   mode="entry",   tax_clause="taxonomy_id:11118",                                    len_range=(50,120)),
    dict(name="influenza_M2",    mode="entry",   tax_clause="(taxonomy_id:11320 OR taxonomy_id:11520)",             len_range=(75,130)),
    dict(name="HIV_Vpu",         mode="entry",   tax_clause="taxonomy_id:11676",                                    len_range=(60,120)),
    dict(name="coronavirus_ORF3a", mode="entry", tax_clause="taxonomy_id:11118",                                    len_range=(200,340)),
    dict(name="paramyxo_SH",     mode="entry",   tax_clause="taxonomy_id:11158",                                    len_range=(40,120)),
    dict(name="alphavirus_6K",   mode="entry",   tax_clause="(organism_name:Alphavirus OR taxonomy_id:11019 OR taxonomy_id:11018)", len_range=(40,100)),
    dict(name="rotavirus_NSP4",  mode="entry",   tax_clause="taxonomy_id:10956",                                    len_range=(120,250)),

    dict(name="HCV_p7",          mode="feature", tax_clause='(organism_name:"Hepatitis C virus" OR organism_name:Hepacivirus OR organism_name:Pestivirus)', len_range=(45,120)),
    dict(name="polyoma_VP4",     mode="feature", tax_clause="taxonomy_id:10624",                                    len_range=(90,200)),
    dict(name="picorna_2B",      mode="feature", tax_clause="taxonomy_id:12059",                                    len_range=(70,180)),
]

# ---------------- Helpers ----------------

def or_terms(field, terms):
    q = []
    for t in terms:
        q.append(f'{field}:"{t}"' if any(c in t for c in ' :-') else f"{field}:{t}")
    return "(" + " OR ".join(q) + ")"

def build_entry_query(tax_clause, min_len, max_len, reviewed, pe_max,
                      synonyms, require_kw=None, allow_viroporin_text=False, require_tm=True):
    parts = [
        tax_clause,
        f"length:[{min_len} TO {max_len}]",
        "(" + " OR ".join(f"existence:{i}" for i in range(1, pe_max + 1)) + ")",
    ]
    if require_tm:
        parts.append("ft_transmem:*")
    if require_kw is True and not allow_viroporin_text:
        parts.append("keyword:KW-1273")
    elif allow_viroporin_text:
        parts.append('(keyword:KW-1273 OR viroporin OR "ion channel")')
    if reviewed is True:
        parts.append("reviewed:true")
    elif reviewed is False:
        parts.append("reviewed:false")
    # name clauses across multiple fields
    parts.append("(" + " OR ".join(
        [or_terms("gene", synonyms), or_terms("protein_name", synonyms), or_terms("id", synonyms)]
    ) + ")")
    return " AND ".join(parts)

def fetch_fasta_paged(query, out_fasta, max_records, page_size=500, sleep=0.2, max_retries=3):
    """Fetch FASTA pages; write raw FASTA; return count of entries seen (approx)."""
    out = open(out_fasta, "w", encoding="utf-8")
    params = {"query": query, "format": "fasta",
              "fields": "accession,id,protein_name,organism_name,length", "size": page_size}
    cursor = None; total = 0; sess = requests.Session(); retries = 0
    while True:
        if cursor: params["cursor"] = cursor
        r = sess.get(UNIPROT_URL, params=params, timeout=60)
        if r.status_code == 200:
            retries = 0
            text = (r.text or "").strip()
            if not text: break
            page_count = text.count("\n>") + (1 if text.startswith(">") else 0)
            out.write(text)
            if not text.endswith("\n"): out.write("\n")
            total += page_count
            print(f"[page] wrote {page_count} (total={total})")
            if max_records and total >= max_records: break
            link = r.headers.get("Link")
            if not link or 'rel="next"' not in link: break
            m = re.search(r'<([^>]+)>;\s*rel="next"', link)
            if not m: break
            next_url = m.group(1)
            m2 = re.search(r"[?&]cursor=([^&]+)", next_url)
            if not m2: break
            cursor = m2.group(1)
            time.sleep(sleep); continue
        body = (r.text or "")[:300].replace("\n"," ")
        print(f"[warn] HTTP {r.status_code}: {body}", file=sys.stderr)
        if 400 <= r.status_code < 500:
            retries += 1
            if retries >= max_retries:
                print("[error] persistent client error; aborting this query.", file=sys.stderr)
                break
            time.sleep(0.5)
        else:
            time.sleep(2.0)
    out.close(); return total

def dedupe_fasta(in_fa, out_fa, tag=None):
    """Deduplicate by exact sequence. If tag is given, append '|{tag}' to header."""
    seen = {}
    w = open(out_fa, "w", encoding="utf-8")
    with open(in_fa, "r", encoding="utf-8") as f:
        cur_id, cur_seq = None, []
        for line in f:
            line = line.rstrip()
            if not line: continue
            if line.startswith(">"):
                if cur_id is not None:
                    seq = "".join(cur_seq)
                    if seq not in seen:
                        seen[seq] = True
                        hdr = cur_id if tag is None else f"{cur_id}|{tag}"
                        w.write(f">{hdr}\n")
                        for i in range(0, len(seq), 60): w.write(seq[i:i+60] + "\n")
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line.strip())
        if cur_id is not None:
            seq = "".join(cur_seq)
            if seq not in seen:
                seen[seq] = True
                hdr = cur_id if tag is None else f"{cur_id}|{tag}"
                w.write(f">{hdr}\n")
                for i in range(0, len(seq), 60): w.write(seq[i:i+60] + "\n")
    w.close(); return len(seen)

def write_manifest_from_fasta(fasta_path, csv_path):
    """
    Parse headers like:
      >ACCESSION|FAMILY|LEVEL
      >ACCESSION|FAMILY|b-e
    and write a manifest with: accession,family,level_or_span,length
    """
    rows = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        h, seq = None, []
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if h is not None:
                    s = "".join(seq)
                    rows.append(summarize_header(h, len(s)))
                h = line[1:]  # strip '>'
                seq = []
            else:
                seq.append(line)
        if h is not None:
            s = "".join(seq)
            rows.append(summarize_header(h, len(s)))

    with open(csv_path, "w", newline="", encoding="utf-8") as w:
        cw = csv.DictWriter(w, fieldnames=["accession","family","level_or_span","length"])
        cw.writeheader()
        cw.writerows(rows)

def summarize_header(h, L):
    # expected forms: ACC|FAMILY|LEVEL   or   ACC|FAMILY|b-e
    parts = h.split("|")
    acc = parts[0] if parts else ""
    fam = parts[1] if len(parts) > 1 else ""
    lvl = parts[2] if len(parts) > 2 else ""
    return dict(accession=acc, family=fam, level_or_span=lvl, length=L)

def _parse_summary_row(h, L):
    parts = h.split("|")
    acc = parts[0] if parts else ""
    fam = parts[1] if len(parts) > 1 else ""
    lvl = parts[2] if len(parts) > 2 else ""
    return dict(accession=acc, family=fam, level_or_span=lvl, length=L)

def write_manifest_from_fasta(fasta_path, csv_path):
    rows = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        h, seq = None, []
        for line in f:
            line = line.rstrip()
            if not line: continue
            if line.startswith(">"):
                if h is not None:
                    s = "".join(seq)
                    rows.append(_parse_summary_row(h, len(s)))
                h = line[1:]
                seq = []
            else:
                seq.append(line)
        if h is not None:
            s = "".join(seq)
            rows.append(_parse_summary_row(h, len(s)))
    with open(csv_path, "w", newline="", encoding="utf-8") as w:
        cw = csv.DictWriter(w, fieldnames=["accession","family","level_or_span","length"])
        cw.writeheader(); cw.writerows(rows)

# ---------- JSON helpers for feature-mode (still outputs FASTA) ----------

def fetch_json_paged(query, size=500, sleep=0.2):
    params = {"query": query, "format": "json", "size": size}
    cursor = None; sess = requests.Session()
    while True:
        if cursor: params["cursor"] = cursor
        r = sess.get(UNIPROT_URL, params=params, timeout=60)
        if r.status_code != 200:
            print(f"[warn] JSON HTTP {r.status_code}: {(r.text or '')[:200]}", file=sys.stderr)
            if 400 <= r.status_code < 500: break
            time.sleep(1.0); continue
        data = r.json()
        for rec in data.get("results", []):
            yield rec
        link = r.headers.get("Link")
        if not link or 'rel="next"' not in link: break
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        if not m: break
        next_url = m.group(1)
        m2 = re.search(r"[?&]cursor=([^&]+)", next_url)
        if not m2: break
        cursor = m2.group(1)
        time.sleep(sleep)

def aa_subseq(seq, start, end):
    # UniProt features are 1-based inclusive
    return seq[start-1:end]

# ---------------- Entry-mode families ----------------

def attempt_entry_family(fam, per_family_max, page_size, sleep, pe_max):
    minL,maxL = fam["len_range"]
    terms = ENTRY_SYNONYMS[fam["name"]]
    levels = [
        dict(level="L1", reviewed=True,  require_kw=True,  allow_viroporin_text=False, require_tm=True),
        dict(level="L2", reviewed=None,  require_kw=True,  allow_viroporin_text=False, require_tm=True),
        dict(level="L3", reviewed=None,  require_kw=False, allow_viroporin_text=True,  require_tm=True),
        dict(level="L4", reviewed=None,  require_kw=False, allow_viroporin_text=True,  require_tm=False),
        dict(level="L5", reviewed=None,  require_kw=False, allow_viroporin_text=False, require_tm=False),
    ]
    tmp_raw = out_dir / f"{fam['name']}.raw.fasta"
    tmp_dedup = out_dir / f"{fam['name']}.fasta"
    for cfg in levels:
        q = build_entry_query(
            tax_clause=fam["tax_clause"], min_len=minL, max_len=maxL,
            reviewed=cfg["reviewed"], pe_max=pe_max,
            synonyms=terms, require_kw=cfg["require_kw"],
            allow_viroporin_text=cfg["allow_viroporin_text"],
            require_tm=cfg["require_tm"]
        )
        print("\n" + "="*70)
        print(f"[family] {fam['name']} ({cfg['level']})")
        print("[query] ", q, "\n")
        count = fetch_fasta_paged(q, str(tmp_raw), max_records=per_family_max, page_size=page_size, sleep=sleep)
        print(f"[download:{fam['name']}/{cfg['level']}] raw ~{count}")
        if count:
            # Tag headers with FAMILY|LEVEL so merged FASTA is traceable
            uniq = dedupe_fasta(str(tmp_raw), str(tmp_dedup), tag=f"{fam['name']}|{cfg['level']}")
            print(f"[dedupe:{fam['name']}] {uniq} unique sequences (via {cfg['level']})")
            return uniq, cfg["level"], str(tmp_dedup)
    print(f"[warn] {fam['name']}: no hits after entry-mode relaxation.")
    return 0, "NONE", None

# ---------------- Feature-mode families (p7 / VP4 / 2B) ----------------

def attempt_feature_family(fam, per_family_max, page_size, sleep, pe_max):
    minL,maxL = fam["len_range"]
    names = FEATURE_SYNONYMS[fam["name"]]

    # Server-side feature field clause
    ft_terms = []
    for n in names:
        nq = f"\"{n}\"" if any(c in n for c in ' :-') else n
        ft_terms += [f"ft_chain:{nq}", f"ft_region:{nq}", f"ft_peptide:{nq}", f"ft_propep:{nq}"]
    ft_clause = "(" + " OR ".join(ft_terms) + ")"

    ladders = [
        # Lf1: strict taxonomy + PE + explicit ft_* match
        [fam["tax_clause"], "(" + " OR ".join(f"existence:{i}" for i in range(1, pe_max + 1)) + ")", ft_clause],
        # Lf2: drop PE
        [fam["tax_clause"], ft_clause],
        # Lf3: global viruses
        ["taxonomy_id:10239", ft_clause],
        # Lf4: protein_name/id backstop (server-side)
        ["taxonomy_id:10239", "(" + " OR ".join([or_terms("protein_name", names), or_terms("id", names)]) + ")"],
    ]

    out_path = out_dir / f"{fam['name']}.fasta"
    total_kept = 0

    def write_slice(w, acc, start, end, sub):
        nonlocal total_kept
        L = len(sub)
        if not (minL <= L <= maxL):
            return
        # Header: >ACC|FAMILY|b-e
        w.write(f">{acc}|{fam['name']}|{start}-{end}\n")
        for i in range(0, L, 60): w.write(sub[i:i+60] + "\n")
        total_kept += 1

    # Try Lf1..Lf4
    with open(out_path, "w", encoding="utf-8") as w:
        for li, parts in enumerate(ladders, start=1):
            q = " AND ".join(parts)
            print("\n" + "="*70)
            print(f"[family] {fam['name']} (feature-mode Lf{li})")
            print("[json-query] ", q, "\n")
            for rec in fetch_json_paged(q, size=page_size, sleep=sleep):
                acc = rec.get("primaryAccession") or rec.get("accession")
                seq = (rec.get("sequence") or {}).get("value", "")
                feats = rec.get("features", []) or []
                if not seq or not feats: continue
                for ft in feats:
                    t = (ft.get("type") or "").lower()
                    if t not in {"chain","peptide","propep","region"}: continue
                    desc = (ft.get("description") or "").lower()
                    if not any(n.lower() in desc for n in names): continue
                    loc = ft.get("location") or {}
                    b = (loc.get("start") or {}).get("value"); e = (loc.get("end") or {}).get("value")
                    if not (isinstance(b,int) and isinstance(e,int) and 1 <= b <= e <= len(seq)): continue
                    sub = aa_subseq(seq, b, e)
                    write_slice(w, acc, b, e, sub)
                    if per_family_max and total_kept >= per_family_max: break
                if per_family_max and total_kept >= per_family_max: break
            if total_kept:
                print(f"[feature:{fam['name']}] sliced {total_kept} subsequences (via Lf{li})")
                break

        # Lf5: client-side sweep (taxonomy only)
        if not total_kept:
            q5 = fam["tax_clause"]
            print("\n" + "="*70)
            print(f"[family] {fam['name']} (feature-mode Lf5 CLIENT-SIDE)")
            print("[json-query] ", q5, "\n")
            for rec in fetch_json_paged(q5, size=page_size, sleep=sleep):
                acc = rec.get("primaryAccession") or rec.get("accession")
                seq = (rec.get("sequence") or {}).get("value", "")
                feats = rec.get("features", []) or []
                if not seq or not feats: continue
                for ft in feats:
                    t = (ft.get("type") or "").lower()
                    if t not in {"chain","peptide","propep","region"}: continue
                    desc = (ft.get("description") or "").lower()
                    if not any(n.lower() in desc for n in names): continue
                    loc = ft.get("location") or {}
                    b = (loc.get("start") or {}).get("value"); e = (loc.get("end") or {}).get("value")
                    if not (isinstance(b,int) and isinstance(e,int) and 1 <= b <= e <= len(seq)): continue
                    sub = aa_subseq(seq, b, e)
                    write_slice(w, acc, b, e, sub)
                    if per_family_max and total_kept >= per_family_max: break
                if per_family_max and total_kept >= per_family_max: break

        # Lf6: client-side heuristic — keep ANY feature slice whose length is in window
        if not total_kept:
            q6 = fam["tax_clause"]
            print("\n" + "="*70)
            print(f"[family] {fam['name']} (feature-mode Lf6 LEN-ONLY)")
            print("[json-query] ", q6, "\n")
            with open(out_path, "a", encoding="utf-8") as w2:
                for rec in fetch_json_paged(q6, size=page_size, sleep=sleep):
                    acc = rec.get("primaryAccession") or rec.get("accession")
                    seq = (rec.get("sequence") or {}).get("value", "")
                    feats = rec.get("features", []) or []
                    if not seq or not feats: continue
                    for ft in feats:
                        t = (ft.get("type") or "").lower()
                        if t not in {"chain","peptide","propep","region"}: continue
                        loc = ft.get("location") or {}
                        b = (loc.get("start") or {}).get("value"); e = (loc.get("end") or {}).get("value")
                        if not (isinstance(b,int) and isinstance(e,int) and 1 <= b <= e <= len(seq)): continue
                        sub = aa_subseq(seq, b, e); L = len(sub)
                        if minL <= L <= maxL:
                            w2.write(f">{acc}|{fam['name']}|{b}-{e}\n")
                            for i in range(0, L, 60): w2.write(sub[i:i+60] + "\n")
                            total_kept += 1
                        if per_family_max and total_kept >= per_family_max: break
                    if per_family_max and total_kept >= per_family_max: break

    if total_kept == 0:
        print(f"[warn] {fam['name']}: no sliced subsequences found.")
        return 0, "FEATURE", None
    return total_kept, "FEATURE", str(out_path)

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Max-recall viroporin FASTA builder (entry + feature modes).")
    ap.add_argument("--out", default="data/viroporins_balanced_uniprot.fasta", help="final merged, globally deduped FASTA")
    ap.add_argument("--dir", default="data/viroporin_families", help="directory for per-family FASTAs")
    ap.add_argument("--per_family_max", type=int, default=20000, help="cap per family after paging")
    ap.add_argument("--page_size", type=int, default=500)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--pe_max", type=int, default=4, choices=[2,3,4], help="max PE level included (1..pe_max)")
    args = ap.parse_args()

    global out_dir
    out_dir = Path(args.dir); out_dir.mkdir(parents=True, exist_ok=True)
    merged_tmp = out_dir / "_merged.tmp.fasta"
    if merged_tmp.exists(): merged_tmp.unlink()

    totals = {}; which_level = {}
    for fam in FAMILIES:
        if fam["mode"] == "entry":
            uniq, lvl, path = attempt_entry_family(fam, args.per_family_max, args.page_size, args.sleep, args.pe_max)
        else:
            uniq, lvl, path = attempt_feature_family(fam, args.per_family_max, args.page_size, args.sleep, args.pe_max)
        totals[fam["name"]] = uniq; which_level[fam["name"]] = lvl
        if path:
            with open(path, "r", encoding="utf-8") as f, open(merged_tmp, "a", encoding="utf-8") as w:
                for line in f: w.write(line)

    final_out = Path(args.out)
    final_out.parent.mkdir(parents=True, exist_ok=True)
    if merged_tmp.exists():
        total_unique = dedupe_fasta(str(merged_tmp), str(final_out))
        manifest_csv = final_out.with_suffix(".csv")
        write_manifest_from_fasta(str(final_out), str(manifest_csv))
        print(f"[manifest] wrote {manifest_csv}")
        manifest_csv = final_out.with_suffix(".csv")
        write_manifest_from_fasta(str(final_out), str(manifest_csv))
        print(f"[manifest] wrote {manifest_csv}")
        print("\n" + "="*70)
        print("[result] per-family unique (mode/level):")
        for k in totals:
            print(f"  {k:16s}: {totals[k]:5d}  ({which_level[k]})")
        print(f"[result] global unique sequences: {total_unique}")
        print(f"[done] wrote {final_out}")
        merged_tmp.unlink(missing_ok=True)
    else:
        print("[done] nothing downloaded; no output written.")

if __name__ == "__main__":
    main()
