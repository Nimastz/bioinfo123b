# scripts/build_viroporins_ncbi_fasta.py
# High-recall sequence downloader for collecting viroporin protein sequences from NCBI.
# - Queries NCBI protein databases using E-utilities (esearch, efetch) for multiple viral families.
# - Builds search terms with organism tax IDs, name/synonym filters, and sequence length windows.
# - Downloads, deduplicates, and merges FASTA files for each family into one master FASTA output.
# - Supports config-driven execution (config.yaml) specifying email, API key, and output paths.
# Used to assemble a comprehensive FASTA dataset of viral membrane proteins for model training.
#
# Usage:
#   python build_viroporins_ncbi_fasta.py `
#     --config config.yaml
#
# Or explicitly:
#   python build_viroporins_ncbi_fasta.py `
#     --out data/viroporins_ncbi.fasta `
#     --dir data/viroporin_families_ncbi `
#     --entrez_email you@example.com `
#     --ncbi_retmax 200000 `
#     --ncbi_api_key YOUR_KEY   # optional




import argparse, sys, time, re, requests
from pathlib import Path
import yaml

# ---------- NCBI endpoints ----------
NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# ---------- Families & synonyms ----------
FAMILIES = [
    dict(name="coronavirus_E",     mode="entry",   txids=[11118],               len_range=(50, 120)),
    dict(name="influenza_M2",      mode="entry",   txids=[11320, 11520],        len_range=(65, 140)),
    dict(name="HIV_Vpu",           mode="entry",   txids=[11676],               len_range=(60, 120)),
    dict(name="coronavirus_ORF3a", mode="entry",   txids=[11118],               len_range=(180, 360)),
    dict(name="paramyxo_SH",       mode="entry",   txids=[11158],               len_range=(40, 120)),
    dict(name="alphavirus_6K",     mode="entry",   txids=[11019,11018],         len_range=(40, 100)),
    dict(name="rotavirus_NSP4",    mode="entry",   txids=[10956],               len_range=(120, 250)),
    dict(name="HCV_p7",            mode="feature", txids=[11103],               len_range=(45, 120)),
    dict(name="polyoma_VP4",       mode="feature", txids=[10624],               len_range=(90, 200)),
    dict(name="picorna_2B",        mode="feature", txids=[12059],               len_range=(70, 180)),
]

ENTRY_SYNONYMS = {
    "coronavirus_E": ["E","envelope protein","small envelope protein","envelope small membrane protein",
                      "E protein","viroporin","ion channel","pannexin-like","envelope viroporin"],
    "influenza_M2": ["M2","Matrix protein 2","matrix protein 2","AM2","BM2","CM2",
                     "ion channel protein M2","proton channel","M2 protein"],
    "HIV_Vpu": ["Vpu","viral protein U","accessory protein Vpu","Vpu protein","ion channel","viroporin"],
    "coronavirus_ORF3a": ["ORF3a","3a","3a protein","protein 3a","SARS 3a","SARS-CoV 3a","SARS-CoV-2 3a",
                          "accessory protein 3a","viroporin","ion channel"],
    "paramyxo_SH": ["SH","small hydrophobic","small hydrophobic protein","SH protein","viroporin","ion channel"],
    "alphavirus_6K": ["6K","6K2","protein 6K","6K protein","small hydrophobic","viroporin","ion channel"],
    "rotavirus_NSP4": ["NSP4","nonstructural protein 4","enterotoxin","viroporin","ion channel"],
}

FEATURE_SYNONYMS = {
    "HCV_p7": ["p7","protein 7","p7 protein","viroporin p7","p7 viroporin","p7 (viroporin)",
               "peptide p7","7b","p7 protein","p7a","p7b","ion channel p7","channel-forming p7","membrane protein p7"],
    "polyoma_VP4": ["VP4","late protein VP4","virion-associated protein VP4","lytic protein VP4",
                    "lytic protein","apoptosis-inducing protein","perforating protein","viroporin VP4",
                    "agoprotein","agnoprotein","viroporin","viroporin-like","permeabilization protein"],
    "picorna_2B": ["2B","protein 2B","viroporin 2B","2B viroporin","membrane permeabilization protein",
                   "ion channel 2B","permeabilization protein 2B"],
}

# ---------- Config ----------
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---------- Term builder ----------
def _quote_ncbi(t):
    return f'"{t}"' if any(c in t for c in " -:/()[]") else t

def build_ncbi_term(fam):
    """
    (txid…[Organism:exp]) AND (synonyms[All Fields]) AND (Lmin:Lmax[SLEN])
    """
    Lmin, Lmax = fam["len_range"]
    tx_filters = [f"txid{tx}[Organism:exp]" for tx in fam.get("txids", [])]
    org_block = "(" + " OR ".join(tx_filters) + ")" if tx_filters else "txid10239[Organism:exp]"

    raw_syn = ENTRY_SYNONYMS.get(fam["name"]) if fam["mode"] == "entry" else FEATURE_SYNONYMS.get(fam["name"])
    syn = []
    seen = set()
    for s in (raw_syn or []):
        s = s.strip()
        if not s:
            continue
        # keep short tokens if they contain a digit (e.g., M2), skip bare "E"
        if len(s) < 3 and (" " not in s) and not any(c.isdigit() for c in s):
            continue
        k = s.lower()
        if k in seen: continue
        seen.add(k)
        syn.append(s)
    if not syn:
        syn = ["viroporin"]

    name_block = "(" + " OR ".join(f"{_quote_ncbi(s)}[All Fields]" for s in syn) + ")"
    slen_block = f"{Lmin}:{Lmax}[SLEN]"
    return f"({org_block}) AND {name_block} AND {slen_block}"

# ---------- E-utilities (robust) ----------
def esearch_with_history(term, email=None, api_key=None, sleep=0.34):
    params = {"db":"protein","retmode":"json","term":term,"usehistory":"y","retmax":"0"}
    if email:   params["email"] = email
    if api_key: params["api_key"] = api_key
    r = requests.get(NCBI_ESEARCH, params=params, timeout=60)
    r.raise_for_status()
    js = r.json().get("esearchresult", {})
    count = int(js.get("count", "0"))
    webenv = js.get("webenv", "")
    qk = js.get("querykey", "")
    return count, webenv, qk

def esearch_all_ids(term, email=None, api_key=None, retmax=200000, sleep=0.34):
    ids = []
    retstart = 0
    while True:
        params = {
            "db":"protein","retmode":"json","term":term,
            "retmax":str(min(100000, retmax - len(ids))), "retstart":str(retstart)
        }
        if email:   params["email"] = email
        if api_key: params["api_key"] = api_key
        r = requests.get(NCBI_ESEARCH, params=params, timeout=60)
        r.raise_for_status()
        js = r.json().get("esearchresult", {})
        idlist = js.get("idlist", []) or []
        ids.extend(idlist)
        count = int(js.get("count", "0"))
        retstart += len(idlist)
        if retstart >= count or len(ids) >= retmax or not idlist:
            break
        time.sleep(sleep)
    return ids[:retmax]

def efetch_fasta(ids, out_path, email=None, api_key=None, sleep=0.34, batch_size=500):
    """POST efetch in safe chunks to avoid 414."""
    if not ids: return 0
    total = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for i in range(0, len(ids), batch_size):
            chunk = ids[i:i+batch_size]
            data = {"db":"protein","rettype":"fasta","retmode":"text","id":",".join(chunk)}
            if email:   data["email"] = email
            if api_key: data["api_key"] = api_key
            r = requests.post(NCBI_EFETCH, data=data, timeout=180)
            if r.status_code != 200:
                print(f"[warn] efetch {r.status_code}: {r.text[:200]}", file=sys.stderr)
                time.sleep(sleep); continue
            text = (r.text or "").strip()
            if not text: time.sleep(sleep); continue
            cnt = text.count("\n>") + (1 if text.startswith(">") else 0)
            w.write(text); 
            if not text.endswith("\n"): w.write("\n")
            total += cnt
            print(f"[ncbi-efetch] wrote {cnt} (total={total})")
            time.sleep(sleep)
    return total

def efetch_history_fasta(out_path, webenv, query_key, email=None, api_key=None, sleep=0.34, page=10000, max_count=None):
    """Page efetch using history handle (no long URLs)."""
    if not webenv or not query_key: return 0
    total = 0
    retstart = 0
    with open(out_path, "w", encoding="utf-8") as w:
        while True:
            if max_count is not None and retstart >= max_count: break
            retmax = page if max_count is None else min(page, max_count - retstart)
            data = {
                "db":"protein","rettype":"fasta","retmode":"text",
                "query_key":str(query_key),"WebEnv":webenv,
                "retstart":str(retstart),"retmax":str(retmax)
            }
            if email:   data["email"] = email
            if api_key: data["api_key"] = api_key
            r = requests.post(NCBI_EFETCH, data=data, timeout=180)
            if r.status_code != 200:
                print(f"[warn] efetch(hist) {r.status_code}: {r.text[:200]}", file=sys.stderr)
                time.sleep(sleep); break
            text = (r.text or "").strip()
            if not text: break
            cnt = text.count("\n>") + (1 if text.startswith(">") else 0)
            w.write(text); 
            if not text.endswith("\n"): w.write("\n")
            total += cnt
            retstart += retmax
            print(f"[ncbi-efetch-hist] wrote {cnt} (total={total})")
            time.sleep(sleep)
    return total

# ---------- FASTA utils ----------
def parse_fasta_stream(lines):
    h, seq = None, []
    for line in lines:
        line = line.rstrip()
        if not line: continue
        if line.startswith(">"):
            if h is not None: yield h, "".join(seq)
            h = line[1:]; seq = []
        else:
            seq.append(line)
    if h is not None: yield h, "".join(seq)

def dedupe_and_write(in_fa, out_fa, tag):
    seen, kept = set(), 0
    with open(in_fa, "r", encoding="utf-8") as f, open(out_fa, "w", encoding="utf-8") as w:
        for h, s in parse_fasta_stream(f):
            if not s: continue
            if s in seen: continue
            seen.add(s)
            acc = h.split()[0]  # first token (YP_, WP_, etc.)
            w.write(f">{acc}|{tag}\n")
            for i in range(0, len(s), 60): w.write(s[i:i+60] + "\n")
            kept += 1
    return kept

def dedupe_global(in_fa, out_fa):
    seen, kept = set(), 0
    with open(in_fa, "r", encoding="utf-8") as f, open(out_fa, "w", encoding="utf-8") as w:
        for h, s in parse_fasta_stream(f):
            if not s or s in seen: continue
            seen.add(s)
            w.write(f">{h}\n")
            for i in range(0, len(s), 60): w.write(s[i:i+60] + "\n")
            kept += 1
    return kept

def length_filter_fasta(in_fa, out_fa, minL, maxL):
    kept = 0
    with open(in_fa, "r", encoding="utf-8") as f, open(out_fa, "w", encoding="utf-8") as w:
        for h, s in parse_fasta_stream(f):
            L = len(s)
            if minL <= L <= maxL:
                w.write(f">{h}\n")
                for i in range(0, L, 60): w.write(s[i:i+60] + "\n")
                kept += 1
    return kept

# ---------- Per-family ----------
def fetch_family_ncbi(fam, out_dir, retmax, email, api_key):
    term = build_ncbi_term(fam)
    print("\n" + "="*70)
    print(f"[family] {fam['name']} (NCBI)")
    print(f"[term]   {term}\n")

    # Prefer history for big sets, else explicit ID list
    try:
        count, webenv, qk = esearch_with_history(term, email=email, api_key=api_key)
    except Exception as e:
        print(f"[error] esearch failed: {e}", file=sys.stderr)
        count, webenv, qk = 0, "", ""

    raw_path = out_dir / f"{fam['name']}.ncbi.raw.fasta"
    fam_out  = out_dir / f"{fam['name']}.fasta"

    if count == 0:
        print(f"[ncbi:{fam['name']}] 0 records")
        return 0, None

    max_fetch = min(count, retmax)
    if count > 5000:
        print(f"[ncbi:{fam['name']}] using history paging, count={count}, max={max_fetch}")
        n = efetch_history_fasta(str(raw_path), webenv, qk, email=email, api_key=api_key, page=10000, max_count=max_fetch)
    else:
        print(f"[ncbi:{fam['name']}] fetching explicit IDs, count={count}, max={max_fetch}")
        ids = esearch_all_ids(term, email=email, api_key=api_key, retmax=max_fetch)
        n = efetch_fasta(ids, str(raw_path), email=email, api_key=api_key, batch_size=500)

    if n == 0:
        print(f"[ncbi:{fam['name']}] efetch returned 0")
        return 0, None

    # tag and dedupe per-family
    tagged = out_dir / f"{fam['name']}.ncbi.tagged.fasta"
    kept = dedupe_and_write(str(raw_path), str(tagged), tag=f"{fam['name']}|NCBI")

    # For feature-mode, enforce length window locally as a safeguard
    if fam["mode"] == "feature":
        Lmin, Lmax = fam["len_range"]
        filt = out_dir / f"{fam['name']}.filt.fasta"
        k2 = length_filter_fasta(str(tagged), str(filt), Lmin, Lmax)
        if k2:
            Path(fam_out).write_text(Path(filt).read_text(encoding="utf-8"), encoding="utf-8")
            kept = k2
        else:
            kept = 0
    else:
        Path(fam_out).write_text(Path(tagged).read_text(encoding="utf-8"), encoding="utf-8")

    print(f"[ncbi:{fam['name']}] unique {kept}")
    return kept, str(fam_out)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="NCBI-only viroporin FASTA builder (max recall).")
    ap.add_argument("--config", help="YAML config file with default arguments")
    ap.add_argument("--out", help="final merged, globally deduped FASTA")
    ap.add_argument("--dir", help="directory for per-family FASTAs")
    ap.add_argument("--entrez_email", help="your email for NCBI E-utilities")
    ap.add_argument("--ncbi_api_key", default="", help="NCBI API key (optional)")
    ap.add_argument("--ncbi_retmax", type=int, default=200000, help="max NCBI IDs per family to retrieve")
    args = ap.parse_args()

    # Load config (optional)
    if args.config:
        cfg = load_config(args.config)
        for k, v in (cfg or {}).items():
            if getattr(args, k, None) in (None, "", 0):
                setattr(args, k, v)

    # Basic checks
    if not args.entrez_email:
        print("[error] --entrez_email is required (or set it in --config)", file=sys.stderr)
        sys.exit(2)
    if not args.out: args.out = "data/viroporins_ncbi.fasta"
    if not args.dir: args.dir = "data/viroporin_families_ncbi"

    out_dir = Path(args.dir); out_dir.mkdir(parents=True, exist_ok=True)
    merged_tmp = out_dir / "_merged.tmp.fasta"
    if merged_tmp.exists(): merged_tmp.unlink()

    totals = {}
    for fam in FAMILIES:
        kept, fam_path = fetch_family_ncbi(fam, out_dir, args.ncbi_retmax, args.entrez_email, args.ncbi_api_key)
        totals[fam["name"]] = kept
        if kept and fam_path:
            with open(fam_path, "r", encoding="utf-8") as f, open(merged_tmp, "a", encoding="utf-8") as w:
                for line in f: w.write(line)

    final_out = Path(args.out)
    final_out.parent.mkdir(parents=True, exist_ok=True)
    if merged_tmp.exists():
        total_unique = dedupe_global(str(merged_tmp), str(final_out))
        print("\n" + "="*70)
        print("[result] per-family unique (NCBI):")
        for k in totals:
            print(f"  {k:16s}: {totals[k]:5d}")
        print(f"[result] global unique sequences: {total_unique}")
        print(f"[done] wrote {final_out}")
        merged_tmp.unlink(missing_ok=True)
    else:
        print("[done] nothing downloaded; no output written.")

if __name__ == "__main__":
    main()
