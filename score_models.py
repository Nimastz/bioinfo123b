
#!/usr/bin/env python3
"""
Automated structure scoring:
- Input: a JSONL index of proteins (e.g., test_diverse_subset.jsonl)
- Looks for PDBs in 3 folders: results_af2/, results_esm/, results_viroporinAFMini/
- Uses AF2 as the default reference per protein (if available), else ESM, else skips.
- Computes:
    * TM-score (approx) after Kabsch superposition on CA atoms
    * lDDT-CA (CA-only, neighborhood radius 15 Å, thresholds 0.5/1/2/4 Å)
- Writes: scores.csv with one row per (protein, method).
No external deps; parses PDB directly and implements Kabsch + lDDT-CA.
"""
import os, json, math, csv, re, sys
from typing import List, Tuple, Dict, Optional

# ---------------------------
# Minimal PDB parsing (CA only)
# ---------------------------
def parse_pdb_ca_coords(pdb_path: str) -> Dict[str, List[Tuple[int, Tuple[float,float,float]]]]:
    """
    Returns: { chain_id : [(res_serial, (x,y,z)), ...] } ordered by appearance.
    Collects only ATOM lines with atom name 'CA'.
    """
    chains = {}
    if not os.path.exists(pdb_path):
        return chains
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            chain_id = line[21].strip() or "A"
            try:
                resseq = int(line[22:26])
            except:
                # fallback: serial number
                try:
                    resseq = int(line[6:11])
                except:
                    resseq = len(chains.get(chain_id, [])) + 1
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except:
                continue
            chains.setdefault(chain_id, []).append((resseq, (x,y,z)))
    return chains

def flatten_first_chain(chains: Dict[str, List[Tuple[int, Tuple[float,float,float]]]]) -> List[Tuple[float,float,float]]:
    """Pick the chain with the most residues, return list of coords in order of residue serial."""
    if not chains:
        return []
    # pick the longest chain
    best_chain = max(chains.items(), key=lambda kv: len(kv[1]))[1]
    # sort by residue serial
    best_chain_sorted = sorted(best_chain, key=lambda t: t[0])
    return [xyz for _, xyz in best_chain_sorted]

# ---------------------------
# Geometry helpers
# ---------------------------
def centroid(coords: List[Tuple[float,float,float]]) -> Tuple[float,float,float]:
    n = len(coords)
    if n == 0: return (0.0,0.0,0.0)
    sx = sum(c[0] for c in coords); sy = sum(c[1] for c in coords); sz = sum(c[2] for c in coords)
    return (sx/n, sy/n, sz/n)

def kabsch(P: List[Tuple[float,float,float]], Q: List[Tuple[float,float,float]]) -> Tuple[List[Tuple[float,float,float]], float]:
    """
    Kabsch alignment: Superimpose P (model) onto Q (reference).
    Returns (P_aligned, rmsd).
    """
    import numpy as np
    Pm = np.array(P, dtype=float); Qm = np.array(Q, dtype=float)
    n = min(Pm.shape[0], Qm.shape[0])
    Pm = Pm[:n]; Qm = Qm[:n]
    Pc = Pm - Pm.mean(axis=0)
    Qc = Qm - Qm.mean(axis=0)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = (np.linalg.det(V @ Wt) < 0.0)
    if d:
        V[:, -1] *= -1.0
    U = V @ Wt
    P_aligned = Pc @ U + Qm.mean(axis=0)
    diff = P_aligned - Qm[:n]
    rmsd = float(np.sqrt((diff * diff).sum() / n))
    return P_aligned.tolist(), rmsd

def tm_score(P_aligned: List[Tuple[float,float,float]], Q: List[Tuple[float,float,float]]) -> float:
    """
    Approximate TM-score with 1:1 residue correspondence after Kabsch.
    """
    import numpy as np
    Pa = np.array(P_aligned, dtype=float); Qm = np.array(Q, dtype=float)
    n = min(Pa.shape[0], Qm.shape[0])
    if n == 0: return 0.0
    Pa = Pa[:n]; Qm = Qm[:n]
    L = float(n)
    d0 = 1.24 * (L - 15.0)**(1.0/3.0) - 1.8
    d0 = max(d0, 0.5)
    d2 = ((Pa - Qm)**2).sum(axis=1)
    score = float(((1.0 / (1.0 + (d2 / (d0*d0))))).sum() / L)
    return score

def lddt_ca(P_aligned: List[Tuple[float,float,float]], Q: List[Tuple[float,float,float]], cutoff: float=15.0) -> float:
    """
    CA-only lDDT (approx):
    For each residue i in reference (Q), consider neighbors j != i with ||Qj-Qi|| <= 15 Å.
    Compare | ||Pa_j - Pa_i|| - ||Qj - Qi|| | against thresholds 0.5,1,2,4 Å.
    Return mean fraction across residues.
    """
    import numpy as np
    Qa = np.array(Q, dtype=float); Pa = np.array(P_aligned, dtype=float)
    n = min(Qa.shape[0], Pa.shape[0])
    if n < 3: return 0.0
    Qa = Qa[:n]; Pa = Pa[:n]
    # pairwise distances
    dQ = np.sqrt(np.sum((Qa[:,None,:]-Qa[None,:,:])**2, axis=2))
    dP = np.sqrt(np.sum((Pa[:,None,:]-Pa[None,:,:])**2, axis=2))
    thresholds = [0.5, 1.0, 2.0, 4.0]
    per_res = []
    for i in range(n):
        nbrs = [j for j in range(n) if j != i and dQ[i,j] <= cutoff]
        if not nbrs:
            continue
        diffs = [abs(dP[i,j] - dQ[i,j]) for j in nbrs]
        hits = []
        for t in thresholds:
            good = sum(1 for d in diffs if d <= t) / len(nbrs)
            hits.append(good)
        per_res.append(sum(hits)/len(hits))
    if not per_res: 
        return 0.0
    return float(sum(per_res)/len(per_res))

# ---------------------------
# File discovery
# ---------------------------
def find_pdb_for_id(root: str, prot_id: str) -> Optional[str]:
    """
    Try to find a PDB file under 'root' whose filename contains the prot_id (sanitized).
    """
    if not os.path.isdir(root):
        return None
    # sanitize id for matching
    needle = re.sub(r'[^A-Za-z0-9_.-]+', '_', prot_id)
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.lower().endswith((".pdb",".cif",".mmcif")):
                continue
            if needle in fn:
                return os.path.join(dirpath, fn)
    return None

# ---------------------------
# Main
# ---------------------------
def main(index_jsonl: str,
         dir_af2: str = "results_af2",
         dir_esm: str = "results_esm",
         dir_vpm: str = "results_viroporinAFMini",
         out_csv: str = "scores.csv"):
    rows = []
    with open(index_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            rows.append(json.loads(line))

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    # CSV header
    header = ["id","family","ref","method","L_ca","RMSD","TMscore","lDDT_CA","ref_pdb","pred_pdb"]
    out = open(out_csv, "w", newline="", encoding="utf-8")
    w = csv.writer(out)
    w.writerow(header)

    for r in rows:
        pid = r.get("id") or r.get("name") or r.get("title") or "unknown"
        fam = r.get("family", "unknown")
        # find reference: AF2 preferred, else ESM
        ref_pdb = find_pdb_for_id(dir_af2, pid)
        ref_name = "AF2"
        if not ref_pdb:
            ref_pdb = find_pdb_for_id(dir_esm, pid)
            ref_name = "ESMFold"
        if not ref_pdb:
            # skip if no reference
            print(f"[skip] No reference PDB for {pid}")
            continue

        # load reference coords
        ref_chains = parse_pdb_ca_coords(ref_pdb)
        ref_ca = flatten_first_chain(ref_chains)
        if len(ref_ca) < 5:
            print(f"[skip] Too few CA atoms in reference for {pid}")
            continue

        # methods to score
        methods = [
            ("ESMFold", find_pdb_for_id(dir_esm, pid)),
            ("ViroporinAFMini", find_pdb_for_id(dir_vpm, pid)),
        ]
        # also include AF2 self-score (baseline, should be 1.0 TM, high lDDT)
        methods.insert(0, ("AF2", ref_pdb))

        for mname, mpdb in methods:
            if not mpdb:
                continue
            pred_chains = parse_pdb_ca_coords(mpdb)
            pred_ca = flatten_first_chain(pred_chains)
            if len(pred_ca) < 5:
                continue
            # Kabsch align pred to reference (CA order)
            Paligned, rmsd = kabsch(pred_ca, ref_ca)
            tms = tm_score(Paligned, ref_ca)
            lddt = lddt_ca(Paligned, ref_ca)
            L = min(len(pred_ca), len(ref_ca))
            w.writerow([pid, fam, ref_name, mname, L, f"{rmsd:.3f}", f"{tms:.3f}", f"{lddt:.3f}", ref_pdb, mpdb])

    out.close()
    print(f"[done] Wrote {out_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python score_models.py <index_jsonl> [results_af2] [results_esm] [results_viroporinAFMini] [out_csv]")
        sys.exit(1)
    index = sys.argv[1]
    dir_af2 = sys.argv[2] if len(sys.argv) > 2 else "results_af2"
    dir_esm = sys.argv[3] if len(sys.argv) > 3 else "results_esm"
    dir_vpm = sys.argv[4] if len(sys.argv) > 4 else "results_viroporinAFMini"
    out_csv = sys.argv[5] if len(sys.argv) > 5 else "scores.csv"
    main(index, dir_af2, dir_esm, dir_vpm, out_csv)
