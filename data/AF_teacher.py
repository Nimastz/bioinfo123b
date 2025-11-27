# python AF_teacher.py --index train.jsonl
# python AF_teacher.py --index train.jsonl --fasta_out alphafold/train.fasta --af_out_dir alphafold/pdb --teacher_npz_dir alphafold/npz --max_seqs 50 --chunk_size 4

import json
import subprocess
from pathlib import Path
import math
import numpy as np
import random  # NEW: for random sampling

def load_sequences_from_jsonl(index_path):
    ids = []
    seqs = []
    with open(index_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            seq = rec.get("seq") or rec.get("sequence")
            sid = rec.get("id") or rec.get("name")
            if not seq or not sid:
                raise ValueError(f"Missing seq/id in line: {line}")
            ids.append(sid)
            seqs.append(seq)
    return ids, seqs

def write_fasta(ids, seqs, fasta_path):
    fasta_path = Path(fasta_path)

    if fasta_path.parent != Path(""):
        fasta_path.parent.mkdir(parents=True, exist_ok=True)

    with fasta_path.open("w") as f:
        for sid, seq in zip(ids, seqs):
            f.write(f">{sid}\n")
            # wrap at 60 chars for readability
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")
    print(f"[info] wrote FASTA with {len(ids)} sequences → {fasta_path}")

def run_colabfold(fasta_path, out_dir, model_type="alphafold2"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "colabfold_batch",
        "--model-type", model_type,   # 'alphafold2' by default
        "--num-recycle", "3",
        "--num-models", "1",
        str(fasta_path),
        str(out_dir)
    ]
    print("[info] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[info] ColabFold finished, results in {out_dir}")

def parse_backbone_from_pdb(pdb_path):
    """
    Returns:
      res_ids: list of (chain_id, resseq, icode)
      N_list, CA_list, C_list: lists of xyz (np.array(3,))
    """
    res_map = {}  # key: (chain, resseq, icode) → dict(atom_name->xyz)
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21].strip()
            resseq = int(line[22:26])
            icode = line[26].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            if atom_name not in ("N", "CA", "C"):
                continue

            key = (chain_id, resseq, icode)
            if key not in res_map:
                res_map[key] = {"res_name": res_name}
            res_map[key][atom_name] = np.array([x, y, z], dtype=np.float32)

    # sort by (chain, resseq, icode)
    keys_sorted = sorted(res_map.keys(), key=lambda k: (k[0], k[1], k[2]))
    res_ids = []
    N_list, CA_list, C_list = [], [], []
    for key in keys_sorted:
        rec = res_map[key]
        if "N" in rec and "CA" in rec and "C" in rec:
            res_ids.append(key)
            N_list.append(rec["N"])
            CA_list.append(rec["CA"])
            C_list.append(rec["C"])

    if not CA_list:
        raise ValueError(f"No backbone found in {pdb_path}")

    N_arr = np.stack(N_list, axis=0)   # (L,3)
    CA_arr = np.stack(CA_list, axis=0)
    C_arr = np.stack(C_list, axis=0)
    return res_ids, N_arr, CA_arr, C_arr

def dihedral(p0, p1, p2, p3):
    """
    p0..p3: np.array(3,)
    returns angle in radians
    """
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so we don't scale the torsion vector
    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return math.atan2(y, x)

def compute_torsions(N, CA, C):
    """
    N, CA, C: (L,3) arrays
    returns torsions: (L,3) → [phi, psi, omega] in radians
      phi[i]   = C_{i-1}-N_i-CA_i-C_i  (i>=1)
      psi[i]   = N_i-CA_i-C_i-N_{i+1}  (i<=L-2)
      omega[i] = CA_{i-1}-C_{i-1}-N_i-CA_i (i>=1)
    For undefined ends, fill NaN.
    """
    L = N.shape[0]
    tors = np.full((L, 3), np.nan, dtype=np.float32)

    for i in range(L):
        # phi
        if i >= 1:
            tors[i, 0] = dihedral(C[i-1], N[i], CA[i], C[i])
        # psi
        if i <= L - 2:
            tors[i, 1] = dihedral(N[i], CA[i], C[i], N[i+1])
        # omega
        if i >= 1:
            tors[i, 2] = dihedral(CA[i-1], C[i-1], N[i], CA[i])
    return tors  # (L,3)

def extract_teacher_from_pdb_dir(pdb_dir, out_dir):
    """
    Scan pdb_dir for *_rank_001_*.pdb and write one .npz per sequence id
    into out_dir. This function is idempotent: re-running just overwrites
    existing .npz files.
    """
    pdb_dir = Path(pdb_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Expecting names like: ID_unrelaxed_rank_001_model_1_seed_000.pdb
    pdb_files = list(pdb_dir.glob("*_rank_001_*.pdb"))

    if not pdb_files:
        print(f"[warn] No PDBs found in {pdb_dir}, expected *_rank_001_*.pdb")
        return

    print(f"[info] extracting teachers from {len(pdb_files)} PDB files in {pdb_dir}")

    for pdb_path in pdb_files:
        name = pdb_path.name
        seq_id = name.split("_")[0]  # ID before first underscore

        try:
            res_ids, N, CA, C = parse_backbone_from_pdb(pdb_path)
        except Exception as e:
            print(f"[err] Failed to parse {pdb_path}: {e}")
            continue

        tors = compute_torsions(N, CA, C)
        xyz_ref = CA  # (L,3) CA coordinates as teacher geometry

        out_path = out_dir / f"{seq_id}.npz"
        np.savez_compressed(out_path, xyz_ref=xyz_ref, tors_ref=tors)

        print(f"[info] wrote teacher npz for {seq_id} → {out_path}")

# NEW: stratified random sampling by family
FAMILY_TAGS = [
    "HCV_p7",
    "HIV_Vpu",
    "alphavirus_6K",
    "coronavirus_E",
    "coronavirus_ORF3a",
    "influenza_M2",
    "paramyxo_SH",
    "picorna_2B",
    "polyoma_VP4",
    "rotavirus_NSP4",
]

def infer_family(seq_id: str) -> str:
    """
    Infer viroporin family from ID string by substring match.
    If none matches, return 'other'.
    """
    for tag in FAMILY_TAGS:
        if tag in seq_id:
            return tag
    return "other"

def stratified_sample_by_family(ids, seqs, max_seqs: int):
    """
    Randomly sample up to max_seqs sequences, but proportionally
    to family counts, based on FAMILY_TAGS and infer_family().
    """
    n = len(ids)
    if max_seqs <= 0 or max_seqs >= n:
        # no subsampling
        return ids, seqs

    # group indices by family
    fam_to_indices = {}
    for i, sid in enumerate(ids):
        fam = infer_family(sid)
        fam_to_indices.setdefault(fam, []).append(i)

    # total across all families
    total = sum(len(v) for v in fam_to_indices.values())
    if total == 0:
        # degenerate, just return original
        return ids, seqs

    # compute proportional quotas
    fam_quota_float = {}
    for fam, idxs in fam_to_indices.items():
        fam_quota_float[fam] = max_seqs * (len(idxs) / total)

    # floor + distribute remainder to largest fractional parts
    fam_quota = {fam: int(math.floor(q)) for fam, q in fam_quota_float.items()}
    used = sum(fam_quota.values())
    remaining = max_seqs - used

    # sort families by fractional part desc
    fam_by_frac = sorted(
        fam_quota_float.items(),
        key=lambda kv: kv[1] - math.floor(kv[1]),
        reverse=True,
    )
    for fam, _ in fam_by_frac:
        if remaining <= 0:
            break
        fam_quota[fam] += 1
        remaining -= 1

    # sample within each family
    chosen_indices = []
    for fam, idxs in fam_to_indices.items():
        if not idxs:
            continue
        k = min(fam_quota.get(fam, 0), len(idxs))
        if k <= 0:
            continue
        # random sample from this family
        chosen = random.sample(idxs, k)
        chosen_indices.extend(chosen)

    # in rare case we undershoot due to small families, top up randomly
    if len(chosen_indices) < max_seqs:
        remaining_idxs = [i for i in range(n) if i not in chosen_indices]
        extra_k = min(max_seqs - len(chosen_indices), len(remaining_idxs))
        if extra_k > 0:
            chosen_indices.extend(random.sample(remaining_idxs, extra_k))

    # sort to keep stable-ish order (not required, but nice)
    chosen_indices = sorted(chosen_indices)

    new_ids = [ids[i] for i in chosen_indices]
    new_seqs = [seqs[i] for i in chosen_indices]

    print("[info] stratified sampling by family:")
    for fam in sorted(fam_to_indices.keys()):
        n_fam = len(fam_to_indices[fam])
        n_pick = sum(1 for i in chosen_indices if infer_family(ids[i]) == fam)
        print(f"  {fam:18s}: {n_pick}/{n_fam} picked")

    print(f"[info] total sampled: {len(new_ids)} (target max_seqs={max_seqs})")
    return new_ids, new_seqs

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="jsonl with id, seq")
    parser.add_argument("--fasta_out", default="alphafold/train.fasta")
    parser.add_argument("--af_out_dir", default="alphafold/pdb")
    parser.add_argument("--teacher_npz_dir", default="alphafold/npz")
    parser.add_argument(
        "--skip_af",
        action="store_true",
        help="skip running colabfold, just parse existing PDBs"
    )
    # limit how many sequences we actually send to ColabFold
    parser.add_argument(
        "--max_seqs",
        type=int,
        default=200,
        help="Maximum number of sequences to run through ColabFold. "
             "If the index has more, they are stratified by family and randomly subsampled."
    )
    # configurable chunk size (less aggressive than fixed 5)
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=3,
        help="Number of sequences per ColabFold batch (smaller => less GPU/MSA load)."
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="0-based index into the (possibly subsampled) sequence list to start from. "
             "Useful to continue from where a previous run stopped."
    )

    args = parser.parse_args()
    
    # ---- load all sequences ----
    ids, seqs = load_sequences_from_jsonl(args.index)
    n = len(ids)
    print(f"[info] loaded {n} sequences from {args.index}")

    # stratified subsample by family if we have more sequences than max_seqs
    original_n = n
    if args.max_seqs is not None and args.max_seqs > 0 and n > args.max_seqs:
        ids, seqs = stratified_sample_by_family(ids, seqs, args.max_seqs)
        n = len(ids)
        print(f"[info] stratified subsampling: original {original_n} → {n} sequences "
              f"(max_seqs={args.max_seqs})")

    # Write a master FASTA with all sequences (for record)
    write_fasta(ids, seqs, args.fasta_out)

    # If we only want to regenerate npz from existing PDBs:
    if args.skip_af:
        print("[info] --skip_af set: skipping ColabFold, only extracting teachers")
        extract_teacher_from_pdb_dir(args.af_out_dir, args.teacher_npz_dir)
        return

    # ---- process in chunks ----
    chunk_size = max(1, args.chunk_size)
    start_idx = max(0, min(args.start_idx, n))
    if start_idx > 0:
        print(f"[info] skipping first {start_idx} sequences, starting from index {start_idx}")

    print(f"[info] using chunk_size={chunk_size}")

    for start in range(start_idx, n, chunk_size):
        end = min(n, start + chunk_size)
        batch_ids = ids[start:end]
        batch_seqs = seqs[start:end]

        attempted = end - start_idx
        total = n - start_idx if n > start_idx else 1
        frac = attempted / total

        print(f"[info] processing chunk {start}..{end-1} ({end-start} sequences) "
              f"[progress {attempted}/{total} = {frac:.1%}]")


        # PROGRESS: how many sequences attempted so far
        attempted = end
        frac = attempted / n if n > 0 else 0.0
        print(f"[info] processing chunk {start}..{end-1} ({end-start} sequences) "
              f"[progress {attempted}/{n} = {frac:.1%}]")

        # Per-chunk FASTA, e.g. alphafold/train_chunk_0000_0004.fasta
        base = Path(args.fasta_out)
        chunk_fasta = base.with_name(
            f"{base.stem}_chunk_{start:04d}_{end-1:04d}{base.suffix}"
        )

        write_fasta(batch_ids, batch_seqs, chunk_fasta)

        # Run ColabFold on this chunk
        try:
            run_colabfold(chunk_fasta, args.af_out_dir)
        except subprocess.CalledProcessError as e:
            print(f"[err] ColabFold failed on chunk {start}..{end-1}: {e}")
            print("[err] stopping early so you can keep partial results.")
            break

        # After each chunk, immediately extract teachers for all PDBs so far
        extract_teacher_from_pdb_dir(args.af_out_dir, args.teacher_npz_dir)
        print(f"[info] finished chunk {start}..{end-1}")

    # report how many teacher npz we actually have for the used ids
    teacher_dir = Path(args.teacher_npz_dir)
    teacher_count = 0
    for sid in ids:
        if (teacher_dir / f"{sid}.npz").exists():
            teacher_count += 1

    if n > 0:
        frac_teachers = teacher_count / n
    else:
        frac_teachers = 0.0

    print(f"[info] teacher coverage: {teacher_count}/{n} sequences "
          f"({frac_teachers:.1%}) have .npz teachers")

    print("[info] done. You can rerun with --skip_af to regenerate .npz from any PDBs.")

if __name__ == "__main__":
    main()
