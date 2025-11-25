# AF_teachers.py
# 1) Run ColabFold + extract teachers, everything under alphafold/
# python AF_teacher.py --index train.jsonl --fasta_out alphafold/train.fasta --af_out_dir alphafold/pdb  --teacher_npz_dir alphafold/npz

# regenerate .npz
# python AF_teacher.py --index train.jsonl --fasta_out alphafold/train.fasta --af_out_dir alphafold/pdb  --teacher_npz_dir alphafold/npz --skip_af

# python AF_teacher.py --index train.jsonl



import json
import subprocess
from pathlib import Path
import math
import numpy as np

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
        "--model-type", model_type,   # now 'alphafold2' by default
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
    import numpy as np
    from pathlib import Path

    pdb_dir = Path(pdb_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Expecting names like:  ID_unrelaxed_rank_001_model_1_seed_000.pdb
    pdb_files = list(pdb_dir.glob("*_rank_001_*.pdb"))

    if not pdb_files:
        print(f"[warn] No PDBs found in {pdb_dir}, expected *_rank_001_*.pdb")
        return

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

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="jsonl with id, seq")
    parser.add_argument("--fasta_out", default="alphafold/train.fasta")
    parser.add_argument("--af_out_dir", default="alphafold/pdb")
    parser.add_argument("--teacher_npz_dir", default="alphafold/npz")
    parser.add_argument("--skip_af", action="store_true",
                        help="skip running colabfold, just parse existing PDBs")
    args = parser.parse_args()

    ids, seqs = load_sequences_from_jsonl(args.index)
    write_fasta(ids, seqs, args.fasta_out)

    if not args.skip_af:
        run_colabfold(args.fasta_out, args.af_out_dir, model_type="monomer")

    extract_teacher_from_pdb_dir(args.af_out_dir, args.teacher_npz_dir)

if __name__ == "__main__":
    main()
