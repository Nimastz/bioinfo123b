# test.py
# Evaluation script for the ViroporinAFMini model.
# Loads a trained checkpoint and runs it on the validation dataset to measure performance.
# - load_latest_ckpt(): finds the most recent saved model checkpoint.
# - eval_one(): computes all relevant losses (distogram, torsion, FAPE, and membrane/pore priors)
#   for a single sample without updating model weights.
# - main(): loads configuration, data, and model; evaluates all validation samples; and
#   prints averaged loss metrics as JSON output.
# Used to assess how well the trained model predicts 3D viroporin structures after training.
# try : python test.py --config configs/recommended.yaml --ckpt checkpoints/step_22300.pt --index data/test_diverse_subset.jsonl --dump_dir test_results_22300 --dump_n 5 --dump_mode random --dump_olig


import argparse, yaml, torch, glob, os, json
import yaml
from src.geometry.assembly import pore_radius_profile_ca
from torch.utils.data import DataLoader
from src.data.dataset import JsonlSet, collate
from src.model.viroporin_net import ViroporinAFMini
from src.losses.distogram import distogram_loss
from src.losses.fape import fape_loss
from src.losses.torsion import torsion_l2
from src.geometry.assembly import assemble_cn
from src.losses.viroporin_priors import (
    membrane_z_mask, membrane_slab_loss, interface_contact_loss, ca_clash_loss, pore_target_loss
)
import time, random
from pathlib import Path
import torch.nn.functional as F

# Ideal backbone geometry (Angstroms)
BOND_N_CA = 1.46
BOND_CA_C = 1.53
BOND_C_O  = 1.24

AA3 = {
    "A":"ALA","C":"CYS","D":"ASP","E":"GLU","F":"PHE",
    "G":"GLY","H":"HIS","I":"ILE","K":"LYS","L":"LEU",
    "M":"MET","N":"ASN","P":"PRO","Q":"GLN","R":"ARG",
    "S":"SER","T":"THR","V":"VAL","W":"TRP","Y":"TYR","X":"GLY"
}
IDX_TO_AA = None  
PAD_IDX = 20

def seq_from_indices(seq_idx_tensor):
    import torch
    t = seq_idx_tensor.squeeze(0) if seq_idx_tensor.dim() == 2 else seq_idx_tensor
    letters = []
    L = int(t.shape[0])
    for i in range(L):
        idx = int(t[i].item())
        if idx == PAD_IDX or idx < 0 or idx >= len(IDX_TO_AA):
            letters.append("X")
        else:
            letters.append(IDX_TO_AA[idx])
    return "".join(letters)

def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in (s or ""))

def write_pdb_model_multichain(fh, backbones, seq, chain_ids, model_idx=1):
    """
    Write a single MODEL with multiple chains, each having full backbone (N, CA, C, O).

    backbones: list of backbone dicts, each with keys "N","CA","C","O", shape (L,3)
               e.g. [backbone_chainA, backbone_chainB, ...]
    seq:       1-letter monomer sequence (same for each chain here)
    chain_ids: list of chain IDs like ["A","B","C","D"]
    """
    fh.write(f"MODEL        {model_idx}\n")
    serial = 1
    for backbone, cid in zip(backbones, chain_ids):
        serial, _ = write_backbone_pdb(
            fh,
            backbone,
            seq,
            chain_id=cid,
            serial_start=serial,
            resid_start=1,
        )
        fh.write("TER\n")
    fh.write("ENDMDL\n")

def load_latest_ckpt(ckpt_dir: str):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")), key=os.path.getmtime)
    if not paths:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    return paths[-1]

def eval_one(cfg, model, batch, device):
    model.eval()
    with torch.no_grad():  # run eval in full fp32 to avoid AMP under/overflows
        torch.set_float32_matmul_precision("high")

        out = model(batch["seq_idx"].to(device), batch.get("emb"))
        xyz_raw  = out["xyz"]
        dist_raw = out["dist"]
        tors_raw = out["tors"]

        # quick debug stats on raw coords
        mx = torch.nan_to_num(xyz_raw).abs().max().item()
        sd = torch.nan_to_num(xyz_raw).std().item()
        print(f"[dbg] xyz_raw.abs().max={mx:.2f}, xyz_raw.std={sd:.2f}")

        L = xyz_raw.shape[0]

        loss_dist = distogram_loss(
        dist_raw,
        xyz_raw,
        n_bins=dist_raw.shape[-1],   # <- make loss use real #bins
        )
        loss_tors = torsion_l2(tors_raw)
        loss_fape = fape_loss(xyz_raw)

        total = (
            cfg["loss_weights"]["distogram"] * loss_dist
            + cfg["loss_weights"]["torsion"]   * loss_tors
            + cfg["loss_weights"]["fape"]      * loss_fape
        )
        logs = dict(
            distogram=float(loss_dist),
            torsion=float(loss_tors),
            fape=float(loss_fape),
        )

        if cfg["priors"].get("use_cn", True):
            n  = cfg["priors"]["n_copies"]
            rr = cfg["priors"]["ring_radius"]
            olig = assemble_cn(xyz_raw, n_copies=n, ring_radius=rr)

            tm_mask = membrane_z_mask(L, cfg["priors"]["tm_span"]).to(xyz_raw.device)
            tm_frac = float(tm_mask.float().mean().item())

            xyz = xyz_raw.clone()
            z = xyz[:, 2]
            z_tm = z[tm_mask > 0.5]
            z_center = (z_tm.median() if z_tm.numel() > 0 else z.median())
            if not torch.isfinite(z_center):
                z_center = torch.tensor(0.0, device=z.device)
            xyz[:, 2] -= z_center.detach()

            mem   = membrane_slab_loss(xyz, tm_mask)
            intf  = interface_contact_loss(olig, cutoff=9.0)
            clash = ca_clash_loss(olig, min_dist=3.6)
            pore  = pore_target_loss(olig, target_A=cfg["priors"]["pore_target_A"])

            zs, rs = pore_radius_profile_ca(olig)
            if rs.numel() > 0:
                finite_rs = rs[torch.isfinite(rs)]
                pore_minA = float(finite_rs.min().item()) if finite_rs.numel() > 0 else float("nan")
            else:
                pore_minA = float("nan")

            logs.update(
                mem=float(mem),
                intf=float(intf),
                clash=float(clash),
                pore=float(pore),
                tm_frac=float(tm_frac),
                pore_minA=float(pore_minA),
            )

            total = total + cfg["loss_weights"]["priors"] * (0.3*mem + 0.4*intf + 0.3*pore) + 0.1*clash

        return float(total), logs

def check_checkpoint_sanity(ckpt_path: str, max_abs_value: float = 1e5):
    bad = []
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    print(f"[check] Scanning checkpoint '{ckpt_path}' for NaN/Inf/huge values...")
    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            continue
        if not torch.isfinite(tensor).all():
            bad.append((name, "NaN/Inf"))
        else:
            m = tensor.abs().max().item()
            if m > max_abs_value:
                bad.append((name, f"too large (>{max_abs_value:g})"))
    if bad:
        print("[warn] Found issues:")
        for n, msg in bad[:10]:
            print(f"  - {n}: {msg}")
        if len(bad) > 10:
            print(f"  ... and {len(bad)-10} more.")
        return False
    print("[ok] All checkpoint tensors finite and reasonable.")
    return True

def reconstruct_backbone_from_ca(xyz_ca: torch.Tensor):
        """
        Very simple CA -> (N, CA, C, O) reconstruction.
        xyz_ca: (L, 3) tensor in Å.
        Returns: dict with "N", "CA", "C", "O" each (L, 3).
        NOTE: This is a crude approximation for visualization, not
            a chemically perfect reconstruction.
        """
        xyz_ca = xyz_ca.clone()
        L = xyz_ca.shape[0]

        # --- approximate local backbone direction (tangent) ---
        # forward: CA[i+1] - CA[i], backward: CA[i] - CA[i-1]
        forward = torch.zeros_like(xyz_ca)
        backward = torch.zeros_like(xyz_ca)

        forward[:-1] = xyz_ca[1:] - xyz_ca[:-1]
        backward[1:] = xyz_ca[1:] - xyz_ca[:-1]

        # average the two to get a smoother tangent
        tangent = forward + backward
        tangent = F.normalize(tangent, dim=-1, eps=1e-8)  # (L,3)

        # --- arbitrary "up" vector to define a plane for placing O ---
        # We'll use a fixed global up, then orthogonalize to tangent
        up_global = torch.tensor([0.0, 0.0, 1.0], device=xyz_ca.device)
        up = up_global.expand_as(xyz_ca)
        # remove component along tangent
        up = up - (up * tangent).sum(-1, keepdim=True) * tangent
        up = F.normalize(up, dim=-1, eps=1e-8)

        # --- place backbone atoms relative to CA ---
        # Place C slightly "forward" from CA; N slightly "backward"
        C  = xyz_ca + BOND_CA_C * tangent
        N  = xyz_ca - BOND_N_CA  * tangent

        # Place O offset from C in the (up) direction
        O  = C + BOND_C_O * up

        backbone = {
            "N":  N,
            "CA": xyz_ca,
            "C":  C,
            "O":  O,
        }
        return backbone

def write_backbone_pdb(fh, backbone, seq, chain_id="A",
                       serial_start=1, resid_start=1):
    """
    Write N, CA, C, O for each residue.
    backbone: dict "N","CA","C","O" each (L,3) tensor
    seq: 1-letter sequence string
    """
    serial = serial_start
    resid  = resid_start
    L = backbone["CA"].shape[0]

    for i in range(L):
        aa1 = seq[i] if i < len(seq) else "X"
        resname = AA3.get(aa1, "GLY")
        xN,  yN,  zN  = backbone["N"][i].tolist()
        xCA, yCA, zCA = backbone["CA"][i].tolist()
        xC,  yC,  zC  = backbone["C"][i].tolist()
        xO,  yO,  zO  = backbone["O"][i].tolist()

        # N
        fh.write(
            f"ATOM  {serial:5d}  N   {resname:>3s} {chain_id}{resid:4d}    "
            f"{xN:8.3f}{yN:8.3f}{zN:8.3f}  1.00 20.00           N\n"
        ); serial += 1

        # CA
        fh.write(
            f"ATOM  {serial:5d}  CA  {resname:>3s} {chain_id}{resid:4d}    "
            f"{xCA:8.3f}{yCA:8.3f}{zCA:8.3f}  1.00 20.00           C\n"
        ); serial += 1

        # C
        fh.write(
            f"ATOM  {serial:5d}  C   {resname:>3s} {chain_id}{resid:4d}    "
            f"{xC:8.3f}{yC:8.3f}{zC:8.3f}  1.00 20.00           C\n"
        ); serial += 1

        # O
        fh.write(
            f"ATOM  {serial:5d}  O   {resname:>3s} {chain_id}{resid:4d}    "
            f"{xO:8.3f}{yO:8.3f}{zO:8.3f}  1.00 20.00           O\n"
        ); serial += 1

        resid += 1

    return serial, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="path to step_*.pt (defaults to latest)")
    ap.add_argument("--dump_dir", default=None, help="Directory to dump per-sample outputs (JSONL/PDB)")
    ap.add_argument("--dump_n", type=int, default=3, help="How many 3D examples to dump")
    ap.add_argument("--dump_mode", default="first", choices=["first","random","hard","all"],help="Which samples to dump as PDBs")
    ap.add_argument("--dump_olig", action="store_true", help="Also dump CN-assembled oligomer PDBs")
    ap.add_argument("--index", default=None, help="JSONL to evaluate (overrides cfg['data']['val_index'])")
    args = ap.parse_args()
    root_results = Path("results")
    root_results.mkdir(exist_ok=True)
    

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    index_path = args.index if args.index else cfg["data"]["val_index"]

    # data
    ds = JsonlSet(index_path, cfg["data"].get("max_len"))
    
    # ---- sync alphabet with dataset/tokenizer ----
    global IDX_TO_AA, PAD_IDX
    IDX_TO_AA = getattr(ds, "alphabet", None)
    PAD_IDX   = getattr(ds, "pad_idx", 20)
    if IDX_TO_AA is None:
        # Fallback to a very common order if dataset doesn't expose it
        IDX_TO_AA = "ARNDCQEGHILKMFPSTWYV"
    print(f"[info] alphabet='{IDX_TO_AA}', pad_idx={PAD_IDX}")
    
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                pin_memory=(device.type=="cuda"), collate_fn=collate)

    # model
    model = ViroporinAFMini(cfg["model"]).to(device)
    ckpt = args.ckpt or load_latest_ckpt(cfg["train"]["ckpt_dir"])

    # run sanity check first
    if not check_checkpoint_sanity(ckpt):
        raise RuntimeError(f"Aborting: checkpoint {ckpt} has NaN/Inf or invalid values")

    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    print(f"[info] loaded {ckpt}")
    print(f"[info] evaluating index: {index_path}")

        # loop
    total, agg = 0.0, {}
    n = len(ds)
    print(f"[eval] Starting evaluation on {n} samples...")

    per = []  # collect per-sample results
    if args.dump_dir:
        dump_dir = root_results / args.dump_dir
    else:
        dump_dir = root_results / "eval_results"

    dump_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] results will be saved in: {dump_dir}")


    for i, batch in enumerate(dl):
        loss, logs = eval_one(cfg, model, batch, device)
        total += loss
        for k, v in logs.items():
            agg[k] = agg.get(k, 0.0) + v

        # sample identity (prefer dataset-provided id)
        try:
            sample_id = batch.get("id", [None])[0]
        except Exception:
            sample_id = None
        sid = safe_name(sample_id) if sample_id else f"idx_{i:05d}"

        rec = {"idx": i, "id": sample_id, "loss": float(loss), **{k: float(v) for k, v in logs.items()}}
        per.append(rec)

        if dump_dir:
            with open(dump_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
                json.dump(rec, f)
                f.write("\n")

        # progress
        if (i + 1) % 10 == 0 or (i + 1) == n:
            avg_loss = total / (i + 1)
            print(f"[eval] {i + 1}/{n} done | running avg loss={avg_loss:.4f}")

    # choose which samples to dump
    to_dump = []
    if dump_dir and args.dump_n > 0 and len(per) > 0:
        if args.dump_mode == "first":
            to_dump = per[:args.dump_n]
        elif args.dump_mode == "random":
            k = min(args.dump_n, len(per))
            to_dump = random.sample(per, k=k)
        elif args.dump_mode == "hard":
            to_dump = sorted(per, key=lambda r: r["loss"], reverse=True)[:args.dump_n]
        elif args.dump_mode == "all":
            to_dump = per 

        print(f"[dump] Preparing {len(to_dump)} example(s) → {dump_dir}")

        # second pass to dump coords (no AMP)
        dl2 = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        pin_memory=(device.type=="cuda"), collate_fn=collate)
        want = {r["idx"]: r for r in to_dump}

        for j, batch in enumerate(dl2):
            if j not in want:
                continue

            model.eval()
            with torch.no_grad():
                out = model(batch["seq_idx"].to(device), batch.get("emb"))
                xyz_ca = out["xyz"].detach().cpu()  # (L,3)

                # --- RECENTER (fix stupid 200/300 offset) ---
                center = xyz_ca.median(dim=0).values
                xyz_ca = xyz_ca - center

                # --- monomer backbone ---
                backbone = reconstruct_backbone_from_ca(xyz_ca)

                sid = safe_name(want[j]["id"]) if want[j]["id"] else f"idx_{j:05d}"
                seq_str = seq_from_indices(batch["seq_idx"])

                mono_path = dump_dir / f"{sid}_mono.pdb"
                with open(mono_path, "w") as fh:
                    fh.write("MODEL        1\n")
                    write_backbone_pdb(
                        fh,
                        backbone,
                        seq_str,
                        chain_id="A",
                        serial_start=1,
                        resid_start=1,
                    )
                    fh.write("ENDMDL\n")

                # --- CN oligomer ---
                if args.dump_olig:
                    n_copies = cfg["priors"]["n_copies"]
                    ring_radius = cfg["priors"]["ring_radius"]

                    olig = assemble_cn(
                        torch.tensor(xyz_ca).to(device),
                        n_copies=n_copies,
                        ring_radius=ring_radius
                    ).cpu()  # (n, L, 3)

                    import string
                    chain_ids = list(string.ascii_uppercase[:n_copies])

                    olig_path = dump_dir / f"{sid}_cn{n_copies}.pdb"
                    with open(olig_path, "w") as fh:
                        fh.write("MODEL        1\n")
                        serial = 1
                        for chain_idx, cid in enumerate(chain_ids):
                            ca_chain = olig[chain_idx]  # centered CA coords

                            # reconstruct backbone
                            bb_chain = reconstruct_backbone_from_ca(ca_chain)

                            serial, _ = write_backbone_pdb(
                                fh,
                                bb_chain,
                                seq_str,
                                chain_id=cid,
                                serial_start=serial,
                                resid_start=1,
                            )
                            fh.write("TER\n")
                        fh.write("ENDMDL\n")



    # --- final report ---
    print(json.dumps({
        "samples": n,
        "loss_val": total / n,
        **{f"{k}_val": agg[k] / n for k in agg}
    }, indent=2))

if __name__ == "__main__":
    main()
