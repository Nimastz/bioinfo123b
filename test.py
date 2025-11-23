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

def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in (s or ""))

def write_pdb_ca_lines(xyz):
    """
    Yield CA-only PDB lines from (L,3) tensor (Å). Residue = index, ALA dummy.
    """
    for i, (x, y, z) in enumerate(xyz.tolist(), start=1):
        yield ("ATOM  {atom:5d}  CA  ALA A{resid:4d}    "
               "{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n").format(
            atom=i, resid=i, x=x, y=y, z=z
        )

def write_pdb_model(fh, xyz, model_idx=1):
    fh.write(f"MODEL        {model_idx}\n")
    for line in write_pdb_ca_lines(xyz):
        fh.write(line)
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

        # ---- sanitize activations to avoid NaN/Inf crashing the loss ----
        def _safe(t, clip=1e4):
            if t is None:
                return t
            t = torch.nan_to_num(t, nan=0.0, posinf=clip, neginf=-clip)
            return t.clamp(-clip, clip)

        out["dist"] = _safe(out["dist"], clip=1e4)   # logits
        out["xyz"]  = _safe(out["xyz"],  clip=1e3)   # coordinates
        out["tors"] = _safe(out["tors"], clip=1e2)   # angles

        L = out["xyz"].shape[0]
        loss_dist = distogram_loss(out["dist"], out["xyz"])
        loss_tors = torsion_l2(out["tors"])
        loss_fape = fape_loss(out["xyz"])

        total = (
            cfg["loss_weights"]["distogram"]*loss_dist
            + cfg["loss_weights"]["torsion"]*loss_tors
            + cfg["loss_weights"]["fape"]*loss_fape
        )
        logs = dict(distogram=float(loss_dist), torsion=float(loss_tors), fape=float(loss_fape))

        if cfg["priors"].get("use_cn", True):
            n  = cfg["priors"]["n_copies"]
            rr = cfg["priors"]["ring_radius"]
            olig = assemble_cn(out["xyz"], n_copies=n, ring_radius=rr)

            tm_mask = membrane_z_mask(L, cfg["priors"]["tm_span"]).to(out["xyz"].device)
            tm_frac = float(tm_mask.float().mean().item())

            # (optional but recommended) center on TM median z before membrane loss, to mirror training
            xyz = out["xyz"].clone()
            z = xyz[:, 2]
            z_tm = z[tm_mask > 0.5]
            z_center = (z_tm.median() if z_tm.numel() > 0 else z.median()).detach()
            xyz[:, 2] -= z_center

            mem   = membrane_slab_loss(xyz, tm_mask)
            intf  = interface_contact_loss(olig, cutoff=9.0)
            clash = ca_clash_loss(olig, min_dist=3.6)
            pore  = pore_target_loss(olig, target_A=cfg["priors"]["pore_target_A"])

            # pore_minA via pore radius profile on the oligomer
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

            total = total + cfg["loss_weights"]["priors"]*(0.3*mem + 0.4*intf + 0.3*pore) + 0.1*clash

        return float(total), logs

def check_checkpoint_sanity(ckpt_path: str, max_abs_value: float = 1e5):
    """
    Checks checkpoint tensors for NaN, Inf, or abnormally large values.
    Prints a brief report and returns True if all tensors are clean.
    """
    import torch
    bad = []
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # handle both raw state_dict or {"model": state_dict, ...}
    state = ckpt.get("model", ckpt)
    print(f"[check] Scanning checkpoint '{ckpt_path}' for NaN/Inf...")

    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            continue
        if not torch.isfinite(tensor).all():
            bad.append((name, "NaN/Inf"))
        elif tensor.abs().max().item() > max_abs_value:
            bad.append((name, f"too large (>{max_abs_value:g})"))

    if bad:
        print("[warn] Found non-finite or extreme values:")
        for n, msg in bad[:10]:
            print(f"  - {n}: {msg}")
        if len(bad) > 10:
            print(f"  ... and {len(bad)-10} more tensors with issues.")
        return False
    else:
        print("[ok] All checkpoint tensors finite and within normal range.")
        return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="path to step_*.pt (defaults to latest)")
    ap.add_argument("--dump_dir", default=None, help="Directory to dump per-sample outputs (JSONL/PDB)")
    ap.add_argument("--dump_n", type=int, default=3, help="How many 3D examples to dump")
    ap.add_argument("--dump_mode", default="first", choices=["first","random","hard"],
                    help="Which samples to dump as PDBs")
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
                xyz = out["xyz"].detach().cpu()  # (L,3)

                sid = safe_name(want[j]["id"]) if want[j]["id"] else f"idx_{j:05d}"

                # monomer
                mono_path = dump_dir / f"{sid}_mono.pdb"
                with open(mono_path, "w") as fh:
                    write_pdb_model(fh, xyz, model_idx=1)

                if args.dump_olig:
                    n_copies = cfg["priors"]["n_copies"]
                    ring_radius = cfg["priors"]["ring_radius"]
                    olig = assemble_cn(out["xyz"], n_copies=n_copies, ring_radius=ring_radius).detach().cpu()  # (n,L,3)
                    olig_path = dump_dir / f"{sid}_cn{n_copies}.pdb"
                    with open(olig_path, "w") as fh:
                        for c in range(olig.shape[0]):
                            write_pdb_model(fh, olig[c], model_idx=c+1)

    # --- final report ---
    print(json.dumps({
        "samples": n,
        "loss_val": total / n,
        **{f"{k}_val": agg[k] / n for k in agg}
    }, indent=2))

if __name__ == "__main__":
    main()
