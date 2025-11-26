#!/usr/bin/env python
"""
Analyze a directory of AlphaFold/ColabFold PDB models and compute
the same geometric / prior-related metrics that your training loop uses.

Usage (from repo root):
  python scripts/analyze_af2_pdbs.py \
      --config configs/viroporin_small.yaml \
      --pdb_dir alphafold/pdb \
      --out_csv af2_summary.csv

This lets you compare AF2 averages (mem_raw, pore, pore_minA, etc.)
to the averages from your training/test logs.
"""

import os
import glob
import argparse
import yaml
import csv
from typing import List, Dict

import torch

from src.losses.viroporin_priors import (
    membrane_z_mask,
    membrane_slab_loss,
    interface_contact_loss,
    ca_clash_loss,
    pore_target_loss,
)
from src.geometry.assembly import assemble_cn, pore_radius_profile_ca


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="YAML config with priors (same you use for training).")
    p.add_argument("--pdb_dir", required=True,
                   help="Directory with AF2/ColabFold PDBs.")
    p.add_argument("--pattern", default="*.pdb",
                   help="Glob pattern inside pdb_dir (default: *.pdb).")
    p.add_argument("--out_csv", default=None,
                   help="Optional: write per-structure stats to this CSV.")
    return p.parse_args()


# --- Simple PDB CA parser (no Biopython needed) ------------------------------

def load_ca_coords_from_pdb(pdb_path: str) -> torch.Tensor:
    """
    Return CA coordinates from a PDB as a (L, 3) float32 tensor.
    Assumes a single chain; takes residues in file order.
    """
    cas = []
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            cas.append([x, y, z])

    if not cas:
        raise ValueError(f"No CA atoms found in {pdb_path}")

    return torch.tensor(cas, dtype=torch.float32)


# --- Core metrics computation per structure ----------------------------------

def analyze_structure(
    xyz_ca: torch.Tensor,
    priors_cfg: Dict,
) -> Dict[str, float]:
    """
    Given CA coords (L,3) and priors config, compute the same style
    membrane/pore/interface metrics as training.
    """
    device = torch.device("cpu")
    xyz = xyz_ca.to(device)
    L = xyz.shape[0]

    # ---- TM mask and fraction ----
    tm_span = priors_cfg["tm_span"]  # [z_min, z_max] in Angstrom or residue span (depending on your impl)
    tm_mask = membrane_z_mask(L, tm_span).to(device)  # (L,)
    val_mask = torch.ones(L, dtype=torch.bool, device=device)  # assume all residues are valid
    tm_mask = tm_mask * val_mask.float()
    tm_frac = float(tm_mask.mean().item())

    # ---- Robust centering (same logic as training) ----
    z_all = xyz[:, 2]
    tm_bool = (tm_mask > 0.5) & val_mask
    val_bool = val_mask

    if tm_bool.any():
        z_center = z_all[tm_bool].median().detach()
    else:
        z_center = z_all[val_bool].median().detach()

    xyz_centered = xyz.clone()
    xyz_centered[:, 2] -= z_center

    # Diagnostics
    if tm_bool.any():
        zc_tm = xyz_centered[tm_bool, 2]
        z_abs_mean_tm = float(zc_tm.abs().mean().item())
    else:
        zc_tm = None
        z_abs_mean_tm = float("nan")

    zc_all = xyz_centered[val_bool, 2]
    z_abs_mean = float(zc_all.abs().mean().item())

    use_z = zc_tm if zc_tm is not None else zc_all
    z_tm_min = float(use_z.min().item())
    z_tm_max = float(use_z.max().item())
    z_tm_range = float((use_z.max() - use_z.min()).item())

    if tm_bool.any():
        z_abs_mean_unc = float(z_all[tm_bool].abs().mean().item())
    else:
        z_abs_mean_unc = float(z_all[val_bool].abs().mean().item())

    # ---- CN assembly (same as training) ----
    n_copies = priors_cfg["n_copies"]
    ring_radius = priors_cfg["ring_radius"]
    olig = assemble_cn(xyz, n_copies=n_copies, ring_radius=ring_radius)

    # ---- Membrane slab loss (uncapped raw) ----
    mem_raw = membrane_slab_loss(xyz_centered, tm_mask)

    # ---- Interface + clash ----
    intf = interface_contact_loss(olig, cutoff=9.0)
    clash = ca_clash_loss(olig, min_dist=3.6)

    # ---- Pore target + profile ----
    pore_target_A = priors_cfg["pore_target_A"]
    pore = pore_target_loss(olig, target_A=pore_target_A)
    if not torch.isfinite(pore):
        pore = torch.tensor(0.0, device=device)

    zs, rs = pore_radius_profile_ca(olig)
    pore_minA = float("nan")
    if rs.numel() > 0:
        finite_rs = rs[torch.isfinite(rs)]
        if finite_rs.numel() > 0:
            pore_minA = float(finite_rs.min().item())

    # Convert to Python floats for JSON/CSV
    out = {
        "L": int(L),
        "tm_frac": float(tm_frac),
        "mem_raw": float(mem_raw.item()),
        "pore": float(pore.item()),
        "pore_minA": float(pore_minA),
        "intf": float(intf.item()),
        "clash": float(clash.item()),
        "z_abs_mean": float(z_abs_mean),
        "z_abs_mean_tm": float(z_abs_mean_tm),
        "z_abs_mean_unc": float(z_abs_mean_unc),
        "z_tm_min": float(z_tm_min),
        "z_tm_max": float(z_tm_max),
        "z_tm_range": float(z_tm_range),
    }
    return out


# --- Main driver -------------------------------------------------------------

def main():
    args = parse_args()

    # Load config to get priors
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    priors_cfg = cfg["priors"]

    pdb_glob = os.path.join(args.pdb_dir, args.pattern)
    pdb_paths = sorted(glob.glob(pdb_glob))
    if not pdb_paths:
        raise SystemExit(f"No PDBs found in {pdb_glob}")

    print(f"[info] found {len(pdb_paths)} PDB files in {args.pdb_dir}")

    # Fields we’ll compute & average
    fields = [
        "L",
        "tm_frac",
        "mem_raw",
        "pore",
        "pore_minA",
        "intf",
        "clash",
        "z_abs_mean",
        "z_abs_mean_tm",
        "z_abs_mean_unc",
        "z_tm_min",
        "z_tm_max",
        "z_tm_range",
    ]

    per_struct = []  # list of dicts
    for i, pdb_path in enumerate(pdb_paths, 1):
        try:
            xyz = load_ca_coords_from_pdb(pdb_path)
        except Exception as e:
            print(f"[warn] skipping {pdb_path}: {e}")
            continue

        metrics = analyze_structure(xyz, priors_cfg)
        metrics["pdb"] = os.path.basename(pdb_path)
        per_struct.append(metrics)

        print(
            f"[{i}/{len(pdb_paths)}] {metrics['pdb']}: "
            f"tm={metrics['tm_frac']:.3f}, "
            f"mem_raw={metrics['mem_raw']:.3f}, "
            f"pore={metrics['pore']:.3f}, "
            f"pore_minA={metrics['pore_minA']:.2f}, "
            f"intf={metrics['intf']:.3f}, "
            f"clash={metrics['clash']:.3f}"
        )

    if not per_struct:
        raise SystemExit("[error] no valid structures analyzed.")

    # ---- Compute global averages ----
    avg = {}
    n = len(per_struct)
    for f in fields:
        vals = [m[f] for m in per_struct if m[f] == m[f]]  # drop NaN
        if vals:
            avg[f] = sum(vals) / len(vals)
        else:
            avg[f] = float("nan")

    print("\n=== Global AF2 averages over analyzed PDBs ===")
    for f in fields:
        print(f"{f:15s} : {avg[f]:.4f}")

    # ---- Optional: write CSV ----
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["pdb"] + fields)
            writer.writeheader()
            for m in per_struct:
                row = {k: m.get(k, "") for k in ["pdb"] + fields}
                writer.writerow(row)
        print(f"\n[info] wrote per-structure stats to {args.out_csv}")


if __name__ == "__main__":
    main()
