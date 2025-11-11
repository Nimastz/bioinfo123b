# test.py
# Evaluation script for the ViroporinAFMini model.
# Loads a trained checkpoint and runs it on the validation dataset to measure performance.
# - load_latest_ckpt(): finds the most recent saved model checkpoint.
# - eval_one(): computes all relevant losses (distogram, torsion, FAPE, and membrane/pore priors)
#   for a single sample without updating model weights.
# - main(): loads configuration, data, and model; evaluates all validation samples; and
#   prints averaged loss metrics as JSON output.
# Used to assess how well the trained model predicts 3D viroporin structures after training.

import argparse, yaml, torch, glob, os, json
from pathlib import Path
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

def load_latest_ckpt(ckpt_dir: str):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")), key=os.path.getmtime)
    if not paths:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    return paths[-1]

def eval_one(cfg, model, batch, device):
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type=="cuda")):
        out = model(batch["seq_idx"].to(device), batch.get("emb"))
        L = out["xyz"].shape[0]
        loss_dist = distogram_loss(out["dist"], out["xyz"])
        loss_tors = torsion_l2(out["tors"])
        loss_fape = fape_loss(out["xyz"])
        total = cfg["loss_weights"]["distogram"]*loss_dist + \
                cfg["loss_weights"]["torsion"]*loss_tors + \
                cfg["loss_weights"]["fape"]*loss_fape
        logs = dict(distogram=float(loss_dist), torsion=float(loss_tors), fape=float(loss_fape))

        if cfg["priors"].get("use_cn", True):
            n = cfg["priors"]["n_copies"]; rr = cfg["priors"]["ring_radius"]
            olig = assemble_cn(out["xyz"], n_copies=n, ring_radius=rr)
            tm_mask = membrane_z_mask(L, cfg["priors"]["tm_span"]).to(out["xyz"].device)
            mem = membrane_slab_loss(out["xyz"], tm_mask)
            intf = interface_contact_loss(olig, cutoff=9.0)
            clash = ca_clash_loss(olig, min_dist=3.6)
            pore = pore_target_loss(olig, target_A=cfg["priors"]["pore_target_A"])
            logs.update(mem=float(mem), intf=float(intf), clash=float(clash), pore=float(pore))
            total = total + cfg["loss_weights"]["priors"]*(0.3*mem + 0.4*intf + 0.3*pore) + 0.1*clash
        return float(total), logs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="path to step_*.pt (defaults to latest)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    ds = JsonlSet(cfg["data"]["val_index"], cfg["data"].get("max_len"))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"), collate_fn=collate)

    # model
    model = ViroporinAFMini(cfg["model"]).to(device)
    ckpt = args.ckpt or load_latest_ckpt(cfg["train"]["ckpt_dir"])
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    print(f"[info] loaded {ckpt}")

    # loop
    total, agg = 0.0, {}
    for i, batch in enumerate(dl):
        loss, logs = eval_one(cfg, model, batch, device)
        total += loss
        for k,v in logs.items():
            agg[k] = agg.get(k, 0.0) + v
    n = len(ds)
    print(json.dumps({
        "samples": n,
        "loss_val": total/n,
        **{f"{k}_val": agg[k]/n for k in agg}
    }, indent=2))

if __name__ == "__main__":
    main()
