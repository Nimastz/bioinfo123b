# src/losses/fape.py
# Defines the Frame Aligned Point Error (FAPE) loss for evaluating 3D protein coordinates.
# Measures how close predicted atomic positions (pred_xyz) are to reference coordinates (ref_xyz).
# If no reference is provided, applies a self-consistency loss that encourages smooth,
# realistic backbone spacing (~3.8 Å between consecutive residues).

import torch

def fape_loss(pred_xyz, ref_xyz=None):
    # If no ref supplied (self-consistency toy): encourage chain smoothness
    if ref_xyz is None:
        return ((pred_xyz[1:] - pred_xyz[:-1]).norm(dim=-1) - 3.8).abs().mean()
    return ((pred_xyz - ref_xyz)**2).mean()
