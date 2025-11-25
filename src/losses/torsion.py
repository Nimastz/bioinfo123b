# src/losses/torsion.py
# Defines a simple L2 loss on predicted torsion angles (in radians).
# Penalizes large deviations in backbone torsion predictions to maintain stable geometry.
# Used as a lightweight regularization term during protein structure training.

import torch

def torsion_l2(pred_rad, ref_rad=None):
    """
    If ref_rad is None: simple L2 on pred (regularizer).
    If ref_rad is given: L2 on (pred - ref).
    """
    if ref_rad is None:
        diff = pred_rad
    else:
        diff = pred_rad - ref_rad

    return (diff ** 2).mean()

def torsion_loss(pred_rad, helix_bias=True):
    if helix_bias:
        target = torch.tensor([-0.9948, -0.8203, 3.1416], device=pred_rad.device)  # (-57,-47,180 deg)
        return ((pred_rad - target)**2).mean()
    return pred_rad.new_tensor(0.0)
