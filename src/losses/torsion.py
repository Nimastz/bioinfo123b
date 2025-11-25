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
