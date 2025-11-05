# src/losses/torsion.py
# Defines a simple L2 loss on predicted torsion angles (in radians).
# Penalizes large deviations in backbone torsion predictions to maintain stable geometry.
# Used as a lightweight regularization term during protein structure training.

import torch

def torsion_l2(pred_rad):
    return (pred_rad**2).mean()