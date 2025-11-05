# src/geometry/torsion.py
# Defines a simple loss function for protein backbone torsion angles (phi, psi, omega).
# Encourages predicted torsion angles to match an ideal alpha-helix geometry when helix_bias=True.
# Used during training to promote physically realistic backbone conformations in viroporin models.

import torch

def torsion_loss(pred_rad, helix_bias=True):
    # pred_rad: (L,3) radians; simple L2 toward ideal helix if desired
    if helix_bias:
        target = torch.tensor([-0.9948, -0.8203, 3.1416], device=pred_rad.device)  # (-57,-47,180 deg)
        return ((pred_rad - target)**2).mean()
    return pred_rad.new_tensor(0.0)
