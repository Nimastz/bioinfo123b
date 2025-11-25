# src/model/ipa_module.py
# Implements simplified versions of AlphaFold's structure module components.
# - IPA (Invariant Point Attention-like block): refines per-residue embeddings to predict
#   3D backbone coordinates (xyz) by iteratively updating structure representations.
# - TorsionHeadSimple: predicts backbone torsion angles (phi, psi, omega) in radians from residue embeddings.
# These modules convert learned sequence features into explicit 3D geometry for protein structure prediction.

import torch, torch.nn as nn
from src.geometry.rigid import Rigid

import torch, torch.nn as nn
from src.geometry.rigid import Rigid

class IPA(nn.Module):
    def __init__(self, d_single=256, n_blocks=4, n_points=4, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_single),
                nn.Linear(d_single, d_single),
                nn.GELU(),
                nn.Linear(d_single, d_single),
                nn.Dropout(dropout),
            )
            for _ in range(n_blocks)
        ])
        self.to_xyz = nn.Linear(d_single, 3)

    def forward(self, s):
        """
        s: (B, L, d_single)
        Returns xyz: (B, L, 3), centered per-chain.
        """
        # ensure 3D (batch) even if caller gives (L,D)
        if s.dim() == 2:
            s = s.unsqueeze(0)  # (1, L, D)

        h = s
        for blk in self.blocks:
            h = h + blk(h)      # simple residual MLP stack

        xyz = self.to_xyz(h)    # (B, L, 3)
        # center each chain around its own mean to keep numbers moderate
        xyz = xyz - xyz.mean(dim=1, keepdim=True)
        return xyz

class TorsionHeadSimple(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.out = nn.Linear(d, 3)  # phi, psi, omega (radians)
    def forward(self, s):
        return self.out(s)
