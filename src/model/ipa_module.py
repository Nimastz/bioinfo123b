import torch, torch.nn as nn
from src.geometry.rigid import Rigid

class IPA(nn.Module):
    def __init__(self, d_single=256, n_blocks=4, n_points=4, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_single),
                nn.Linear(d_single, d_single), nn.GELU(),
                nn.Linear(d_single, d_single),
                nn.Dropout(dropout)
            ) for _ in range(n_blocks)
        ])
        self.to_xyz = nn.Linear(d_single, 3)

    def forward(self, s):
        # frames start at origin; predict deltas
        xyz = []
        h = s
        for blk in self.blocks:
            h = h + blk(h)
            xyz.append(self.to_xyz(h).cumsum(dim=0))  # crude incremental backbone
        return xyz[-1]  # (L,3)

class TorsionHeadSimple(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.out = nn.Linear(d, 3)  # phi, psi, omega (radians)
    def forward(self, s):
        return self.out(s)
