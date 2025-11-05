# src/model/heads.py
# Defines output (head) modules for the ViroporinAFMini model.
# - DistogramHead: predicts residue–residue distance distributions (distograms) from pair features.
# - ConfidenceHeads: predicts per-residue (pLDDT) and pairwise (PAE) confidence scores
#   indicating the reliability of the predicted 3D structure.
# These heads convert internal model representations into interpretable structural predictions
# and confidence metrics for evaluation.

import torch.nn as nn

class DistogramHead(nn.Module):
    def __init__(self, d_pair, n_bins=64):
        super().__init__()
        self.proj = nn.Linear(d_pair, n_bins)

    def forward(self, z):
        # (L,L,dp) -> (L,L,B)
        return self.proj(z)

class ConfidenceHeads(nn.Module):
    def __init__(self, d_single, d_pair):
        super().__init__()
        self.plddt = nn.Sequential(nn.LayerNorm(d_single), nn.Linear(d_single, 1))
        self.pae   = nn.Sequential(nn.LayerNorm(d_pair),   nn.Linear(d_pair, 1))

    def forward(self, s, z):
        return self.plddt(s).squeeze(-1), self.pae(z).squeeze(-1)
