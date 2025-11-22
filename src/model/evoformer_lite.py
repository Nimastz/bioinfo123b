# src/model/evoformer_lite.py
# Implements a lightweight version of the Evoformer architecture inspired by AlphaFold.
# Combines single-sequence (s) and pairwise (z) representations using attention and feed-forward blocks.
# - PairUpdate: mixes per-residue embeddings into pair features.
# - TriangleBlock: applies multihead self-attention on pair features to capture residue–residue relations.
# - SingleBlock: updates per-residue embeddings using self-attention.
# EvoformerLite enables efficient information exchange between sequence and pair features
# during protein structure prediction.

import torch, torch.nn as nn

class PairUpdate(nn.Module):
    def __init__(self, d_pair, d_single):
        super().__init__()
        self.proj = nn.Linear(2*d_single, d_pair)

    def forward(self, s, z):
        # s: (L, ds), z: (L,L,dp)
        L = s.shape[0]
        a = s.unsqueeze(1).expand(L, L, -1)
        b = s.unsqueeze(0).expand(L, L, -1)
        z = z + self.proj(torch.cat([a,b], dim=-1))
        return z

class TriangleBlock(nn.Module):
    def __init__(self, d_pair, n_heads=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_pair, n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(d_pair), nn.Linear(d_pair, 4*d_pair),
                                nn.GELU(), nn.Linear(4*d_pair, d_pair))

    def forward(self, z):
        z2, _ = self.self_attn(z, z, z)  # z: (batch=L, seq=L, d_pair) thanks to batch_first=True
        z = z + z2
        z = z + self.ff(z)
        return z

class SingleBlock(nn.Module):
    def __init__(self, d_single, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_single, n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(d_single), nn.Linear(d_single, 4*d_single),
                                nn.GELU(), nn.Linear(4*d_single, d_single))

    def forward(self, s):
        s2,_ = self.attn(s.unsqueeze(0), s.unsqueeze(0), s.unsqueeze(0))
        s = s + s2.squeeze(0)
        s = s + self.ff(s)
        return s

class EvoformerLite(nn.Module):
    def __init__(self, d_single=256, d_pair=128, n_blocks=8, n_attn_heads=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "pair_from_single": PairUpdate(d_pair, d_single),
                "tri": TriangleBlock(d_pair, n_attn_heads),
                "single": SingleBlock(d_single, n_attn_heads),
            }) for _ in range(n_blocks)
        ])
        self.ln_s = nn.LayerNorm(d_single)
        self.ln_z = nn.LayerNorm(d_pair)

    def forward(self, s, z):
        for b in self.blocks:
            z = b["pair_from_single"](s, z)
            z = b["tri"](z)
            s = b["single"](s)
        return self.ln_s(s), self.ln_z(z)
