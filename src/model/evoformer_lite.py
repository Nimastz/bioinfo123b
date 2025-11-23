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
        self.proj = nn.Linear(2 * d_single, d_pair)

    def forward(self, s, z):
        # s: (B, L, D)
        B, L, D = s.shape
        a = s.unsqueeze(2).expand(B, L, L, D)   # (B, L, L, D) row i
        b = s.unsqueeze(1).expand(B, L, L, D)   # (B, L, L, D) col j
        z_cat = torch.cat([a, b], dim=-1)       # (B, L, L, 2D)
        z = self.proj(z_cat)                    # (B, L, L, d_pair)
        return z

class TriangleBlock(nn.Module):
    def __init__(self, d_pair, n_heads=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_pair, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair, 4 * d_pair),
            nn.GELU(),
            nn.Linear(4 * d_pair, d_pair),
        )

    def forward(self, z):
        # z: (B, L, L, d_pair) → treat each "row i" as a length-L sequence
        B, L, _, D = z.shape
        z_flat = z.reshape(B * L, L, D)                    # (B*L, L, d_pair)
        z2, _ = self.self_attn(z_flat, z_flat, z_flat)     # (B*L, L, d_pair)
        z_flat = z_flat + z2
        z_flat = z_flat + self.ff(z_flat)
        z = z_flat.reshape(B, L, L, D)
        return z

class SingleBlock(nn.Module):
    def __init__(self, d_single, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_single, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_single),
            nn.Linear(d_single, 4 * d_single),
            nn.GELU(),
            nn.Linear(4 * d_single, d_single),
        )

    def forward(self, s):
        # s: (B, L, d_single)
        s2, _ = self.attn(s, s, s)     # no extra unsqueeze/squeeze
        s = s + s2
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
        # s: (B, L, d_single), z: (B, L, L, d_pair)
        for b in self.blocks:
            z = b["pair_from_single"](s, z)   # (B, L, L, d_pair)
            z = b["tri"](z)                   # (B, L, L, d_pair)
            s = b["single"](s)                # (B, L, d_single)
        return self.ln_s(s), self.ln_z(z)
