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
        if z.dim() == 3:
            # legacy: treat L as both batch and seq
            z2, _ = self.self_attn(z, z, z)
            z = z + z2
            z = z + self.ff(z)
            return z
        else:
            # batched: z = (B, L, L, d_pair) → fold first L as "batch"
            B, L, _, D = z.shape
            z_flat = z.reshape(B * L, L, D)
            z2, _ = self.self_attn(z_flat, z_flat, z_flat)
            z_flat = z_flat + z2
            z_flat = z_flat + self.ff(z_flat)
            return z_flat.reshape(B, L, L, D)


class SingleBlock(nn.Module):
    def __init__(self, d_single, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_single, n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(d_single), nn.Linear(d_single, 4*d_single),
                                nn.GELU(), nn.Linear(4*d_single, d_single))

    def forward(self, s):
        if s.dim() == 2:  # (L,ds)
            s2, _ = self.attn(s.unsqueeze(0), s.unsqueeze(0), s.unsqueeze(0))
            s = s + s2.squeeze(0)
        else:             # (B,L,ds)
            s2, _ = self.attn(s, s, s)
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
        if s.dim() == 2:
            L = s.shape[0]
            a = s.unsqueeze(1).expand(L, L, -1)      # (L,L,ds)
            b = s.unsqueeze(0).expand(L, L, -1)      # (L,L,ds)
            z = z + self.proj(torch.cat([a, b], dim=-1))
            return z
        else:
            B, L, ds = s.shape
            a = s.unsqueeze(2).expand(B, L, L, ds)   # (B,L,L,ds)
            b = s.unsqueeze(1).expand(B, L, L, ds)   # (B,L,L,ds)
            z = z + self.proj(torch.cat([a, b], dim=-1))
            return z

