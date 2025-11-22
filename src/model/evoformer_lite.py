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
    def __init__(self, d_pair, n_heads=4, attn_dropout=0.0, ff_mult=2, tile=None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_pair, num_heads=n_heads, dropout=attn_dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_pair, ff_mult * d_pair),
            nn.GELU(),
            nn.Linear(ff_mult * d_pair, d_pair),
        )
        self.tile = tile

    @torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16)
    def forward(self, z):
        if z.dim() == 3:
            z = z.unsqueeze(0)
        if z.dim() == 4:
            B, L1, L2, D = z.shape
            assert L1 == L2, "TriangleBlock expects square pair map (L, L)."
            z_flat = z.reshape(B * L1, L2, D)
            try:
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                    if self.tile is None:
                        z2, _ = self.self_attn(z_flat, z_flat, z_flat, need_weights=False)
                    else:
                        z2 = torch.empty_like(z_flat)
                        step = int(self.tile)
                        for s in range(0, L2, step):
                            e = min(s + step, L2)
                            q = z_flat[:, s:e, :]
                            chunk_out, _ = self.self_attn(q, z_flat, z_flat, need_weights=False)
                            z2[:, s:e, :] = chunk_out
            except Exception:
                if self.tile is None:
                    z2, _ = self.self_attn(z_flat, z_flat, z_flat, need_weights=False)
                else:
                    z2 = torch.empty_like(z_flat)
                    step = int(self.tile)
                    for s in range(0, L2, step):
                        e = min(s + step, L2)
                        q = z_flat[:, s:e, :]
                        chunk_out, _ = self.self_attn(q, z_flat, z_flat, need_weights=False)
                        z2[:, s:e, :] = chunk_out
            z_flat = z_flat + z2
            z_flat = z_flat + self.ff(z_flat)
            return z_flat.reshape(B, L1, L2, D)
        else:
            raise ValueError(f"Unexpected z shape: {tuple(z.shape)}")

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
    def __init__(self, d_single=256, d_pair=128,
                 n_blocks=8, n_attn_heads=4,
                 attn_dropout=0.1, tile_size=256):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "pair_from_single": PairUpdate(d_pair, d_single),
                "tri": TriangleBlock(
                    d_pair,
                    n_heads=n_attn_heads,
                    attn_dropout=attn_dropout,
                    tile=tile_size
                ),
                "single": SingleBlock(d_single, n_attn_heads),
            })
            for _ in range(n_blocks)
        ])
        self.ln_s = nn.LayerNorm(d_single)
        self.ln_z = nn.LayerNorm(d_pair)
        self.pair_proj = nn.Sequential(
            nn.LayerNorm(2 * d_single),
            nn.Linear(2 * d_single, d_pair)
        )


    def forward(self, s, z):
        s = self.ln_s(s)
        z = self.ln_z(z)

        # initial pair injection from single (your current logic)
        if s.dim() == 2:
            L = s.shape[0]
            a = s.unsqueeze(1).expand(L, L, -1)
            b = s.unsqueeze(0).expand(L, L, -1)
            z = z + self.pair_proj(torch.cat([a, b], dim=-1))
        else:
            B, L, ds = s.shape
            a = s.unsqueeze(2).expand(B, L, L, ds)
            b = s.unsqueeze(1).expand(B, L, L, ds)
            z = z + self.pair_proj(torch.cat([a, b], dim=-1))

        # iterate lite evoformer blocks
        for blk in self.blocks:
            z = blk["pair_from_single"](s, z)
            z = blk["tri"](z)
            s = blk["single"](s)

        return s, z


