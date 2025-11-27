# src/model/msa_block.py
# Tiny MSA encoder + attention block used by ViroporinAFMini.
# - TinyMSAEncoder: embeds MSA token indices (gap/pad aware) → (B, N, L, d_msa)
# - TinyMSABlock: very small row- and column-wise attention over the MSA
#
# Shapes:
#   msa_idx: (B, N, L) int64, values in [0..21] where:
#       0..19 = standard AAs
#       20    = gap ("-")
#       21    = PAD (padding; see src/data/dataset.py)
#
#   output: (B, N, L, d_msa)

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyMSAEncoder(nn.Module):
    """
    Embed MSA integer tokens into a small d_msa space.

    msa_idx: (B, N, L) or (N, L)
    returns: (B, N, L, d_msa)
    """

    def __init__(self, d_msa: int = 128, vocab_size: int = 22, pad_idx: int = 21):
        super().__init__()
        self.d_msa = d_msa
        self.pad_idx = pad_idx

        # 0..19 AAs, 20 = GAP, 21 = PAD
        self.embed = nn.Embedding(vocab_size, d_msa, padding_idx=pad_idx)

    def forward(self, msa_idx: torch.Tensor) -> torch.Tensor:
        # (N, L) → (1, N, L)
        if msa_idx.dim() == 2:
            msa_idx = msa_idx.unsqueeze(0)

        # (B, N, L) → (B, N, L, d_msa)
        emb = self.embed(msa_idx)
        return emb


class _RowAttention(nn.Module):
    """
    Self-attention across MSA *rows* (N dimension) for each position L.
    Input:  (B, N, L, d)
    Output: (B, N, L, d)
    """

    def __init__(self, d_msa: int, n_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_msa, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_msa)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, N, L, d)
        B, N, L, D = x.shape

        # We attend over N for each (B, L), so reshape to (B*L, N, D)
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * L, N, D)  # (B*L, N, D)

        if pad_mask is not None:
            # pad_mask: (B, N, L) bool → for each (B,L) we need (N,)
            pm = pad_mask.permute(0, 2, 1).reshape(B * L, N)  # (B*L, N)
        else:
            pm = None

        y, _ = self.mha(x_reshaped, x_reshaped, x_reshaped, key_padding_mask=pm)
        y = self.ln(x_reshaped + y)  # residual + norm

        # Back to (B, N, L, D)
        y = y.reshape(B, L, N, D).permute(0, 2, 1, 3)
        return y


class _ColAttention(nn.Module):
    """
    Self-attention across *columns* (L dimension) for each MSA sequence N.
    Input:  (B, N, L, d)
    Output: (B, N, L, d)
    """

    def __init__(self, d_msa: int, n_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_msa, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_msa)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, N, L, d)
        B, N, L, D = x.shape

        # We attend over L for each (B, N), so reshape to (B*N, L, D)
        x_reshaped = x.reshape(B * N, L, D)  # (B*N, L, D)

        if pad_mask is not None:
            # pad_mask: (B, N, L) bool → (B*N, L)
            pm = pad_mask.reshape(B * N, L)
        else:
            pm = None

        y, _ = self.mha(x_reshaped, x_reshaped, x_reshaped, key_padding_mask=pm)
        y = self.ln(x_reshaped + y)

        # Back to (B, N, L, D)
        y = y.reshape(B, N, L, D)
        return y


class TinyMSABlock(nn.Module):
    """
    Very small MSA block:
      1) Row attention over sequences (N) per position L
      2) Column attention over positions (L) per sequence N

    Input:  (B, N, L, d_msa)
    Output: (B, N, L, d_msa)
    """

    def __init__(self, d_msa: int = 128, n_heads: int = 4, pad_idx: int = 21):
        super().__init__()
        self.pad_idx = pad_idx
        self.row_attn = _RowAttention(d_msa, n_heads=n_heads)
        self.col_attn = _ColAttention(d_msa, n_heads=n_heads)

    def forward(self, x: torch.Tensor, msa_idx: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:       (B, N, L, d_msa) MSA embeddings
        msa_idx: (B, N, L) or None – if provided, used for pad masking
        """

        pad_mask = None
        if msa_idx is not None:
            # PAD positions are True in the key_padding_mask
            if msa_idx.dim() == 2:
                msa_idx = msa_idx.unsqueeze(0)
            pad_mask = (msa_idx == self.pad_idx)  # (B, N, L) bool

        # Row attention
        x = self.row_attn(x, pad_mask=pad_mask)

        # Column attention
        x = self.col_attn(x, pad_mask=pad_mask)

        return x
