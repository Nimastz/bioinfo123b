# src/losses/torsion.py
# Defines a simple loss function for protein backbone torsion angles (phi, psi, omega).
# Encourages predicted torsion angles to match an ideal alpha-helix geometry when helix_bias=True.
# Used during training to promote physically realistic backbone conformations in viroporin models.

import torch
import torch.nn.functional as F

def distogram_loss(
    dist_logits: torch.Tensor,   # [L, L, n_bins]
    xyz: torch.Tensor,           # [L, 3]
    bin_size: float = 0.5,
    n_bins: int = 64,
    dmax: float | None = None,
    pair_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
    label_smoothing: float = 0.0,
):
    """Computes distogram cross-entropy loss between predicted logits and true pairwise distances."""

    # ---- Sanitize inputs ----
    if not torch.isfinite(dist_logits).all():
        bad = torch.sum(~torch.isfinite(dist_logits)).item()
        print(f"[warn] distogram_loss(): {bad} non-finite logits found, replacing with safe defaults.")
        dist_logits = torch.nan_to_num(dist_logits, nan=0.0, posinf=1e4, neginf=-1e4)

    if not torch.isfinite(xyz).all():
        bad = torch.sum(~torch.isfinite(xyz)).item()
        print(f"[warn] distogram_loss(): {bad} non-finite xyz values found, replacing with zeros.")
        xyz = torch.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)

    L = xyz.shape[0]
    if dmax is None:
        dmax = bin_size * n_bins

    # ---- Compute target bins ----
    D = torch.cdist(xyz, xyz)                     # [L, L]
    D = torch.clamp(D, 0.0, dmax - eps)
    bins = torch.floor(D / bin_size).to(torch.long)
    bins = torch.clamp(bins, 0, n_bins - 1)

    if pair_mask is not None:
        pair_mask = pair_mask.to(dtype=torch.bool)
        tgt = bins[pair_mask]                     # [N_kept]
        logit = dist_logits[pair_mask]            # [N_kept, n_bins]
    else:
        tgt = bins.reshape(-1)                    # [L*L]
        logit = dist_logits.reshape(-1, n_bins)   # [L*L, n_bins]

    # ---- Cross-entropy loss ----
    if label_smoothing and label_smoothing > 0:
        ce = F.cross_entropy(logit, tgt, reduction="mean",
                             label_smoothing=float(label_smoothing))
    else:
        ce = F.cross_entropy(logit, tgt, reduction="mean")

    # ---- Final safety ----
    if not torch.isfinite(ce):
        raise RuntimeError("[distogram_loss] produced non-finite CE")

    return ce
