# src/losses/torsion.py
# Defines a simple loss function for protein backbone torsion angles (phi, psi, omega).
# Encourages predicted torsion angles to match an ideal alpha-helix geometry when helix_bias=True.
# Used during training to promote physically realistic backbone conformations in viroporin models.

# src/losses/distogram.py
import torch, torch.nn.functional as F

def distogram_loss(
    dist_logits: torch.Tensor,   # [L, L, n_bins]
    xyz: torch.Tensor,           # [L, 3] (or [L, 3?]) cartesian coords
    bin_size: float = 0.5,
    n_bins: int = 64,
    dmax: float | None = None,
    pair_mask: torch.Tensor | None = None,  # [L, L] bool or 0/1; True=keep
    eps: float = 1e-6,
    label_smoothing: float = 0.0,           # e.g., 0.01 to stabilize early steps
):
    # ---- Guard & sanitize inputs ----
    if not torch.isfinite(dist_logits).all():
        raise RuntimeError("[distogram_loss] non-finite dist_logits detected")
    if not torch.isfinite(xyz).all():
        # Replace NaN/Inf coords with zeros so cdist won’t propagate NaNs
        xyz = torch.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)

    L = xyz.shape[0]
    if dmax is None:
        dmax = bin_size * n_bins

    # distances in [0, +inf) → clamp into [0, dmax - eps]
    D = torch.cdist(xyz, xyz)                     # [L, L]
    D = torch.clamp(D, 0.0, dmax - eps)

    # Convert to target bin indices in [0, n_bins-1]
    # (use floor on D/bin; safe due to clamp above)
    bins = torch.floor(D / bin_size).to(dtype=torch.long)
    bins = torch.clamp(bins, 0, n_bins - 1)       # [L, L]

    # Optional: ignore diagonal (distance 0) if you don’t want self-pairs
    # diag_mask = ~torch.eye(L, dtype=torch.bool, device=xyz.device)
    # if pair_mask is None: pair_mask = diag_mask
    # else: pair_mask = pair_mask & diag_mask

    if pair_mask is not None:
        # Flatten + ignore masked positions
        pair_mask = pair_mask.to(dtype=torch.bool)
        tgt = bins[pair_mask]                               # [N_kept]
        logit = dist_logits[pair_mask]                      # [N_kept, n_bins]
    else:
        tgt = bins.reshape(-1)                              # [L*L]
        logit = dist_logits.reshape(-1, n_bins)             # [L*L, n_bins]

    # Small, optional label smoothing to avoid overconfident spikes
    if label_smoothing and label_smoothing > 0:
        ce = F.cross_entropy(
            logit, tgt, reduction="mean", label_smoothing=float(label_smoothing)
        )
    else:
        ce = F.cross_entropy(logit, tgt, reduction="mean")

    # Final safety: refuse to return a NaN silently
    if not torch.isfinite(ce):
        raise RuntimeError("[distogram_loss] produced non-finite CE")

    return ce

