# src/losses/viroporin_priors.py
# Defines biophysical prior losses that guide viroporin structure prediction toward realistic membrane geometry.
# - membrane_z_mask(): marks residues that lie within the transmembrane region.
# - membrane_slab_loss(): penalizes transmembrane residues that drift too far from the membrane center.
# - interface_contact_loss(): encourages favorable inter-chain contacts between protomers in the membrane.
# - ca_clash_loss(): penalizes steric clashes (atoms too close between chains).
# - pore_target_loss(): steers the predicted pore radius toward a target range (Å) for stable ion-channel geometry.
# These priors act as gentle physical constraints during training to keep predicted oligomeric structures plausible.

import torch
import torch.nn.functional as F

_EPS = 1e-3
_MAX_R = 20.0

def _huber(x, delta=1.0):
    ax = x.abs()
    return torch.where(ax < delta, 0.5 * ax * ax / max(delta, _EPS), ax - 0.5 * delta)

def membrane_z_mask(L, tm_span):
    a, b = int(tm_span[0]), int(tm_span[1])
    a = max(0, min(a, L))
    b = max(0, min(b, L))
    if b <= a: 
        w = max(1, int(0.6 * L))
        a = (L - w) // 2
        b = a + w
    m = torch.zeros(L, dtype=torch.float32)
    m[a:b] = 1.0
    return m

def membrane_slab_loss(xyz, tm_mask, half_thick=4.0):
    z = xyz[..., 2].abs()                 # (..., L)
    excess = torch.clamp(z - half_thick, min=0.0)

    # match mask shape and dtype
    tm = tm_mask.to(z.dtype)
    while tm.dim() < z.dim():
        tm = tm.unsqueeze(0)

    # mean over transmembrane residues, sum over batch if present
    valid = tm.sum(dim=-1).clamp(min=1.0)
    loss = (excess * tm).sum(dim=-1) / valid

    return loss.mean()                    # scalar over batch

def interface_contact_loss(olig_xyz, cutoff=8.0):
    n, L, _ = olig_xyz.shape
    if n < 2:
        return olig_xyz.new_tensor(0.0)
    loss = olig_xyz.new_tensor(0.0)
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            D = torch.cdist(olig_xyz[i], olig_xyz[j], p=2)
            frac_far = (D > float(cutoff)).float().mean()
            loss = loss + frac_far
            pairs += 1
    return loss / max(1, pairs)

def ca_clash_loss(olig_xyz, min_dist=3.6):
    n, L, _ = olig_xyz.shape
    if n < 2:
        return olig_xyz.new_tensor(0.0)
    penalty = olig_xyz.new_tensor(0.0)
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            D = torch.cdist(olig_xyz[i], olig_xyz[j], p=2)
            penalty = penalty + (float(min_dist) - D).clamp_min(0.0).mean()
            pairs += 1
    return penalty / max(1, pairs)

def pore_target_loss(olig_xyz, target_A=4.0, reduce="mean"):
    from src.geometry.assembly import pore_radius_profile_ca
    z, rs = pore_radius_profile_ca(olig_xyz)
    if rs.numel() == 0:
        return olig_xyz.new_tensor(0.0)
    valid = torch.isfinite(rs) & (rs > 0)
    if not valid.any():
        return olig_xyz.new_tensor(0.0)
    rs = rs[valid].clamp(min=_EPS, max=_MAX_R)
    if rs.shape[0] < 3:
        return olig_xyz.new_tensor(0.0)

    r25 = torch.quantile(rs, 0.25)
    main = F.smooth_l1_loss(r25, torch.as_tensor(float(target_A), device=rs.device))
    dr = rs[1:] - rs[:-1]
    smooth = _huber(dr, delta=0.5).mean()
    anti_collapse = _huber(1.0 / rs, delta=1.0).mean()

    loss = main + 0.5 * smooth + 0.2 * anti_collapse
    if reduce == "sum":
        return loss
    return loss



