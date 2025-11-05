# src/losses/viroporin_priors.py
# Defines biophysical prior losses that guide viroporin structure prediction toward realistic membrane geometry.
# - membrane_z_mask(): marks residues that lie within the transmembrane region.
# - membrane_slab_loss(): penalizes transmembrane residues that drift too far from the membrane center.
# - interface_contact_loss(): encourages favorable inter-chain contacts between protomers in the membrane.
# - ca_clash_loss(): penalizes steric clashes (atoms too close between chains).
# - pore_target_loss(): steers the predicted pore radius toward a target range (Å) for stable ion-channel geometry.
# These priors act as gentle physical constraints during training to keep predicted oligomeric structures plausible.

import torch

def membrane_z_mask(L, tm_span):
    m = torch.zeros(L, dtype=torch.float32)
    m[tm_span[0]:tm_span[1]] = 1.0
    return m

def membrane_slab_loss(xyz, tm_mask, z_half_thickness=10.0):
    # penalize TM residues far from slab center (z=0)
    z = xyz[:,2]
    return (tm_mask*(z.abs() - z_half_thickness).relu()).mean()

def interface_contact_loss(olig_xyz, cutoff=8.0):
    # encourage inter-chain contacts in TM (coarse)
    n, L, _ = olig_xyz.shape
    loss = olig_xyz.new_tensor(0.0)
    for i in range(n):
        for j in range(i+1, n):
            D = torch.cdist(olig_xyz[i], olig_xyz[j])
            loss = loss + (D > cutoff).float().mean()
    return loss / max(1, n*(n-1)//2)

def ca_clash_loss(olig_xyz, min_dist=3.6):
    n, L, _ = olig_xyz.shape
    penalty = olig_xyz.new_tensor(0.0)
    for i in range(n):
        for j in range(i+1, n):
            D = torch.cdist(olig_xyz[i], olig_xyz[j])
            penalty = penalty + (min_dist - D).relu().mean()
    return penalty / max(1, n*(n-1)//2)

def pore_target_loss(olig_xyz, target_A=4.0, reduce="mean"):
    from src.geometry.assembly import pore_radius_profile_ca
    _, rs = pore_radius_profile_ca(olig_xyz)
    if rs.numel() == 0 or not torch.isfinite(rs).any():
        return olig_xyz.new_tensor(0.0)
    valid = ~torch.isnan(rs)
    if not valid.any(): return olig_xyz.new_tensor(0.0)
    r25 = torch.nanquantile(rs[valid], 0.25)
    low, high = target_A-0.5, target_A+1.0
    return (low - r25).relu()**2 + (r25 - high).relu()**2
