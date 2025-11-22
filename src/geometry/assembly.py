# src/geometry/assembly.py
# Geometry utilities for building and analyzing symmetric viroporin assemblies.
# - rotate_z(): rotates atomic coordinates around the z-axis.
# - assemble_cn(): arranges one protein subunit into an n-fold (Cn) ring to form an oligomer.
# - pore_radius_profile_ca(): computes the pore radius along the z-axis from Cα coordinates,
#   estimating how wide the central channel is at each height.
# Used for modeling and evaluating viral ion-channel (viroporin) pore geometry.

import torch, math

def rotate_z(xyz, angle):
    c, s = math.cos(angle), math.sin(angle)
    R = torch.tensor([[c,-s,0.],[s,c,0.],[0,0,1.]], device=xyz.device, dtype=xyz.dtype)
    return xyz @ R.T

def assemble_cn(protomer_xyz, n_copies=4, ring_radius=5.5, phase_offset_deg=0.0):
    out = []
    phase0 = math.radians(phase_offset_deg)
    for k in range(n_copies):
        ang = 2*math.pi*k/n_copies + phase0
        moved = rotate_z(protomer_xyz, ang)
        moved = moved + torch.tensor([ring_radius*math.cos(ang), ring_radius*math.sin(ang), 0.0],
                                     device=protomer_xyz.device, dtype=protomer_xyz.dtype)
        out.append(moved)
    return torch.stack(out, dim=0)  # (n,L,3)

import torch

def _finite_min(x: torch.Tensor) -> torch.Tensor:
    m = torch.isfinite(x)
    if m.any():
        return x[m].min()
    return x.new_tensor(float("nan"))

def _finite_max(x: torch.Tensor) -> torch.Tensor:
    m = torch.isfinite(x)
    if m.any():
        return x[m].max()
    return x.new_tensor(float("nan"))

def pore_radius_profile_ca(olig_xyz: torch.Tensor, z_step=0.5, z_pad=2.0, ca_radius=1.9):

    # Work in float32 to avoid fp16/bf16 edge-cases
    xyz = olig_xyz.to(torch.float32)
    zvals = xyz[..., 2]

    zmin_t = _finite_min(zvals)
    zmax_t = _finite_max(zvals)
    if not torch.isfinite(zmin_t) or not torch.isfinite(zmax_t):
        zs = xyz.new_zeros((0,))
        rs = xyz.new_full((0,), float("nan"))
        return zs, rs

    zmin = (zmin_t - z_pad).item()
    zmax = (zmax_t + z_pad).item()
    if not (zmax > zmin + 1e-6):
        zs = xyz.new_zeros((0,))
        rs = xyz.new_full((0,), float("nan"))
        return zs, rs

    zs = torch.arange(zmin, zmax + 1e-6, z_step, device=xyz.device, dtype=xyz.dtype)
    rs = torch.full_like(zs, float("nan"))

    half = 0.5 * z_step
    xy = xyz[..., :2]  # (n, L, 2)
    for i, z in enumerate(zs):
        mask = (zvals > z - half) & (zvals <= z + half)
        pts = xy[mask]  # (K, 2)
        if pts.numel() == 0:
            continue
        r = torch.linalg.norm(pts, dim=-1).min()
        rs[i] = r - ca_radius
    return zs, rs

