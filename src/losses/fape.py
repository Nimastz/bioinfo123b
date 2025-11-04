import torch

def fape_loss(pred_xyz, ref_xyz=None):
    # If no ref supplied (self-consistency toy): encourage chain smoothness
    if ref_xyz is None:
        return ((pred_xyz[1:] - pred_xyz[:-1]).norm(dim=-1) - 3.8).abs().mean()
    return ((pred_xyz - ref_xyz)**2).mean()
