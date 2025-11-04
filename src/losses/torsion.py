import torch

def torsion_l2(pred_rad):
    return (pred_rad**2).mean()