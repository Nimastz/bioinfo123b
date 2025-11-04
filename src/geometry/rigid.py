import torch

class Rigid:
    # Placeholder for future SE(3) ops if needed.
    @staticmethod
    def center(xyz):
        return xyz - xyz.mean(dim=0, keepdim=True)