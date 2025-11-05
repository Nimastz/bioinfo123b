# src/geometry/rigid.py
# Basic geometric helper for manipulating 3D protein coordinates.
# The Rigid class is a placeholder for future rigid-body (SE(3)) operations such as
# rotation and translation of atomic coordinates.
# Currently, it provides a simple utility to center coordinates around the origin.

import torch

class Rigid:
    # Placeholder for future SE(3) ops if needed.
    @staticmethod
    def center(xyz):
        return xyz - xyz.mean(dim=0, keepdim=True)