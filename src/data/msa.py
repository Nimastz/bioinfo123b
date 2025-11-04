import torch

def dummy_msa(seq_idx):
    # Minimal: single sequence embedding (L, d_msa). Real MSA goes here.
    L = seq_idx.shape[0]
    return torch.nn.functional.one_hot(seq_idx, num_classes=21).float()  # (L,21)