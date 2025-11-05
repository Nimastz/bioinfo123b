# src/data/msa.py
# Provides a minimal placeholder for multiple sequence alignment (MSA) embeddings.
# - dummy_msa(): creates a one-hot encoding of a single protein sequence as a simple MSA substitute.
# In a full pipeline, this would be replaced with real MSA features gathered from homologous sequences.

import torch

def dummy_msa(seq_idx):
    # Minimal: single sequence embedding (L, d_msa). Real MSA goes here.
    L = seq_idx.shape[0]
    return torch.nn.functional.one_hot(seq_idx, num_classes=21).float()  # (L,21)