# src/model/viroporin_net.py
# Defines the main neural network architecture (ViroporinAFMini) for viral membrane protein prediction.
# Combines several key modules:
# - EvoformerLite: captures sequence and pairwise residue relationships.
# - IPA: predicts 3D backbone coordinates from residue embeddings.
# - TorsionHeadSimple: outputs backbone torsion angles (phi, psi, omega).
# - DistogramHead and ConfidenceHeads: predict residue–residue distances and confidence scores.
# The model processes amino acid sequences or embeddings to generate full 3D structural predictions
# and confidence estimates for viroporin proteins.

import torch, torch.nn as nn
from src.model.evoformer_lite import EvoformerLite
from src.model.ipa_module import IPA, TorsionHeadSimple
from src.model.heads import DistogramHead, ConfidenceHeads

class ViroporinAFMini(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ds, dp = cfg["d_single"], cfg["d_pair"]
        self.embed = nn.Linear(cfg.get("feat_dim", 480), ds)  # if using ESM; else overwritten
        self.embed_oh = nn.Embedding(21, ds)
        self.evo = EvoformerLite(ds, dp, cfg["evoformer"]["n_blocks"], cfg["evoformer"]["n_attn_heads"])
        self.ipa = IPA(ds, cfg["ipa"]["n_blocks"], cfg["ipa"]["n_points"], cfg["ipa"]["dropout"])
        self.dist = DistogramHead(dp, cfg["heads"]["distogram_bins"])
        self.tors = TorsionHeadSimple(ds)
        self.conf = ConfidenceHeads(ds, dp)

    def forward(self, seq_idx, emb=None):
        # Accept both (L,) and (B,L)
        if seq_idx.dim() == 1:
            seq_idx = seq_idx.unsqueeze(0)     # (1, L)
        B, L = seq_idx.shape

        # --- Input embedding ---
        if emb is not None:
            if emb.dim() == 2:
                emb = emb.unsqueeze(0)         # (1, L, D)
            # sanitize embeddings (NaN/Inf safe)
            emb = torch.nan_to_num(emb, nan=0.0, posinf=1e4, neginf=-1e4)
            want = self.embed.in_features
            x = self.embed(emb) if emb.shape[-1] != want else emb
        else:
            x = self.embed_oh(seq_idx)         # (B, L, ds)

        # --- Initialize pair representation ---
        d_pair = self.evo.blocks[0]["tri"].self_attn.embed_dim
        z = torch.zeros(B, L, L, d_pair, device=x.device)

        # --- Evoformer update ---
        s, z = self.evo(x, z)

        # --- IPA and heads ---
        xyz = self.ipa(s)                      # (B, L, 3)
        dist_logits = self.dist(z)             # (B, L, L, BINS)
        tors = self.tors(s)                    # (B, L, 3)
        plddt, pae = self.conf(s, z)           # (B, L), (B, L, L)

        return {
            "xyz": xyz.squeeze(0),
            "dist": dist_logits.squeeze(0),
            "tors": tors.squeeze(0),
            "plddt": plddt.squeeze(0),
            "pae": pae.squeeze(0),
            "s": s,
            "z": z,
        }

        