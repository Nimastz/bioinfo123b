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
from src.model.msa_block import TinyMSAEncoder, TinyMSABlock 

class ViroporinAFMini(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ds, dp = cfg["d_single"], cfg["d_pair"]
        d_msa = cfg.get("d_msa", 128)  # small MSA dim, default 128

        # sequence / embedding
        self.embed = nn.Linear(cfg.get("feat_dim", 480), ds)  
        self.embed_oh = nn.Embedding(21, ds)

        # --- NEW: tiny MSA branch ---
        self.msa_encoder = TinyMSAEncoder(d_msa=d_msa, vocab_size=22)        # 20 aa + gap + pad
        self.msa_block   = TinyMSABlock(d_msa=d_msa, n_heads=4)              # light row/col attn
        self.msa_to_single = nn.Linear(d_msa, ds)                            # MSA → single
        self.msa_to_pair   = nn.Linear(d_msa, dp)                            # MSA → pair

        # evoformer + heads
        self.evo = EvoformerLite(
            ds, dp,
            cfg["evoformer"]["n_blocks"],
            cfg["evoformer"]["n_attn_heads"],
        )
        self.ipa = IPA(ds + dp, cfg["ipa"]["n_blocks"], cfg["ipa"]["n_points"], cfg["ipa"]["dropout"])
        self.dist = DistogramHead(dp, cfg["heads"]["distogram_bins"])
        self.tors = TorsionHeadSimple(ds)
        self.conf = ConfidenceHeads(ds, dp)


    def forward(self, seq_idx, emb=None, msa=None):
        """
        seq_idx: (L,) or (B, L) int tokens
        emb:     optional ESM/other embeddings, (L, D) or (B, L, D)
        msa:     optional MSA tokens, (B, N, L) with PAD/gap indices
        """
        # Accept both (L,) and (B,L)
        if isinstance(seq_idx, dict):
            # safety if someone passes the whole batch dict by mistake
            emb = seq_idx.get("emb", emb)
            msa = seq_idx.get("msa", msa)
            seq_idx = seq_idx["seq_idx"]

        if seq_idx.dim() == 1:
            seq_idx = seq_idx.unsqueeze(0)     # (1, L)
        B, L = seq_idx.shape

        # --- Input embedding ---
        if emb is not None:
            if emb.dim() == 2:
                emb = emb.unsqueeze(0)         # (1, L, D)
            emb = torch.nan_to_num(emb, nan=0.0, posinf=1e4, neginf=-1e4)
            want = self.embed.in_features
            x = self.embed(emb) if emb.shape[-1] != want else emb  # (B, L, ds)
        else:
            x = self.embed_oh(seq_idx)         # (B, L, ds)

        # --- Initialize pair representation ---
        d_pair = self.evo.blocks[0]["tri"].self_attn.embed_dim
        z = torch.zeros(B, L, L, d_pair, device=x.device)

        # --- NEW: MSA injection (if available) ---
        if msa is not None:
            # expect msa: (B, N, L)
            if msa.dim() == 2:
                # rare case: (N, L) → assume B=1
                msa = msa.unsqueeze(0)

            # (B, N, L, d_msa)
            msa_repr = self.msa_encoder(msa)
            msa_repr = self.msa_block(msa_repr)        # tiny row+col attention

            # average over MSA sequences: (B, L, d_msa)
            msa_mean = msa_repr.mean(dim=1)

            # inject into single features
            x = x + self.msa_to_single(msa_mean)       # (B, L, ds)

            # inject a simple MSA-derived bias into pair features
            a = self.msa_to_pair(msa_mean)             # (B, L, dp)
            z = z + a.unsqueeze(2) + a.unsqueeze(1)    # (B, L, L, dp)

        # --- Evoformer update ---
        s, z = self.evo(x, z)

        # --- IPA and heads ---
        z_row = z.mean(dim=2)                  # (B, L, d_pair)
        s_ipa = torch.cat([s, z_row], dim=-1)  # (B, L, ds + dp)
        xyz = self.ipa(s_ipa)                  # (B, L, 3)
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

        