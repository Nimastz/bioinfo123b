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
        # s0: prefer provided per-residue embedding; else one-hot
        if emb is not None:
            s = self.embed(emb) if emb.shape[-1] != self.embed.out_features else emb
        else:
            s = self.embed_oh(seq_idx)
        L = s.shape[0]
        z = torch.zeros(L, L, self.evo.blocks[0]["tri"].self_attn.embed_dim, device=s.device)
        s, z = self.evo(s, z)
        xyz = self.ipa(s)               # (L,3)
        dist_logits = self.dist(z)      # (L,L,B)
        tors = self.tors(s)             # (L,3)
        plddt, pae = self.conf(s, z)    # (L,), (L,L)
        return {"xyz": xyz, "dist": dist_logits, "tors": tors, "plddt": plddt, "pae": pae, "s": s, "z": z}
