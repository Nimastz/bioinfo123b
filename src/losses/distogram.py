import torch, torch.nn.functional as F

def distogram_loss(dist_logits, xyz, bin_size=0.5, n_bins=64, dmax=None):
    # Teacher = distances from xyz -> soft bin CE
    L = xyz.shape[0]
    D = torch.cdist(xyz, xyz)
    if dmax is None:
        dmax = bin_size * n_bins
    bins = torch.clamp((D / bin_size).long(), 0, n_bins-1)
    ce = F.cross_entropy(dist_logits.view(-1, n_bins), bins.view(-1), reduction="mean")
    return ce
