# src/train/optim.py
# Defines optimizer and learning rate scheduler setup for model training.
# - Uses AdamW optimizer for stable weight updates with decoupled weight decay.
# - Applies a CosineAnnealingLR scheduler to gradually reduce the learning rate over training steps.
# Returns both optimizer and scheduler objects for use in the training loop.

import torch

def make_optim(model, tc):
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["steps"])
    return opt, sched
