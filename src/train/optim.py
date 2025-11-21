# src/train/optim.py
# Defines optimizer and learning rate scheduler setup for model training.
# - Uses AdamW optimizer for stable weight updates with decoupled weight decay.
# - Applies a CosineAnnealingLR scheduler to gradually reduce the learning rate over training steps.
# Returns both optimizer and scheduler objects for use in the training loop.

import math, torch

def make_optim(model, tc):
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tc["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=tc["weight_decay"]
    )
    steps = int(tc["steps"])
    warm = int(tc.get("warmup_steps", 5000))

    def lr_lambda(step):
        if step < warm:
            return max(1e-8, (step + 1) / max(1, warm))  # linear warmup
        # cosine to zero after warmup
        t = (step - warm) / max(1, steps - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched
