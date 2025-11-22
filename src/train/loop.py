# src/train/loop.py
# Implements the main training loop for the ViroporinAFMini model.
# Handles forward passes, loss computation, backpropagation, gradient scaling (AMP), and checkpointing.
# - step_losses(): computes individual and combined losses (distogram, torsion, FAPE, membrane/pore priors).
# - fit(): iterates over training steps, updates model weights, logs progress, and saves checkpoints.
# - save(): safely writes model and optimizer state to disk for resuming or evaluation.
# Includes automatic mixed precision (AMP) and optional loss warmup for membrane/pore priors.

import torch, os
from tqdm import trange
from src.losses.distogram import distogram_loss
from src.losses.fape import fape_loss
from src.losses.torsion import torsion_l2
from src.losses.viroporin_priors import (
    membrane_z_mask, membrane_slab_loss, interface_contact_loss, ca_clash_loss, pore_target_loss
)
from src.geometry.assembly import assemble_cn

class Trainer:
    def __init__(self, cfg, model, opt, sched, device, start_step=0):
        self.start_step = int(start_step)
        self.global_step = int(start_step)
        self.cfg, self.model, self.opt, self.sched, self.device = cfg, model, opt, sched, device
        self.w = cfg["loss_weights"]; self.pr = cfg["priors"]
        self.use_cuda = (device.type == "cuda")
        self.amp_enabled_cfg = bool(self.cfg["train"].get("amp_enabled", False))
        self.amp_on_step     = int(self.cfg["train"].get("amp_on_step", 10000))
        # AMP dtype: bf16 if truly supported, else fp16 on CUDA; CPU runs fp32
        self.amp_dtype = (torch.bfloat16 if (self.use_cuda and torch.cuda.is_bf16_supported()) else
                          (torch.float16 if self.use_cuda else torch.float32))
        # New GradScaler API (fallback to old if needed)
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_cuda)
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_cuda)
        
        self.priors_warmup = int(self.cfg["train"].get("priors_warmup_steps", 0))
        self.global_step = 0
        if bool(self.cfg["train"].get("detect_anomaly", False)):
            torch.autograd.set_detect_anomaly(True)
            
    def _amp_active(self):
        return (self.use_cuda
                and self.amp_enabled_cfg
                and (self.global_step >= self.amp_on_step))
        
    def huber(x, delta=1.0):
        absx = torch.abs(x)
        return torch.where(absx < delta, 0.5*absx*absx/delta, absx - 0.5*delta)
        
    def step_losses(self, batch, out):
        L = out["xyz"].shape[0]
        loss = out["xyz"].new_tensor(0.0)
        logs = {}

        # Base structural losses
        loss_dist = distogram_loss(out["dist"], out["xyz"])
        loss_tors = torsion_l2(out["tors"])
        loss_fape = fape_loss(out["xyz"])

        for name, val in {"distogram": loss_dist, "torsion": loss_tors, "fape": loss_fape}.items():
            if not torch.isfinite(val):
                raise RuntimeError(f"[nan-guard] {name} is non-finite @ step {self.global_step}")

        loss = (
            self.w["distogram"] * loss_dist
            + self.w["torsion"] * loss_tors
            + self.w["fape"] * loss_fape
        )

        # ---- Viroporin priors ----
        gs = self.global_step
        if self.pr.get("use_cn", True):
            n, rr = self.pr["n_copies"], self.pr["ring_radius"]
            olig = assemble_cn(out["xyz"], n_copies=n, ring_radius=rr)
            tm_mask = membrane_z_mask(L, self.pr["tm_span"]).to(out["xyz"].device)

            mem   = membrane_slab_loss(out["xyz"], tm_mask)
            intf  = interface_contact_loss(olig, cutoff=9.0)
            clash = ca_clash_loss(olig, min_dist=3.6)
            pore  = pore_target_loss(olig, target_A=self.pr["pore_target_A"])

            # guard pore to avoid NaN/Inf
            if not torch.isfinite(pore):
                pore = torch.tensor(0.0, device=olig.device)

            # ---- Single consistent curriculum ----
            if gs < 2_000:
                w_mem = w_intf = w_pore = 0.0
            elif gs < 8_000:
                w_mem, w_intf, w_pore = 0.1, 0.0, 0.0
            else:
                t = min(1.0, (gs - 8_000) / 2_000.0)
                w_mem, w_intf, w_pore = 0.3 * t, 0.4 * t, 0.3 * t

            # ---- Combine once ----
            loss = loss + self.w["priors"] * (w_mem * mem + w_intf * intf + w_pore * pore) + 0.1 * clash

            # Logging
            logs.update(
                mem=mem.detach().cpu().item(),
                intf=intf.detach().cpu().item(),
                clash=clash.detach().cpu().item(),
                pore=pore.detach().cpu().item(),
            )

        return loss, logs

    def fit(self, train_loader, val_loader):
        steps = self.cfg["train"]["steps"]
        log_every = self.cfg["train"]["log_every"]
        eval_every = self.cfg["train"]["eval_every"]
        ckpt_dir = self.cfg["train"]["ckpt_dir"]

        pbar = trange(self.global_step, steps, desc="train")
        it = iter(train_loader)
        try:
            for step in pbar:
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(train_loader); batch = next(it)

                # non-blocking H2D copies for CUDA
                for k in batch:
                    v = batch[k]
                    if v is not None and hasattr(v, "to"):
                        batch[k] = v.to(self.device, non_blocking=True)

                self.global_step = step

                self.model.train()
                # forward
                use_amp = self._amp_active()
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=use_amp):
                    out = self.model(batch["seq_idx"], batch.get("emb"))
                    loss, logs = self.step_losses(batch, out)

                # backward
                self.opt.zero_grad(set_to_none=True)
                if use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["train"]["grad_clip"])
                    self.scaler.step(self.opt); self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["train"]["grad_clip"])
                    self.opt.step()
                if self.sched is not None:
                    self.sched.step()
                
                if step % log_every == 0:
                    pbar.set_postfix_str(
                        f"loss={loss.item():.2f}, "
                        + ", ".join([f"{k}={v:.3f}" for k, v in logs.items() if v is not None])
                    )

                if step and step % eval_every == 0:
                    # save a regular checkpoint
                    self.save(step, ckpt_dir)

                    # run validation
                    val = self.evaluate(val_loader)
                    msg = " | ".join([f"{k}={v:.3f}" for k,v in val.items()])
                    print(f"[val] step={step} | {msg}")

                    # track the best validation loss and save a 'best.pt'
                    best = getattr(self, "_best_val", float("inf"))
                    if val["loss_val"] < best:
                        self._best_val = val["loss_val"]
                        best_path = os.path.join(ckpt_dir, "best.pt")
                        self.save(step, ckpt_dir)  # keep step checkpoint
                        # also write/update a separate best file
                        state = {
                            "model": self.model.state_dict(),
                            "opt": self.opt.state_dict(),
                            "sched": self.sched.state_dict() if self.sched else None,
                            "scaler": self.scaler.state_dict(),
                            "cfg": self.cfg,
                            "step": int(step),
                            "best_val": float(self._best_val),
                        }
                        tmp = best_path + ".tmp"
                        torch.save(state, tmp); os.replace(tmp, best_path)
                        print(f"[val] new best checkpoint → {best_path} (loss_val={self._best_val:.4f})")

            # normal end
            self.save(steps, ckpt_dir)

        except KeyboardInterrupt:
            # ensure we always save something useful when you hit Ctrl+C
            safe_step = step if "step" in locals() else 0
            print(f"\n[info] interrupted @ step {safe_step} — saving checkpoint")
            self.save(safe_step, ckpt_dir)
            return

    def evaluate(self, val_loader):
        self.model.eval()
        tot, n = 0.0, 0
        agg = {"mem":0.0,"intf":0.0,"clash":0.0,"pore":0.0}
        with torch.no_grad():
            # run eval in full fp32 (more stable)
            for batch in val_loader:
                for k in batch:
                    v = batch[k]
                    if v is not None and hasattr(v, "to"):
                        batch[k] = v.to(self.device, non_blocking=True)
                out = self.model(batch["seq_idx"], batch.get("emb"))

                loss, logs = self.step_losses(batch, out)  # reuses same loss composition
                tot += float(loss.item()); n += 1
                for k in agg:
                    if k in logs:
                        agg[k] += float(logs[k])

        val = {"loss_val": tot/max(1,n)}
        for k in agg:
            if n > 0:
                val[f"{k}_val"] = agg[k]/n
        return val


    def save(self, step, ckpt_dir):
        state = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "sched": self.sched.state_dict() if self.sched else None,
            "scaler": self.scaler.state_dict(),
            "cfg": self.cfg,
            "step": int(step),
        }
        fn = os.path.join(ckpt_dir, f"step_{step}.pt")
        tmp = fn + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, fn)  
    
    def _priors_weight(self):
        w = float(self.w["priors"])
        if self.priors_warmup > 0:
            t = min(1.0, self.global_step / max(1, self.priors_warmup))
            return w * t
        return w