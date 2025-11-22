# src/train/loop.py
# Implements the main training loop for the ViroporinAFMini model.
# Handles forward passes, loss computation, backpropagation, gradient scaling (AMP), and checkpointing.
# - step_losses(): computes individual and combined losses (distogram, torsion, FAPE, membrane/pore priors).
# - fit(): iterates over training steps, updates model weights, logs progress, and saves checkpoints.
# - save(): safely writes model and optimizer state to disk for resuming or evaluation.
# Includes automatic mixed precision (AMP) and optional loss warmup for membrane/pore priors.

import math
import torch, os
from tqdm import trange
from src.losses.distogram import distogram_loss
from src.losses.fape import fape_loss
from src.losses.torsion import torsion_l2
from src.losses.viroporin_priors import (
    membrane_z_mask, membrane_slab_loss, interface_contact_loss, ca_clash_loss, pore_target_loss
)
from src.geometry.assembly import assemble_cn
from src.utils.logger import CSVLogger
import json, numpy as np

class Trainer:
    def __init__(self, cfg, model, opt, sched, device, start_step=0):
        self.start_step = int(start_step)
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
        
        # ---- logging ----
        log_dir = self.cfg["train"].get("log_dir", "checkpoints/logs")
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "train_steps.csv")
        csv_fields = ["step","loss","lr","dist","tors","fape",
              "mem","pore","mem_raw","pore_raw","intf","clash"]
        self.csv = CSVLogger(csv_path, fieldnames=csv_fields)
        self.best_ema = float("inf")
        self.loss_ema = None
        self.ema_alpha = float(self.cfg["train"].get("ema_alpha", 0.1))
            
    def _amp_active(self):
        return (self.use_cuda
                and self.amp_enabled_cfg
                and (self.global_step >= self.amp_on_step))
        
    @staticmethod    
    def huber(x, delta=1.0):
        absx = torch.abs(x)
        return torch.where(absx < delta, 0.5*absx*absx/delta, absx - 0.5*delta)
        
    def step_losses(self, batch, out):
        L = out["xyz"].shape[0]
        loss = out["xyz"].new_tensor(0.0)
        logs = {}

        # Base structural losses
        loss_dist = distogram_loss(out["dist"], out["xyz"], label_smoothing=0.01)
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
        if self.pr.get("use_cn", True):
            n, rr = self.pr["n_copies"], self.pr["ring_radius"]
            olig = assemble_cn(out["xyz"], n_copies=n, ring_radius=rr)
            tm_mask = membrane_z_mask(L, self.pr["tm_span"]).to(out["xyz"].device)

            # --- robust TM-only centering for mem prior ---
            z_all = out["xyz"][:, 2]
            z_tm  = z_all[tm_mask > 0.5]
            z_center = (z_tm.median() if z_tm.numel() > 0 else z_all.median()).detach()

            xyz_centered = out["xyz"].clone()
            xyz_centered[:, 2] -= z_center

            # single compute
            mem_raw  = membrane_slab_loss(xyz_centered, tm_mask)
            mem_eff  = 5.0 * torch.tanh(mem_raw / 5.0)

            intf  = interface_contact_loss(olig, cutoff=9.0)
            clash = ca_clash_loss(olig, min_dist=3.6)
            pore  = pore_target_loss(olig, target_A=self.pr["pore_target_A"])
            if not torch.isfinite(pore):
                pore = torch.tensor(0.0, device=olig.device)
            pore_raw  = pore
            pore_eff  = 5.0 * torch.tanh(pore_raw / 5.0)

            # ---- Smooth prior warmup ----
            pw = self._priors_weight()   # global factor (0..w_priors)

            base_mem  = float(self.w.get("membrane",  0.1))
            base_intf = float(self.w.get("interface", 0.1))
            base_pore = float(self.w.get("pore",     0.1))

            t = self.global_step
            W = max(1, int(self.priors_warmup))
            w_mem  = base_mem  * min(1.0, t / W)
            w_intf = base_intf * min(1.0, t / (0.5*W))
            w_pore = base_pore * min(1.0, max(0.0, (t - 0.5*W) / W))

            loss = loss + pw * (w_mem * mem_eff + w_intf * intf + w_pore * pore_eff) + 0.1 * clash

            # Logging
            logs.update(
                mem=float(mem_eff.detach().cpu()),
                mem_raw=float(mem_raw.detach().cpu()),
                pore=float(pore_eff.detach().cpu()),
                pore_raw=float(pore_raw.detach().cpu()),
                intf=float(intf.detach().cpu()),
                clash=float(clash.detach().cpu()),
                dist=float(loss_dist.detach().cpu()),
                tors=float(loss_tors.detach().cpu()),
                fape=float(loss_fape.detach().cpu()),
            )

        return loss, logs

    def fit(self, train_loader, val_loader):
        steps = self.cfg["train"]["steps"]
        log_every = self.cfg["train"]["log_every"]
        eval_every = self.cfg["train"]["eval_every"]
        ckpt_dir = self.cfg["train"]["ckpt_dir"]

        pbar = trange(self.start_step, steps, desc="train")
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
                lr = self.opt.param_groups[0]["lr"]
                
                row = {
                    "step": int(step),
                    "loss": float(loss.item()),
                    "lr": float(lr),
                    "dist": float(logs.get("dist", float("nan"))),
                    "tors": float(logs.get("tors", float("nan"))),
                    "fape": float(logs.get("fape", float("nan"))),
                    "mem":  float(logs.get("mem",  float("nan"))),
                    "pore": float(logs.get("pore", float("nan"))),
                    "mem_raw":  float(logs.get("mem_raw",  float("nan"))),
                    "pore_raw": float(logs.get("pore_raw", float("nan"))),
                    "intf": float(logs.get("intf", float("nan"))),
                    "clash":float(logs.get("clash",float("nan"))),
                    }
                # write every step (or guard with: if step % log_every == 0:)
                self.csv.log(row)
                
                if step % log_every == 0:
                    pbar.set_postfix_str(
                        f"loss={loss.item():.2f} "
                        + ", ".join([f"{k}={v:.3f}" for k, v in logs.items() if v is not None])
                    )

                if step and step % eval_every == 0:
                    self.save(step, ckpt_dir)

                # ---- end-of-training summary (written once) ----
                if (step % 200 == 0) or (step == steps - 1):
                    ema_loss = self.loss_ema if self.loss_ema is not None else float(loss.item())
                    cur = float(loss.item())
                    self.loss_ema = cur if self.loss_ema is None else (1.0 - self.ema_alpha) * self.loss_ema + self.ema_alpha * cur
                    cap = float(self.pr.get("cap_A", 6.0))
                    mem_ratio  = min(1.0, (logs.get("mem", 0.0)) / cap)
                    pore_ratio = min(1.0, (logs.get("pore", 0.0)) / cap)
                    score = ema_loss + 0.2 * max(0.0, mem_ratio - 0.7) + 0.2 * max(0.0, pore_ratio - 0.7)
                    with open(os.path.join(ckpt_dir, "summary.json"), "w", encoding="utf-8") as f:
                        json.dump({"score": float(score),
                                "ema_loss": float(ema_loss),
                                "mem_ratio": float(mem_ratio),
                                "pore_ratio": float(pore_ratio),
                                "steps": int(step+1)}, f, indent=2)

        except KeyboardInterrupt:
            # ensure we always save something useful when you hit Ctrl+C
            safe_step = step if "step" in locals() else 0
            print(f"\n[info] interrupted @ step {safe_step} — saving checkpoint")
            self.save(safe_step, ckpt_dir)
            return
        
        finally:
                # make sure file handle is flushed/closed even on errors
                if hasattr(self, "csv") and self.csv:
                    self.csv.close()


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
            return w * 0.5 * (1.0 - math.cos(math.pi * t))
        return w