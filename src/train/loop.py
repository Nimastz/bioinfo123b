# src/train/loop.py
# Implements the main training loop for the ViroporinAFMini model.
# Handles forward passes, loss computation, backpropagation, gradient scaling (AMP), and checkpointing.
# - step_losses(): computes individual and combined losses (distogram, torsion, FAPE, membrane/pore priors).
# - fit(): iterates over training steps, updates model weights, logs progress, and saves checkpoints.
# - save(): safely writes model and optimizer state to disk for resuming or evaluation.
# Includes automatic mixed precision (AMP) and optional loss warmup for membrane/pore priors.
#  try: python train.py --config configs/recommended.yaml --ckpt checkpoints/step_25000.pt

import math
import torch, os
from tqdm import trange, tqdm
from src.losses.distogram import distogram_loss
from src.losses.fape import fape_loss
from src.geometry.torsion import torsion_loss
from src.losses.viroporin_priors import (
    membrane_z_mask, membrane_slab_loss, interface_contact_loss, ca_clash_loss, pore_target_loss, self_ca_clash_loss
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
        csv_fields = [
            "step","loss","lr",
            "dist","tors","fape",
            "mem","mem_raw","pore","pore_raw","intf","clash",
            "gate","tm_frac","z_abs_mean","z_abs_mean_tm","z_abs_mean_unc","z_tm_min","z_tm_max","z_tm_range",
            "mem_hinge_mean_tm","mem_hinge_max_tm",
            "pw_global","w_mem_lin","w_intf_lin","w_pore_lin","w_mem_eff","w_intf_eff","w_pore_eff",
            "pore_minA"
        ]
        if os.path.exists(csv_path):
            try:
                os.remove(csv_path)
                print(f"[info] old CSV log removed: {csv_path}")
            except Exception as e:
                print(f"[warn] could not remove old CSV: {e}")
        self.csv = CSVLogger(csv_path, fieldnames=csv_fields)
        self.best_ema = float("inf")
        self.loss_ema = None
        self.ema_alpha = float(self.cfg["train"].get("ema_alpha", 0.1))

        # ---- best tracking (global) ----
        self.best_metric_name = str(self.cfg["train"].get("best_metric", "loss"))  # "loss" or "score"
        self.best_metric_value = float("inf")
        self.best_step = -1

        # ---- window-best for logging (every log_every steps) ----
        self.win_best_metric_value = float("inf")
        self.win_best_step = -1
        self.win_best_snapshot = {}

        # ---- checkpoint window-best (e.g. best of each 1000-step block) ----
        # default window size = 1000 if not set in YAML
        self.ckpt_window = int(self.cfg["train"].get("ckpt_window", 1000))
        self.ckpt_best_metric_value = float("inf")
        self.ckpt_best_step = -1

    def _compute_score(self, ema_loss: float, logs: dict) -> float:
        # same score you already write to summary.json
        cap = float(self.pr.get("cap_A", 6.0))
        mem_ratio  = min(1.0, (logs.get("mem", 0.0)) / cap)
        pore_ratio = min(1.0, (logs.get("pore", 0.0)) / cap)
        return ema_loss + 0.2 * max(0.0, mem_ratio - 0.7) + 0.2 * max(0.0, pore_ratio - 0.7)

    def _current_metric(self, step: int, logs: dict, ema_loss: float) -> float:
        if self.best_metric_name == "score":
            return self._compute_score(ema_loss, logs)
        # default: "loss"
        return float(logs.get("loss", float("inf")))

    def _amp_active(self):
        return (self.use_cuda
                and self.amp_enabled_cfg
                and (self.global_step >= self.amp_on_step))
        
    @staticmethod    
    def huber(x, delta=1.0):
        absx = torch.abs(x)
        return torch.where(absx < delta, 0.5*absx*absx/delta, absx - 0.5*delta)
        
    def step_losses(self, batch, out):
        """
        Compute losses with batch support.
        Preserves original behavior for unbatched outputs (L,3)/(L,L,BINS).
        """
        logs = {}

        # -------- [CHANGED] Normalize outputs to batched tensors --------
        xyz = out["xyz"]
        dist_logits = out["dist"]
        tors_out = out["tors"]

        if xyz.dim() == 2:            # (L,3) -> (1,L,3)
            xyz = xyz.unsqueeze(0)
            dist_logits = dist_logits.unsqueeze(0)
            tors_out = tors_out.unsqueeze(0)

        B, L, _ = xyz.shape
        device = xyz.device

        # Accumulators for averaging across batch
        total_loss = torch.zeros((), device=device)
        acc = {
            "dist": 0.0, "tors": 0.0, "fape": 0.0,
            "mem": 0.0, "mem_raw": 0.0, "pore": 0.0, "pore_raw": 0.0, "intf": 0.0, "clash": 0.0,
        }
        n_prior = 0

        # -------- Loop per sample to keep priors on (L,3)/(L,) API --------
        for b in range(B):
            # per-sample tensors
            xyz_b  = xyz[b]            # (L,3)
            dist_b = dist_logits[b]    # (L,L,BINS)
            tors_b = tors_out[b]       # (L,3)

            # --- enforce CA–CA ≈ 3.8 Å during training ---
            with torch.no_grad():
                if xyz_b.shape[0] > 1:
                    diffs = xyz_b[1:] - xyz_b[:-1]          # (L-1,3)
                    dists = diffs.norm(dim=-1)              # (L-1)
                    d_mean = dists.mean()
                    if d_mean > 1e-6:
                        scale = (3.8 / d_mean).clamp(0.25, 4.0)
                        xyz_b[:] = xyz_b * scale

            # ---- valid-residue mask (PAD token = 20) ----
            if batch["seq_idx"].dim() == 2:
                val_mask = (batch["seq_idx"][b] != 20)     # (L,)
            else:
                val_mask = (batch["seq_idx"] != 20)        # (L,)

            n_valid = int(val_mask.sum().item())
            if n_valid < 2:
                continue

            # pair mask for distogram
            pair_mask = val_mask[:, None] & val_mask[None, :]
            if pair_mask.sum() == 0:
                continue

            # --- teacher xyz? ---
            has_teacher = False
            xyz_ref_b = None
            if "xyz_ref" in batch and batch["xyz_ref"] is not None:
                xyz_ref_b = batch["xyz_ref"][b]
                if xyz_ref_b is not None and torch.isfinite(xyz_ref_b).all():
                    has_teacher = True

            # --- target coords for distogram ---
            if has_teacher:
                xyz_target = xyz_ref_b        # teacher-supervised
            else:
                xyz_target = xyz_b            # self-distogram

            # --- DISTOGRAM ---
            loss_dist = distogram_loss(
                dist_b,
                xyz_target,
                label_smoothing=0.01,
                pair_mask=pair_mask,
            )

            # --- TORSION ---
            if "tors_ref" in batch and batch["tors_ref"] is not None:
                tors_ref_b = batch["tors_ref"][b]
                tors_ok = torch.isfinite(tors_ref_b).all(dim=-1)
                use_t = tors_ok & val_mask
                if use_t.any():
                    loss_tors = torsion_loss(tors_b[val_mask], helix_bias=True)
                else:
                    loss_tors = torsion_loss(tors_b[val_mask], helix_bias=True)
            else:
                loss_tors = torsion_loss(tors_b[val_mask], helix_bias=True)

            # --- FAPE ---
            if has_teacher:
                use = val_mask
                loss_fape = fape_loss(xyz_b[use], xyz_ref_b[use])
                w_fape_eff = self.w["fape"]
            else:
                loss_fape = fape_loss(xyz_b[val_mask])   # self-FAPE
                w_fape_eff = 0.05 * self.w["fape"]

            # nan-guard
            for name, val in {"distogram": loss_dist, "torsion": loss_tors, "fape": loss_fape}.items():
                if not torch.isfinite(val):
                    raise RuntimeError(f"[nan-guard] {name} is non-finite @ step {self.global_step}, sample {b}")

            # --- base loss ---
            loss_b = (
                self.w["distogram"] * loss_dist
                + self.w["torsion"]  * loss_tors
                + w_fape_eff         * loss_fape
            )

            # --- NEW: self CA–CA clash penalty (within chain) ---
            if n_valid > 4:  # small guard
                self_clash = self_ca_clash_loss(xyz_b[val_mask])
                loss_b = loss_b + 0.05 * self_clash
                # optional: if you *want* to include this in "clash" metric:
                # acc["clash"] += float(self_clash.detach().cpu())

            # ---- Viroporin priors (CN assembly + membrane / pore / interface) ----
            if self.pr.get("use_cn", True):
                n_copies   = self.pr["n_copies"]
                ring_radius = self.pr["ring_radius"]

                # build Cn oligomer
                olig = assemble_cn(xyz_b, n_copies=n_copies, ring_radius=ring_radius)

                # TM mask
                tm_mask = membrane_z_mask(L, self.pr["tm_span"]).to(device)  # (L,)
                tm_mask = tm_mask * val_mask.float()
                tm_frac = float(tm_mask.float().mean().item())
                logs["tm_frac"] = float(tm_mask.mean().item())

                # --- robust TM-only centering & diagnostics ---
                z_all   = xyz_b[:, 2]
                tm_bool = (tm_mask > 0.5) & val_mask
                val_bool = val_mask

                # center on TM median if possible, else all-valid median
                if tm_bool.any():
                    z_center = z_all[tm_bool].median().detach()
                else:
                    z_center = z_all[val_bool].median().detach()

                xyz_centered = xyz_b.clone()
                xyz_centered[:, 2] -= z_center

                # diagnostics (prefer TM, fall back to all-valid)
                if tm_bool.any():
                    zc_tm = xyz_centered[tm_bool, 2]
                    logs["z_abs_mean_tm"] = float(zc_tm.abs().mean().item())
                else:
                    zc_tm = None
                    logs["z_abs_mean_tm"] = float("nan")

                zc_all = xyz_centered[val_bool, 2]
                logs["z_abs_mean"] = float(zc_all.abs().mean().item())

                use_z = zc_tm if zc_tm is not None else zc_all
                logs["z_tm_min"]   = float(use_z.min().item())
                logs["z_tm_max"]   = float(use_z.max().item())
                logs["z_tm_range"] = float((use_z.max() - use_z.min()).item())

                # uncentered drift
                if tm_bool.any():
                    logs["z_abs_mean_unc"] = float(z_all[tm_bool].abs().mean().item())
                else:
                    logs["z_abs_mean_unc"] = float(z_all[val_bool].abs().mean().item())

                # pore radius profile
                from src.geometry.assembly import pore_radius_profile_ca
                zs, rs = pore_radius_profile_ca(olig)
                if rs.numel() > 0:
                    finite_rs = rs[torch.isfinite(rs)]
                    if finite_rs.numel() > 0:
                        logs["pore_minA"] = float(finite_rs.min().item())

                # optional debug
                if b == 0 and self.global_step % 100 == 0:
                    zmin = xyz_centered[:, 2].min().item()
                    zmax = xyz_centered[:, 2].max().item()
                    print(f"[dbg] z_centered range: {zmin:.2f}..{zmax:.2f}")

                # ---- membrane / interface / pore priors ----
                mem_raw  = membrane_slab_loss(xyz_centered, tm_mask)
                intf     = interface_contact_loss(olig, cutoff=9.0)
                clash    = ca_clash_loss(olig, min_dist=3.6)
                pore     = pore_target_loss(olig, target_A=self.pr["pore_target_A"])
                if not torch.isfinite(pore):
                    pore = torch.tensor(0.0, device=olig.device)
                pore_raw = pore

                # ---- prior weights (using your config) ----
                base_mem  = float(self.w.get("membrane",  0.05))
                base_intf = float(self.w.get("interface", 0.05))
                base_pore = float(self.w.get("pore",      0.05))

                # if you later want true warmup, use priors_warmup; for now turn fully on:
                w_mem_lin  = base_mem
                w_intf_lin = base_intf
                w_pore_lin = base_pore

                pw_global = self._priors_weight()
                gate = 1.0

                w_mem_eff  = pw_global * gate * max(w_mem_lin,  base_mem)
                w_intf_eff = pw_global * gate * max(w_intf_lin, base_intf)
                w_pore_eff = pw_global * gate * max(w_pore_lin, base_pore)

                # TM-fraction gating
                gate_tm   = max(0.0, min(1.0, (tm_frac - 0.02) / (0.8 - 0.02)))
                w_mem_eff *= float(gate_tm)

                logs["pw_global"]  = float(pw_global)
                logs["w_mem_lin"]  = float(w_mem_lin)
                logs["w_intf_lin"] = float(w_intf_lin)
                logs["w_pore_lin"] = float(w_pore_lin)
                logs["w_mem_eff"]  = float(w_mem_eff)
                logs["w_intf_eff"] = float(w_intf_eff)
                logs["w_pore_eff"] = float(w_pore_eff)
                logs["gate"]       = float(gate)

                # ---- clamps & tanh smoothing ----
                mem_raw  = mem_raw.clamp(-10, 10)
                pore_raw = pore_raw.clamp(-10, 10)
                mem_eff  = 5.0 * torch.tanh(mem_raw / 5.0)
                pore_eff = 5.0 * torch.tanh(pore_raw / 5.0)

                prior_contrib = (w_mem_eff * mem_eff) + (w_intf_eff * intf) + (w_pore_eff * pore_eff)
                prior_contrib = torch.clamp(prior_contrib, min=-0.5, max=0.5)

                # add priors + oligomer clash
                loss_b = loss_b + prior_contrib + 0.1 * clash

                # accumulate metrics
                acc["mem"]      += float(mem_eff.detach().cpu())
                acc["mem_raw"]  += float(mem_raw.detach().cpu())
                acc["pore"]     += float(pore_eff.detach().cpu())
                acc["pore_raw"] += float(pore_raw.detach().cpu())
                acc["intf"]     += float(intf.detach().cpu())
                acc["clash"]    += float(clash.detach().cpu())
                n_prior += 1

            # accumulate base loss metrics
            acc["dist"] += float(loss_dist.detach().cpu())
            acc["tors"] += float(loss_tors.detach().cpu())
            acc["fape"] += float(loss_fape.detach().cpu())

            total_loss = total_loss + loss_b

        # -------- [CHANGED] Average over batch and finalize logs --------
        total_loss = total_loss / B

        # average base losses across batch; priors across the #samples that had them
        logs.update({
            "dist": acc["dist"] / B,
            "tors": acc["tors"] / B,
            "fape": acc["fape"] / B,
            "mem":      (acc["mem"]      / max(1, n_prior)),
            "mem_raw":  (acc["mem_raw"]  / max(1, n_prior)),
            "pore":     (acc["pore"]     / max(1, n_prior)),
            "pore_raw": (acc["pore_raw"] / max(1, n_prior)),
            "intf":     (acc["intf"]     / max(1, n_prior)),
            "clash":    (acc["clash"]    / max(1, n_prior)),
            # geometry diagnostics
            "z_abs_mean": logs.get("z_abs_mean", float("nan")),
            "z_abs_mean_tm": logs.get("z_abs_mean_tm", float("nan")),
            "pore_minA": logs.get("pore_minA", float("nan")),
            "gate": logs.get("gate", float("nan")),
        })

        return total_loss, logs

    def fit(self, train_loader, val_loader):
        steps = self.cfg["train"]["steps"]
        log_every = self.cfg["train"]["log_every"]
        eval_every = self.cfg["train"]["eval_every"]
        ckpt_dir = self.cfg["train"]["ckpt_dir"]
        accum_steps = int(self.cfg["train"].get("grad_accum_steps", 1))

        pbar = trange(self.start_step, steps, desc="", bar_format="train: {n_fmt}/{total_fmt} {postfix}",miniters=log_every)
        it = iter(train_loader)
        try:
            for step in pbar:
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(train_loader); batch = next(it)

                # ---- one-time batch debug ----
                if step == self.start_step:
                    print("\n====== TRAIN BATCH DEBUG ======")
                    print("seq_idx:", batch["seq_idx"].shape)
                    print("emb:", None if batch.get("emb") is None else batch["emb"].shape)
                    print("msa:", None if batch.get("msa") is None else batch["msa"].shape)
                    print("xyz_ref:", None if batch.get("xyz_ref") is None else batch["xyz_ref"].shape)
                    print("tors_ref:", None if batch.get("tors_ref") is None else batch["tors_ref"].shape)
                    print("================================")
                    
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
                    out = self.model(batch["seq_idx"], batch.get("emb"),msa=batch.get("msa"))
                    if ("xyz" in out) and (not torch.isfinite(out["xyz"]).all()):
                        print(f"[nan-guard] non-finite xyz @ step {self.global_step} — skipping update")
                        self.opt.zero_grad(set_to_none=True)
                        continue
                    for key in ("dist", "xyz", "tors"):
                        if key in out:
                            out[key] = torch.nan_to_num(out[key], nan=0.0, posinf=1e4, neginf=-1e4)
                
                loss, logs = self.step_losses(batch, out)
                if not torch.isfinite(loss):
                    print(f"[warn] Non-finite loss at step {step}; skipping update.")
                    self.opt.zero_grad(set_to_none=True)
                    continue
                # ---- backward & optim (gradient accumulation) ----
                # zero grads only at the START of an accumulation window
                if (step % accum_steps) == 0:
                    self.opt.zero_grad(set_to_none=True)

                # scale loss for accumulation
                pre_accum_loss = loss.detach().clone().float() 
                logs["loss"] = float(pre_accum_loss.item()) 
                # --- update window-best (100-step) every step ---
                if self.best_metric_name == "score":
                    ema_now = float(self.loss_ema) if (self.loss_ema is not None) else float(logs["loss"])
                    metric_now = self._compute_score(ema_now, logs)
                else:
                    metric_now = float(logs["loss"])

                if metric_now < self.win_best_metric_value:
                    self.win_best_metric_value = metric_now
                    self.win_best_step = step
                    # snapshot what we want to display for the window-best line
                    self.win_best_snapshot = {
                        "loss": logs.get("loss", 0.0),
                        "tors": logs.get("tors", 0.0),
                        "dist": logs.get("dist", 0.0),
                        "fape": logs.get("fape", 0.0),
                        "mem": logs.get("mem", 0.0),
                        "mem_raw": logs.get("mem_raw", 0.0),
                        "pore": logs.get("pore", 0.0),
                        "pore_raw": logs.get("pore_raw", 0.0),
                        "pore_minA": logs.get("pore_minA", 0.0),
                        "intf": logs.get("intf", 0.0),
                        "clash": logs.get("clash", 0.0),
                    }
                # --- NEW: checkpoint-window best (e.g. best of each 1000-step block) ---
                if self.ckpt_window > 0:
                    # update best in current ckpt window
                    if metric_now < self.ckpt_best_metric_value:
                        self.ckpt_best_metric_value = metric_now
                        self.ckpt_best_step = step

                    # if we hit the boundary of a window, save that window's best
                    if step > 0 and (step % self.ckpt_window == 0):
                        if self.ckpt_best_step >= 0:
                            self.save(self.ckpt_best_step, ckpt_dir)
                        # reset for next window
                        self.ckpt_best_metric_value = float("inf")
                        self.ckpt_best_step = -1

                loss = loss / accum_steps

                if use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % accum_steps == 0:
                    if use_amp:
                        self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["train"]["grad_clip"])

                    if use_amp:
                        self.scaler.step(self.opt)
                        self.scaler.update()
                    else:
                        self.opt.step()

                    if self.sched is not None:
                        self.sched.step()

                # read LR after (possible) step
                lr = self.opt.param_groups[0]["lr"]

                row = {
                    "step": int(step),
                    "loss": float(pre_accum_loss.item()),
                    "lr": float(lr),
                    "dist": float(logs.get("dist", float("nan"))),
                    "tors": float(logs.get("tors", float("nan"))),
                    "fape": float(logs.get("fape", float("nan"))),
                    "mem":  float(logs.get("mem",  float("nan"))),
                    "mem_raw":  float(logs.get("mem_raw",  float("nan"))),
                    "pore": float(logs.get("pore", float("nan"))),
                    "pore_raw": float(logs.get("pore_raw", float("nan"))),
                    "intf": float(logs.get("intf", float("nan"))),
                    "clash": float(logs.get("clash", float("nan"))),
                    "z_abs_mean": float(logs.get("z_abs_mean", float("nan"))),
                    "z_abs_mean_tm": float(logs.get("z_abs_mean_tm", float("nan"))),
                    "z_abs_mean_unc": float(logs.get("z_abs_mean_unc", float("nan"))),
                    "pore_minA": float(logs.get("pore_minA", float("nan"))),
                    "gate": float(logs.get("gate", float("nan"))),
                    "z_tm_min": float(logs.get("z_tm_min", float("nan"))),
                    "z_tm_max": float(logs.get("z_tm_max", float("nan"))),
                    "z_tm_range": float(logs.get("z_tm_range", float("nan"))),
                    "tm_frac": float(logs.get("tm_frac", float("nan"))),
                    "pw_global": float(logs.get("pw_global", float("nan"))),
                    "w_mem_lin": float(logs.get("w_mem_lin", float("nan"))),
                    "w_intf_lin": float(logs.get("w_intf_lin", float("nan"))),
                    "w_pore_lin": float(logs.get("w_pore_lin", float("nan"))),
                    "w_mem_eff": float(logs.get("w_mem_eff", float("nan"))),
                    "w_intf_eff": float(logs.get("w_intf_eff", float("nan"))),
                    "w_pore_eff": float(logs.get("w_pore_eff", float("nan"))),
                }
                # write every log_every steps: CSV row = BEST of the last window, not last step
                if step % log_every == 0:
                    if self.win_best_snapshot:
                        src = self.win_best_snapshot
                        step_for_row = self.win_best_step
                    else:
                        # fallback: if for some reason no best was recorded, use current logs
                        src = logs
                        step_for_row = step

                    row = {
                        "step": int(step_for_row),
                        "loss": float(src.get("loss", float("nan"))),
                        "lr": float(lr),
                        "dist": float(src.get("dist", float("nan"))),
                        "tors": float(src.get("tors", float("nan"))),
                        "fape": float(src.get("fape", float("nan"))),
                        "mem":  float(src.get("mem",  float("nan"))),
                        "mem_raw":  float(src.get("mem_raw",  float("nan"))),
                        "pore": float(src.get("pore", float("nan"))),
                        "pore_raw": float(src.get("pore_raw", float("nan"))),
                        "intf": float(src.get("intf", float("nan"))),
                        "clash": float(src.get("clash", float("nan"))),
                        "z_abs_mean": float(logs.get("z_abs_mean", float("nan"))),
                        "z_abs_mean_tm": float(logs.get("z_abs_mean_tm", float("nan"))),
                        "z_abs_mean_unc": float(logs.get("z_abs_mean_unc", float("nan"))),
                        "pore_minA": float(src.get("pore_minA", float("nan"))),
                        "gate": float(logs.get("gate", float("nan"))),
                        "z_tm_min": float(logs.get("z_tm_min", float("nan"))),
                        "z_tm_max": float(logs.get("z_tm_max", float("nan"))),
                        "z_tm_range": float(logs.get("z_tm_range", float("nan"))),
                        "tm_frac": float(logs.get("tm_frac", float("nan"))),
                        "pw_global": float(logs.get("pw_global", float("nan"))),
                        "w_mem_lin": float(logs.get("w_mem_lin", float("nan"))),
                        "w_intf_lin": float(logs.get("w_intf_lin", float("nan"))),
                        "w_pore_lin": float(logs.get("w_pore_lin", float("nan"))),
                        "w_mem_eff": float(logs.get("w_mem_eff", float("nan"))),
                        "w_intf_eff": float(logs.get("w_intf_eff", float("nan"))),
                        "w_pore_eff": float(logs.get("w_pore_eff", float("nan"))),
                    }

                    self.csv.log(row)

                    show = {
                        "step": int(step_for_row),
                        "loss": src.get("loss", 0.0),
                        "dist": src.get("dist", 0.0),
                        "fape": src.get("fape", 0.0),
                        "mem_raw": src.get("mem_raw", 0.0),
                        "pore": src.get("pore", 0.0),
                        "pore_raw": src.get("pore_raw", 0.0),
                        "pore_minA": src.get("pore_minA", 0.0),
                        "z_min": logs.get("z_tm_min", 0.0),
                        "z_max": logs.get("z_tm_max", 0.0),
                        "tm": logs.get("tm_frac", 0.0),
                    }
                    pbar.set_postfix(show)

                    # reset window-best for the *next* 100-step block
                    self.win_best_metric_value = float("inf")
                    self.win_best_step = -1
                    self.win_best_snapshot = {}

                if step and (step % eval_every == 0 or step == steps - 1):
                    # compute EMA (for "score") like your summary block does
                    cur = float(loss.item())
                    self.loss_ema = cur if self.loss_ema is None else (
                        (1.0 - self.ema_alpha) * self.loss_ema + self.ema_alpha * cur
                    )
                    ema_loss = float(self.loss_ema)

                # ---- end-of-training summary (written once) ----
                if (step % 100 == 0) or (step == steps - 1):
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
        try:
            torch.save(state, tmp)
            os.replace(tmp, fn)
        except Exception as e:
            print(f"[warn] failed to save checkpoint at step {step}: {e}")
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass 
    
    def _priors_weight(self):
        w = float(self.w["priors"])
        if self.priors_warmup > 0:
            t = min(1.0, self.global_step / max(1, self.priors_warmup))
            return w * 0.5 * (1.0 - math.cos(math.pi * t))
        return w