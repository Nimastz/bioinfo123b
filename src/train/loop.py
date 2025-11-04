# src/train/loop.py
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
    def __init__(self, cfg, model, opt, sched, device):
        self.cfg, self.model, self.opt, self.sched, self.device = cfg, model, opt, sched, device
        self.w = cfg["loss_weights"]; self.pr = cfg["priors"]
        self.use_cuda = (device.type == "cuda")
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
        
    def step_losses(self, batch, out):
        L = out["xyz"].shape[0]
        loss = out["xyz"].new_tensor(0.0)
        logs = {}

        loss_dist = distogram_loss(out["dist"], out["xyz"])
        loss_tors = torsion_l2(out["tors"])
        loss_fape = fape_loss(out["xyz"])
        logs.update(distogram=float(loss_dist), torsion=float(loss_tors), fape=float(loss_fape))
        loss = loss + self.w["distogram"]*loss_dist + self.w["torsion"]*loss_tors + self.w["fape"]*loss_fape

        if self.pr.get("use_cn", True):
            n = self.pr["n_copies"]; rr = self.pr["ring_radius"]
            olig = assemble_cn(out["xyz"], n_copies=n, ring_radius=rr)  # (n,L,3)
            tm_mask = membrane_z_mask(L, self.pr["tm_span"]).to(out["xyz"].device)

            mem = membrane_slab_loss(out["xyz"], tm_mask)
            intf = interface_contact_loss(olig, cutoff=9.0)
            clash = ca_clash_loss(olig, min_dist=3.6)
            pore = pore_target_loss(olig, target_A=self.pr["pore_target_A"])

            logs.update(mem=float(mem), intf=float(intf), clash=float(clash), pore=float(pore))
            prw = self._priors_weight()
            loss = loss + prw*(0.3*mem + 0.4*intf + 0.3*pore) + 0.1*clash
            logs["priors_w"] = float(prw)
            
        return loss, logs

    def fit(self, train_loader, val_loader):
        steps = self.cfg["train"]["steps"]
        log_every = self.cfg["train"]["log_every"]
        eval_every = self.cfg["train"]["eval_every"]
        ckpt_dir = self.cfg["train"]["ckpt_dir"]

        pbar = trange(steps, desc="train")
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
                # AMP forward + loss
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_cuda):
                    out = self.model(batch["seq_idx"], batch.get("emb"))
                    loss, logs = self.step_losses(batch, out)

                # Skip update if non-finite
                if not torch.isfinite(loss):
                    pbar.set_postfix({"loss": "nan", **{k: round(v,3) for k,v in logs.items()}})
                    self.opt.zero_grad(set_to_none=True)
                    if self.use_cuda: self.scaler.update()
                    continue

                self.opt.zero_grad(set_to_none=True)

                if self.use_cuda:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["train"]["grad_clip"])
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    if self.sched is not None:
                        self.sched.step()  # optimizer.step() -> scheduler.step()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["train"]["grad_clip"])
                    self.opt.step()
                    if self.sched is not None:
                        self.sched.step()

                if step % log_every == 0:
                    pbar.set_postfix({"loss": float(loss.item()), **{k: round(v,3) for k,v in logs.items()}})

                if step and step % eval_every == 0:
                    self.save(step, ckpt_dir)

            # normal end
            self.save(steps, ckpt_dir)

        except KeyboardInterrupt:
            # ensure we always save something useful when you hit Ctrl+C
            safe_step = step if "step" in locals() else 0
            print(f"\n[info] interrupted @ step {safe_step} — saving checkpoint")
            self.save(safe_step, ckpt_dir)
            return


    def save(self, step, ckpt_dir):
        """
        Interrupt-safe save. Writes to a temp file then atomically moves it into place.
        Includes model + optimizer/scheduler/scaler to allow resuming later,
        while remaining compatible with eval scripts that only read ['model'].
        """
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
        os.replace(tmp, fn)  # atomic on Windows & POSIX
    
    def _priors_weight(self):
        w = float(self.w["priors"])
        if self.priors_warmup > 0:
            t = min(1.0, self.global_step / max(1, self.priors_warmup))
            return w * t
        return w