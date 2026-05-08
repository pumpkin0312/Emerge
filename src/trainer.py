"""Training loop for ProteinLM."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .model import ProteinLM

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
class Trainer:
    def __init__(
        self,
        model:       ProteinLM,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          dict,
        device:       torch.device,
    ):
        self.model    = model
        self.train_dl = train_loader
        self.val_dl   = val_loader
        self.cfg      = cfg
        self.device   = device

        tcfg = cfg["training"]

        self.max_epochs = tcfg["max_epochs"]
        self.grad_accum = tcfg["gradient_accumulation"]
        self.max_grad_norm = tcfg["max_grad_norm"]
        self.log_every  = tcfg["log_every_n_steps"]
        self.save_every = tcfg["save_every_n_epochs"]
        self.keep_top_k = tcfg["keep_top_k"]
        self.save_dir   = Path(tcfg["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.patience   = tcfg.get("patience", 5) if tcfg.get("early_stopping") else None

        # precision
        prec = tcfg["precision"]
        self.autocast_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[prec]
        self.use_amp = self.autocast_dtype is not None
        self.scaler  = GradScaler(enabled=(prec == "fp16"))

        self.optimizer = self._build_optimizer(tcfg)
        self.scheduler = self._build_scheduler(tcfg)

        # wandb
        self.use_wandb = tcfg.get("use_wandb", False)
        if self.use_wandb:
            import wandb
            wandb.init(project=tcfg.get("wandb_project", "emerge"))
            wandb.watch(self.model, log_freq=self.log_every)

        self._best_val_loss = float("inf")
        self._no_improve    = 0
        self._saved_ckpts:  list[tuple[float, Path]] = []

    # ------------------------------------------------------------------
    def _build_optimizer(self, tcfg: dict) -> AdamW:
        return AdamW(
            self.model.parameters(),
            lr           = tcfg["lr"],
            betas        = (tcfg["beta1"], tcfg["beta2"]),
            eps          = tcfg["eps"],
            weight_decay = tcfg["weight_decay"],
        )

    def _build_scheduler(self, tcfg: dict) -> SequentialLR:
        total_steps   = self.max_epochs * len(self.train_dl) // self.grad_accum
        warmup_steps  = tcfg["warmup_steps"]
        warmup_sched  = LinearLR(self.optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
        cosine_sched  = CosineAnnealingLR(self.optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6)
        return SequentialLR(self.optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])

    # ------------------------------------------------------------------
    def train(self) -> None:
        logger.info(f"Starting training for {self.max_epochs} epochs")
        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss, val_acc = self._eval_epoch()
            logger.info(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )
            if self.use_wandb:
                import wandb
                wandb.log({"epoch": epoch, "train_loss": train_loss,
                           "val_loss": val_loss, "val_acc": val_acc})

            # save checkpoint
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, val_loss)

            # early stopping
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._no_improve = 0
                self._save_checkpoint(epoch, val_loss, tag="best")
            else:
                self._no_improve += 1
                if self.patience and self._no_improve >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        logger.info("Training complete.")

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        step = 0
        t0   = time.time()

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_dl):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            ctx = (
                torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
                if self.use_amp else _nullcontext()
            )
            with ctx:
                out  = self.model(input_ids, attention_mask)
                loss = _mlm_loss(out["logits"], labels)
                loss = loss / self.grad_accum

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                step += 1

                total_loss += loss.item() * self.grad_accum

                if step % self.log_every == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        f"  Ep{epoch} step {step} | loss={total_loss/step:.4f} "
                        f"lr={self.optimizer.param_groups[0]['lr']:.2e} "
                        f"({elapsed:.1f}s)"
                    )

        return total_loss / max(step, 1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_epoch(self) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for batch in self.val_dl:
            # apply masking on-the-fly for eval
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            out  = self.model(input_ids, attention_mask)
            loss = _mlm_loss(out["logits"], labels)
            total_loss += loss.item()

            mask    = labels != -100
            preds   = out["logits"].argmax(dim=-1)
            correct += (preds[mask] == labels[mask]).sum().item()
            total   += mask.sum().item()

        n = len(self.val_dl)
        return total_loss / max(n, 1), correct / max(total, 1)

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, val_loss: float, tag: str = "") -> None:
        name   = f"ckpt_ep{epoch:03d}_{tag}.pt" if tag else f"ckpt_ep{epoch:03d}.pt"
        path   = self.save_dir / name
        torch.save(
            {
                "epoch":      epoch,
                "val_loss":   val_loss,
                "model":      self.model.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "scheduler":  self.scheduler.state_dict(),
                "config":     self.cfg,
            },
            path,
        )
        logger.info(f"  Saved checkpoint → {path}")

        if not tag:
            import heapq
            heapq.heappush(self._saved_ckpts, (val_loss, path))
            while len(self._saved_ckpts) > self.keep_top_k:
                _, old_path = heapq.heappop(self._saved_ckpts)
                if old_path.exists():
                    old_path.unlink()


# -----------------------------------------------------------------------
def _mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )


class _nullcontext:
    def __enter__(self): return self
    def __exit__(self, *_): pass
