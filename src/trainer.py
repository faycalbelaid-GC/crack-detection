"""Training loop with mixed-precision, early stopping, and TensorBoard logging."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import AverageMeter, ConfusionMatrixTracker
from .model import CrackDetector, CrackLoss


CLASS_NAMES = ["Non-cracked", "Cracked"]


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        improved = (self.mode == "max" and metric > self.best + self.min_delta) or (
            self.mode == "min" and metric < self.best - self.min_delta
        )
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    def __init__(
        self,
        model: CrackDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: Optional[torch.device] = None,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg

        tcfg = cfg["training"]
        pcfg = cfg["paths"]

        self.epochs = tcfg["epochs"]
        self.use_amp = tcfg["use_amp"] and self.device.type == "cuda"
        self.grad_clip = tcfg["gradient_clip"]

        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = CrackLoss(class_weights=class_weights, multitask=model.multitask)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=tcfg["lr"],
            weight_decay=tcfg["weight_decay"],
        )

        steps_per_epoch = len(train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=tcfg["lr"],
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=tcfg["warmup_epochs"] / self.epochs,
        )

        self.scaler = GradScaler(enabled=self.use_amp)
        self.early_stop = EarlyStopping(patience=tcfg["patience"], min_delta=tcfg["min_delta"], mode="max")

        self.ckpt_dir = Path(pcfg["checkpoints"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=pcfg["logs"])

        self.best_f1 = 0.0
        self.global_step = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> Dict:
        print(f"\nTraining on {self.device} | AMP: {self.use_amp}")
        print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>9}  {'Val F1':>7}  {'Val AUC':>8}  LR")
        print("-" * 65)

        history: Dict = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []}

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            val_metrics, val_tracker = self._val_epoch(epoch)

            lr = self.optimizer.param_groups[0]["lr"]
            tl = train_metrics["loss"]
            vl = val_metrics["loss"]
            vf1 = val_metrics.get("val/f1", 0.0)
            vauc = val_metrics.get("val/auc_roc", 0.0)

            history["train_loss"].append(tl)
            history["val_loss"].append(vl)
            history["val_f1"].append(vf1)
            history["val_auc"].append(vauc)

            elapsed = time.time() - t0
            print(f"{epoch:>6}  {tl:>10.4f}  {vl:>9.4f}  {vf1:>7.4f}  {vauc:>8.4f}  {lr:.2e}  [{elapsed:.0f}s]")

            self._tensorboard_log(epoch, train_metrics, val_metrics)

            if vf1 > self.best_f1:
                self.best_f1 = vf1
                self._save_checkpoint(epoch, vf1, is_best=True)

            if epoch % 10 == 0:
                self._save_checkpoint(epoch, vf1, is_best=False)

            if self.early_stop(vf1):
                print(f"\nEarly stopping at epoch {epoch} (best F1: {self.best_f1:.4f})")
                break

        self.writer.close()
        return history

    # ------------------------------------------------------------------
    # Internal epoch loops
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        loss_meter = AverageMeter("loss")

        pbar = tqdm(self.train_loader, desc=f"Train [{epoch}]", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            crack_labels = batch["crack_label"].to(self.device, non_blocking=True)
            surface_labels = batch["surface_label"].to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                losses = self.criterion(outputs, crack_labels, surface_labels)
                loss = losses["total"]

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            loss_meter.update(loss.item(), images.size(0))
            self.global_step += 1
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        return {"loss": loss_meter.avg}

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> tuple:
        self.model.eval()
        loss_meter = AverageMeter("loss")
        tracker = ConfusionMatrixTracker(CLASS_NAMES)

        for batch in tqdm(self.val_loader, desc=f"Val   [{epoch}]", leave=False):
            images = batch["image"].to(self.device, non_blocking=True)
            crack_labels = batch["crack_label"].to(self.device, non_blocking=True)
            surface_labels = batch["surface_label"].to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                losses = self.criterion(outputs, crack_labels, surface_labels)

            loss_meter.update(losses["total"].item(), images.size(0))
            tracker.update(outputs["crack_logits"], crack_labels)

        metrics = tracker.get_metrics(prefix="val/")
        metrics["loss"] = loss_meter.avg
        return metrics, tracker

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _tensorboard_log(self, epoch: int, train: Dict, val: Dict) -> None:
        self.writer.add_scalar("loss/train", train["loss"], epoch)
        self.writer.add_scalar("loss/val", val["loss"], epoch)
        for k, v in val.items():
            if k != "loss":
                self.writer.add_scalar(k, v, epoch)
        self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

    def _save_checkpoint(self, epoch: int, score: float, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_f1": self.best_f1,
            "cfg": self.cfg,
        }
        name = "best.pth" if is_best else f"epoch_{epoch:03d}.pth"
        torch.save(state, self.ckpt_dir / name)
