"""Evaluation metrics for crack detection."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        preds:  (N,) integer predictions
        labels: (N,) integer ground-truth labels
        probs:  (N,) probability for the positive class (cracked)
        prefix: optional key prefix, e.g. "val/"
    """
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    p = prefix
    return {
        f"{p}accuracy": float(acc),
        f"{p}precision": float(prec),
        f"{p}recall": float(rec),
        f"{p}f1": float(f1),
        f"{p}auc_roc": float(auc),
    }


class ConfusionMatrixTracker:
    """Accumulates predictions across batches, computes full report at end."""

    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self._preds: List[int] = []
        self._labels: List[int] = []
        self._probs: List[float] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        self._preds.extend(preds.tolist())
        self._labels.extend(labels.detach().cpu().numpy().tolist())
        self._probs.extend(probs.tolist())

    def reset(self) -> None:
        self._preds.clear()
        self._labels.clear()
        self._probs.clear()

    @property
    def preds(self) -> np.ndarray:
        return np.array(self._preds)

    @property
    def labels(self) -> np.ndarray:
        return np.array(self._labels)

    @property
    def probs(self) -> np.ndarray:
        return np.array(self._probs)

    def get_metrics(self, prefix: str = "") -> Dict[str, float]:
        return compute_metrics(self.preds, self.labels, self.probs, prefix=prefix)

    def get_confusion_matrix(self) -> np.ndarray:
        return confusion_matrix(self.labels, self.preds)

    def print_report(self) -> None:
        print(classification_report(self.labels, self.preds, target_names=self.class_names))
        cm = self.get_confusion_matrix()
        print("Confusion Matrix:")
        print(f"{'':>14}  " + "  ".join(f"{n:>12}" for n in self.class_names))
        for i, row in enumerate(cm):
            print(f"  {self.class_names[i]:>12}  " + "  ".join(f"{v:>12}" for v in row))


class AverageMeter:
    """Running average for scalar losses/metrics."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0
