"""Evaluate a trained checkpoint on the test set and export results.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pth
    python scripts/evaluate.py --checkpoint checkpoints/best.pth --save_plots
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.metrics import roc_curve

from src.dataset import create_dataloaders, SDNET2018Dataset
from src.metrics import ConfusionMatrixTracker
from src.model import CrackDetector, GradCAM
from src.transforms import get_val_transforms
from src.visualize import save_gradcam, plot_batch_predictions


CLASS_NAMES = ["Non-cracked", "Cracked"]


def load_checkpoint(path: str):
    state = torch.load(path, map_location="cpu")
    cfg = state["cfg"]
    model = CrackDetector(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        pretrained=False,
    )
    model.load_state_dict(state["model_state"])
    return model, cfg, state["epoch"]


def plot_confusion_matrix(cm: np.ndarray, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
    )
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix: {save_path}")


def plot_roc_curve(labels: np.ndarray, probs: np.ndarray, save_path: str) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve – Crack Detection")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved ROC curve: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate crack detector")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="./outputs/eval")
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--n_gradcam", type=int, default=8, help="Number of Grad-CAM samples to save")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg, epoch = load_checkpoint(args.checkpoint)
    model = model.to(device).eval()
    print(f"Checkpoint from epoch {epoch}")

    dcfg = cfg["data"]
    _, _, test_loader = create_dataloaders(
        data_dir=dcfg["root"],
        train_transform=get_val_transforms(dcfg["image_size"]),
        val_transform=get_val_transforms(dcfg["image_size"]),
        batch_size=dcfg["batch_size"],
        num_workers=dcfg["num_workers"],
        val_size=dcfg["val_size"],
        test_size=dcfg["test_size"],
        seed=dcfg["seed"],
    )

    tracker = ConfusionMatrixTracker(CLASS_NAMES)
    first_batch = None

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            labels = batch["crack_label"].to(device)
            outputs = model(images)
            tracker.update(outputs["crack_logits"], labels)
            if first_batch is None:
                first_batch = (images.cpu(), labels.cpu(), outputs["crack_logits"].cpu())

    print("\n=== Test Results ===")
    tracker.print_report()
    metrics = tracker.get_metrics(prefix="test/")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if args.save_plots:
        plot_confusion_matrix(tracker.get_confusion_matrix(), str(out_dir / "confusion_matrix.png"))
        plot_roc_curve(tracker.labels, tracker.probs, str(out_dir / "roc_curve.png"))

        imgs, lbls, lgts = first_batch
        fig = plot_batch_predictions(imgs, lbls, lgts, ncols=4, save_path=str(out_dir / "predictions_grid.png"))
        plt.close(fig)
        print(f"Saved prediction grid: {out_dir / 'predictions_grid.png'}")

    # Grad-CAM samples
    if args.n_gradcam > 0:
        cam = GradCAM(model)
        gradcam_dir = out_dir / "gradcam"
        gradcam_dir.mkdir(exist_ok=True)
        saved = 0
        for batch in test_loader:
            if saved >= args.n_gradcam:
                break
            for j in range(min(len(batch["image"]), args.n_gradcam - saved)):
                img = batch["image"][j].to(device)
                label = batch["crack_label"][j].item()
                pred_class = 1

                heatmap = cam(img, class_idx=pred_class)
                with torch.no_grad():
                    out = model(img.unsqueeze(0))
                    prob = torch.softmax(out["crack_logits"], dim=1)[0, 1].item()
                    pred = out["crack_logits"].argmax(dim=1).item()

                save_gradcam(
                    img.cpu(), heatmap, label, pred, prob,
                    save_path=str(gradcam_dir / f"gradcam_{saved:03d}.jpg"),
                )
                saved += 1
        cam.remove_hooks()
        print(f"Saved {saved} Grad-CAM images to {gradcam_dir}")


if __name__ == "__main__":
    main()
