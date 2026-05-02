"""Main training script.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/config.yaml --backbone resnet50
    python scripts/train.py --backbone resnet18 --epochs 30 --batch_size 64
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml

from src.dataset import create_dataloaders
from src.model import CrackDetector
from src.trainer import Trainer
from src.transforms import get_train_transforms, get_val_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Train crack detector")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--backbone", default=None, help="Override backbone (resnet18/34/50/101)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data_dir", default=None, help="Override dataset path")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--freeze_epochs", type=int, default=0, help="Epochs to train with frozen backbone")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args) -> dict:
    if args.backbone:
        cfg["model"]["backbone"] = args.backbone
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["data"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["lr"] = args.lr
    if args.data_dir:
        cfg["data"]["root"] = args.data_dir
    if args.no_amp:
        cfg["training"]["use_amp"] = False
    return cfg


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    dcfg = cfg["data"]
    image_size = dcfg["image_size"]

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=dcfg["root"],
        train_transform=get_train_transforms(image_size),
        val_transform=get_val_transforms(image_size),
        batch_size=dcfg["batch_size"],
        num_workers=dcfg["num_workers"],
        val_size=dcfg["val_size"],
        test_size=dcfg["test_size"],
        seed=dcfg["seed"],
    )

    class_weights = train_loader.dataset.get_class_weights()
    print(f"Class weights: non-cracked={class_weights[0]:.3f}, cracked={class_weights[1]:.3f}")

    model = CrackDetector(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        pretrained=cfg["model"]["pretrained"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {cfg['model']['backbone']}")
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Optional: freeze backbone for the first N epochs (transfer learning warm-up)
    if args.freeze_epochs > 0:
        print(f"\nPhase 1: Training classifier head only for {args.freeze_epochs} epochs")
        model.freeze_backbone(True)
        warm_cfg = dict(cfg)
        warm_cfg["training"] = dict(cfg["training"])
        warm_cfg["training"]["epochs"] = args.freeze_epochs
        warm_cfg["training"]["patience"] = args.freeze_epochs + 1

        trainer = Trainer(model, train_loader, val_loader, warm_cfg, class_weights=class_weights)
        trainer.fit()

        print("\nPhase 2: Fine-tuning full network")
        model.freeze_backbone(False)

    trainer = Trainer(model, train_loader, val_loader, cfg, class_weights=class_weights)
    history = trainer.fit()

    # Final evaluation on test set
    print("\n--- Test Set Evaluation ---")
    from src.metrics import ConfusionMatrixTracker

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    tracker = ConfusionMatrixTracker(["Non-cracked", "Cracked"])

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["crack_label"].to(device)
            outputs = model(images)
            tracker.update(outputs["crack_logits"], labels)

    tracker.print_report()
    metrics = tracker.get_metrics(prefix="test/")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
