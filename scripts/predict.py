"""Run inference on a single image or directory.

Examples:
    # Single image (with Grad-CAM)
    python scripts/predict.py --checkpoint checkpoints/best.pth --input photo.jpg

    # Directory of images
    python scripts/predict.py --checkpoint checkpoints/best.pth --input ./data/test_images/

    # Sliding-window on a large image
    python scripts/predict.py --checkpoint checkpoints/best.pth --input big_wall.jpg --sliding_window
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from src.model import CrackDetector, GradCAM
from src.transforms import get_val_transforms
from src.visualize import save_gradcam, sliding_window_predict


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
CLASS_NAMES = {0: "Non-cracked", 1: "Cracked"}


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    state = torch.load(checkpoint_path, map_location=device)
    cfg = state["cfg"]
    model = CrackDetector(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["model"]["num_classes"],
        dropout=0.0,
        pretrained=False,
    )
    model.load_state_dict(state["model_state"])
    model = model.to(device).eval()
    return model, cfg


def predict_single(
    model,
    image_path: str,
    transform,
    device: torch.device,
    use_gradcam: bool = True,
    save_dir: Path = None,
) -> dict:
    import numpy as np
    from PIL import Image

    img_np = np.array(Image.open(image_path).convert("RGB"))
    tensor = transform(image=img_np)["image"].to(device)

    with torch.no_grad():
        out = model(tensor.unsqueeze(0))
        probs = F.softmax(out["crack_logits"], dim=1)[0]
        pred = probs.argmax().item()
        confidence = probs[pred].item()

    result = {
        "path": image_path,
        "prediction": CLASS_NAMES[pred],
        "confidence": confidence,
        "crack_prob": probs[1].item(),
    }

    print(
        f"  {Path(image_path).name:<40}  "
        f"{CLASS_NAMES[pred]:<14}  "
        f"crack_prob={probs[1].item():.3f}"
    )

    if use_gradcam and save_dir:
        cam = GradCAM(model)
        heatmap = cam(tensor, class_idx=1)
        cam.remove_hooks()

        label = pred  # no GT available
        out_path = save_dir / f"gradcam_{Path(image_path).stem}.jpg"
        save_gradcam(tensor.cpu(), heatmap, label, pred, probs[1].item(), str(out_path))

    return result


def main():
    parser = argparse.ArgumentParser(description="Crack detection inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--output_dir", default="./outputs/predictions")
    parser.add_argument("--no_gradcam", action="store_true")
    parser.add_argument("--sliding_window", action="store_true", help="Sliding window on large images")
    parser.add_argument("--sw_stride", type=int, default=128, help="Sliding window stride")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg = load_model(args.checkpoint, device)
    transform = get_val_transforms(cfg["data"]["image_size"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)

    if input_path.is_dir():
        images = [p for p in input_path.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS]
        print(f"Found {len(images)} images in {input_path}\n")
    else:
        images = [input_path]

    if args.sliding_window:
        for img_path in images:
            print(f"Sliding window: {img_path}")
            save_path = str(out_dir / f"sw_{img_path.stem}.jpg")
            sliding_window_predict(
                model, str(img_path),
                window_size=cfg["data"]["image_size"],
                stride=args.sw_stride,
                threshold=args.threshold,
                device=device,
                save_path=save_path,
            )
            print(f"  Saved → {save_path}")
    else:
        print(f"{'Image':<40}  {'Prediction':<14}  Confidence")
        print("-" * 70)
        results = []
        for img_path in images:
            r = predict_single(
                model, str(img_path), transform, device,
                use_gradcam=not args.no_gradcam,
                save_dir=None if args.no_gradcam else out_dir,
            )
            results.append(r)

        cracked = sum(1 for r in results if r["crack_prob"] >= args.threshold)
        print(f"\nSummary: {cracked}/{len(results)} images classified as cracked")


if __name__ == "__main__":
    main()
