"""Visualization utilities: Grad-CAM overlays, prediction grids, sliding-window inference."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .transforms import denormalize, get_val_transforms

CLASS_NAMES = {0: "Non-cracked", 1: "Cracked"}
COLORMAP = cv2.COLORMAP_JET


def overlay_gradcam(
    image_tensor: torch.Tensor,
    heatmap: torch.Tensor,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap onto the original image.

    Args:
        image_tensor: (3, H, W) normalized tensor
        heatmap:      (H, W) float tensor in [0, 1]
        alpha:        heatmap opacity

    Returns:
        (H, W, 3) uint8 BGR image for OpenCV display / saving
    """
    img = denormalize(image_tensor).permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    hm = (heatmap.numpy() * 255).astype(np.uint8)
    hm_colored = cv2.applyColorMap(hm, COLORMAP)

    overlay = cv2.addWeighted(img_bgr, 1 - alpha, hm_colored, alpha, 0)
    return overlay


def save_gradcam(
    image_tensor: torch.Tensor,
    heatmap: torch.Tensor,
    label: int,
    pred: int,
    prob: float,
    save_path: str,
    alpha: float = 0.5,
) -> None:
    overlay = overlay_gradcam(image_tensor, heatmap, alpha)

    color = (0, 255, 0) if label == pred else (0, 0, 255)
    text = f"GT: {CLASS_NAMES[label]} | Pred: {CLASS_NAMES[pred]} ({prob:.2f})"
    cv2.putText(overlay, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), overlay)


def plot_batch_predictions(
    images: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    ncols: int = 4,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a grid of predictions with GT/pred labels."""
    import torch.nn.functional as F

    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    n = min(len(images), ncols * 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = np.array(axes).flatten()

    for i in range(n):
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        ax = axes[i]
        ax.imshow(img)
        ax.axis("off")
        correct = preds[i] == labels_np[i]
        color = "green" if correct else "red"
        ax.set_title(
            f"GT:{CLASS_NAMES[labels_np[i]]}\nP:{CLASS_NAMES[preds[i]]} ({probs[i]:.2f})",
            fontsize=7,
            color=color,
        )

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def sliding_window_predict(
    model: torch.nn.Module,
    image_path: str,
    window_size: int = 256,
    stride: int = 128,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run sliding-window inference on a large image.

    Returns:
        annotated_image: BGR image with crack regions highlighted
        prob_map:        (H, W) probability map
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_val_transforms(image_size=window_size)
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(image_path)

    h, w = orig.shape[:2]
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                patch = orig[y : y + window_size, x : x + window_size]
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                tensor = transform(image=patch_rgb)["image"].unsqueeze(0).to(device)
                out = model(tensor)
                prob = torch.softmax(out["crack_logits"], dim=1)[0, 1].item()
                prob_map[y : y + window_size, x : x + window_size] += prob
                count_map[y : y + window_size, x : x + window_size] += 1.0

    mask = count_map > 0
    prob_map[mask] /= count_map[mask]

    # Overlay crack regions
    annotated = orig.copy()
    crack_mask = (prob_map >= threshold).astype(np.uint8)
    heatmap_u8 = (prob_map * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_u8, COLORMAP)
    annotated = cv2.addWeighted(annotated, 0.6, heatmap_colored, 0.4, 0)

    # Contours around crack regions
    contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours, -1, (0, 0, 255), 2)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, annotated)
        np.save(save_path.replace(".jpg", "_probmap.npy"), prob_map)

    return annotated, prob_map
