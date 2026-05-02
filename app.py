"""Gradio web demo for crack detection (Hugging Face Spaces compatible)."""

import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from src.model import CrackDetector, GradCAM
from src.transforms import get_val_transforms, denormalize

CHECKPOINT = "checkpoints/best.pth"
IMAGE_SIZE = 224
DEVICE = torch.device("cpu")


@torch.no_grad()
def load_model():
    if Path(CHECKPOINT).exists():
        state = torch.load(CHECKPOINT, map_location=DEVICE)
        cfg = state["cfg"]
        model = CrackDetector(
            backbone=cfg["model"]["backbone"],
            num_classes=cfg["model"]["num_classes"],
            dropout=0.0,
            pretrained=False,
        )
        model.load_state_dict(state["model_state"])
        model.eval()
        return model, True
    model = CrackDetector(backbone="resnet50", pretrained=True, dropout=0.0)
    model.eval()
    return model, False


MODEL, IS_FINETUNED = load_model()
TRANSFORM = get_val_transforms(IMAGE_SIZE)


def predict(image: np.ndarray, threshold: float, alpha: float):
    if image is None:
        return None, "Please upload an image.", {}

    tensor = TRANSFORM(image=image)["image"].to(DEVICE)

    with torch.no_grad():
        out = MODEL(tensor.unsqueeze(0))
        probs = F.softmax(out["crack_logits"], dim=1)[0]

    crack_prob = probs[1].item()
    final_pred = 1 if crack_prob >= threshold else 0

    # Grad-CAM
    cam = GradCAM(MODEL)
    heatmap = cam(tensor, class_idx=1)
    cam.remove_hooks()

    # Build overlay
    img_vis = denormalize(tensor).permute(1, 2, 0).numpy()
    img_vis = (img_vis * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    hm = (heatmap.numpy() * 255).astype(np.uint8)
    hm_colored = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    overlay_bgr = cv2.addWeighted(img_bgr, 1 - alpha, hm_colored, alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    label = "⚠️ CRACKED" if final_pred == 1 else "✅ NON-CRACKED"
    details = (
        f"**Result : {label}**\n\n"
        f"- Crack probability : **{crack_prob:.1%}**\n"
        f"- Non-crack probability : **{1 - crack_prob:.1%}**\n"
        f"- Threshold used : {threshold:.2f}\n"
        f"- Model : {'Fine-tuned ResNet-50' if IS_FINETUNED else 'ResNet-50 (demo mode – ImageNet weights)'}"
    )

    confidences = {
        "Non-cracked": float(probs[0]),
        "Cracked": float(probs[1]),
    }

    return overlay_rgb, details, confidences


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Crack Detection AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔍 Crack Detection on Concrete Structures
        Upload an image of a **concrete surface** (bridge deck, pavement or wall).
        The model classifies it as cracked or non-cracked and highlights suspicious zones with **Grad-CAM**.
        """
    )

    if not IS_FINETUNED:
        gr.Warning("Running in demo mode (ImageNet weights only). Train the model for real results.")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input image", type="numpy")
            threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Crack threshold")
            alpha = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Grad-CAM opacity")
            btn = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=1):
            gradcam_out = gr.Image(label="Grad-CAM heatmap")
            text_out = gr.Markdown(label="Result")
            label_out = gr.Label(label="Probabilities", num_top_classes=2)

    btn.click(
        fn=predict,
        inputs=[input_image, threshold, alpha],
        outputs=[gradcam_out, text_out, label_out],
    )

    gr.Examples(
        examples=[],
        inputs=input_image,
    )

    gr.Markdown(
        """
        ---
        **Model** : ResNet-50 fine-tuned on SDNET2018 (~56,000 images)
        **Dataset** : [SDNET2018 – Mendeley Data](https://data.mendeley.com/datasets/z6n8jg7bky/2)
        **Code** : [GitHub – faycalbelaid-GC/crack-detection](https://github.com/faycalbelaid-GC/crack-detection)
        """
    )

if __name__ == "__main__":
    demo.launch()
