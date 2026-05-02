"""Streamlit web demo for crack detection.

Run locally:
    streamlit run app.py

Deploy on Hugging Face Spaces or Streamlit Cloud.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from src.model import CrackDetector, GradCAM
from src.transforms import get_val_transforms, denormalize

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crack Detection AI",
    page_icon="🔍",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINT = os.environ.get("CHECKPOINT_PATH", "checkpoints/best.pth")
IMAGE_SIZE = 224
DEVICE = torch.device("cpu")
CLASS_NAMES = {0: "✅ Non-cracked", 1: "⚠️ Cracked"}
CLASS_COLORS = {0: "green", 1: "red"}


# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource
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
    else:
        # Demo mode: pretrained ImageNet weights (no fine-tuning)
        model = CrackDetector(backbone="resnet50", pretrained=True, dropout=0.0)
        model.eval()
        return model, False


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(model, image_np):
    transform = get_val_transforms(IMAGE_SIZE)
    tensor = transform(image=image_np)["image"].to(DEVICE)

    with torch.no_grad():
        out = model(tensor.unsqueeze(0))
        probs = F.softmax(out["crack_logits"], dim=1)[0]
        pred = probs.argmax().item()
        confidence = probs[pred].item()

    # Grad-CAM
    cam = GradCAM(model)
    heatmap = cam(tensor, class_idx=1)
    cam.remove_hooks()

    return pred, confidence, probs[1].item(), tensor, heatmap


def make_gradcam_overlay(tensor, heatmap, alpha=0.5):
    img = denormalize(tensor).permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hm = (heatmap.numpy() * 255).astype(np.uint8)
    hm_colored = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, hm_colored, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔍 Crack Detection on Concrete Structures")
st.markdown(
    "Upload an image of a **concrete surface** (bridge deck, pavement or wall). "
    "The model detects cracks and highlights them with **Grad-CAM**."
)

model, is_finetuned = load_model()

if not is_finetuned:
    st.warning(
        "⚠️ Running in **demo mode** — no fine-tuned checkpoint found. "
        "Train the model first: `python scripts/train.py`"
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Crack probability threshold", 0.1, 0.9, 0.5, 0.05)
    gradcam_alpha = st.slider("Grad-CAM opacity", 0.1, 0.9, 0.5, 0.05)
    st.markdown("---")
    st.markdown("**Dataset**: [SDNET2018](https://data.mendeley.com/datasets/z6n8jg7bky/2)")
    st.markdown("**Model**: ResNet-50 fine-tuned")
    st.markdown("**GitHub**: [faycalbelaid-GC/crack-detection](https://github.com/faycalbelaid-GC/crack-detection)")

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Choose an image (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded:
    image_pil = Image.open(uploaded).convert("RGB")
    image_np = np.array(image_pil)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📷 Original")
        st.image(image_pil, use_container_width=True)

    with st.spinner("Analyzing..."):
        pred, confidence, crack_prob, tensor, heatmap = predict(model, image_np)

    final_pred = 1 if crack_prob >= threshold else 0

    with col2:
        st.subheader("🌡️ Grad-CAM")
        overlay = make_gradcam_overlay(tensor, heatmap, gradcam_alpha)
        st.image(overlay, use_container_width=True)
        st.caption("Red zones = areas activating the crack detector")

    with col3:
        st.subheader("📊 Result")
        color = CLASS_COLORS[final_pred]
        label = CLASS_NAMES[final_pred]

        st.markdown(
            f"<h2 style='color:{color};text-align:center'>{label}</h2>",
            unsafe_allow_html=True,
        )
        st.metric("Crack probability", f"{crack_prob:.1%}")
        st.metric("Confidence", f"{confidence:.1%}")
        st.progress(float(crack_prob))

        st.markdown("---")
        st.markdown("**Probabilities**")
        st.write(f"Non-cracked : `{1 - crack_prob:.1%}`")
        st.write(f"Cracked     : `{crack_prob:.1%}`")

else:
    st.info("👆 Upload a concrete surface image to get started.")
    st.markdown("**Example surfaces**: bridge decks, pavements, walls")
