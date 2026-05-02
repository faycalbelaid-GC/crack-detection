---
title: Crack Detection
emoji: 🔍
colorFrom: gray
colorTo: red
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# Crack Detection by Computer Vision

Binary crack classifier for concrete structures (bridge decks, pavements, walls) using a fine-tuned ResNet-50 with Grad-CAM explainability. Trained on the **SDNET2018** dataset (~56,000 images, 256×256).

## Results (ResNet-50, test set)

| Metric | Value |
|--------|-------|
| Accuracy | ~97% |
| F1 Score | ~0.97 |
| AUC-ROC | ~0.99 |
| Precision | ~0.96 |
| Recall | ~0.98 |

## Architecture

```
Input (224×224×3)
    └── ResNet-50 backbone (ImageNet pretrained)
            └── AdaptiveAvgPool2d
                    └── Dropout(0.5) → FC(2048→256) → ReLU → Dropout(0.25)
                            ├── Crack head   → 2 classes (cracked / non-cracked)
                            └── Surface head → 3 classes (deck / pavement / wall)
```

## Features

- Fine-tuned ResNet-18/34/50/101 (configurable)
- Multi-task learning (crack + surface type)
- Grad-CAM heatmaps for explainability
- Sliding-window inference on large images
- Mixed-precision training (AMP), OneCycleLR, early stopping
- Albumentations augmentations
- TensorBoard logging
- ONNX export for deployment

## Quick Start

```bash
pip install -r requirements.txt
python scripts/download_data.py --output ./data
python scripts/train.py
python scripts/evaluate.py --checkpoint checkpoints/best.pth --save_plots
streamlit run app.py
```

## Dataset

**SDNET2018** — Structural Defects Network
[https://data.mendeley.com/datasets/z6n8jg7bky/2](https://data.mendeley.com/datasets/z6n8jg7bky/2)

## Project Structure

```
crack_detection/
├── src/
│   ├── dataset.py       SDNET2018 loader
│   ├── model.py         CrackDetector (ResNet + Grad-CAM)
│   ├── transforms.py    Augmentation pipelines
│   ├── trainer.py       Training loop
│   ├── metrics.py       Metrics & confusion matrix
│   └── visualize.py     Grad-CAM, sliding-window
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── export_onnx.py
├── app.py               Streamlit web demo
└── configs/config.yaml
```
