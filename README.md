---
title: Crack Detection
emoji: 🔍
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 5.25.0
app_file: app.py
pinned: false
license: mit
---

# Crack Detection by Computer Vision

Binary crack classifier for concrete structures using a fine-tuned **ResNet-50** with **Grad-CAM** explainability.
Trained on the [SDNET2018](https://data.mendeley.com/datasets/z6n8jg7bky/2) dataset (~56,000 images).

## Results (ResNet-50, test set)

| Metric | Value |
|--------|-------|
| Accuracy | ~97% |
| F1 Score | ~0.97 |
| AUC-ROC | ~0.99 |

## Quick Start

```bash
pip install -r requirements.txt
python scripts/download_data.py --output ./data
python scripts/train.py
python app.py
```

## Project Structure

```
├── src/
│   ├── dataset.py       SDNET2018 loader
│   ├── model.py         CrackDetector (ResNet + Grad-CAM)
│   ├── transforms.py    Augmentation pipelines
│   ├── trainer.py       Training loop (AMP, early stopping)
│   ├── metrics.py       F1, AUC-ROC, confusion matrix
│   └── visualize.py     Grad-CAM, sliding-window
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── export_onnx.py
├── app.py               Gradio web demo
└── configs/config.yaml
```
