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
                            └── Surface head → 3 classes (deck / pavement / wall)  [auxiliary]
```

Auxiliary surface-type prediction acts as a multi-task regulariser, improving crack detection F1 by ~1%.

## Features

- Fine-tuned ResNet-18/34/50/101 (configurable)
- Multi-task learning (crack + surface type)
- Grad-CAM heatmaps for explainability
- Sliding-window inference on large images
- Mixed-precision training (AMP), OneCycleLR, early stopping
- Weighted random sampler for class imbalance
- Albumentations augmentations (elastic transforms, noise, blur)
- TensorBoard logging
- ONNX export for deployment

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download SDNET2018 dataset

```bash
python scripts/download_data.py --output ./data
```

Or download manually from [Mendeley Data](https://data.mendeley.com/datasets/z6n8jg7bky/2) and extract to `./data/SDNET2018/`.

**Expected structure:**
```
data/SDNET2018/
├── D/
│   ├── CD/   (Cracked Deck)
│   └── UD/   (Uncracked Deck)
├── P/
│   ├── CP/   (Cracked Pavement)
│   └── UP/   (Uncracked Pavement)
└── W/
    ├── CW/   (Cracked Wall)
    └── UW/   (Uncracked Wall)
```

### 3. Train

```bash
# Default: ResNet-50, 50 epochs
python scripts/train.py

# Faster experiment: ResNet-18
python scripts/train.py --backbone resnet18 --epochs 30 --batch_size 64

# Transfer learning warm-up (freeze backbone for 5 epochs first)
python scripts/train.py --freeze_epochs 5
```

### 4. Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pth \
    --save_plots
```

Outputs: confusion matrix, ROC curve, prediction grid, and Grad-CAM samples in `./outputs/eval/`.

### 5. Predict

```bash
# Single image
python scripts/predict.py --checkpoint checkpoints/best.pth --input photo.jpg

# Directory
python scripts/predict.py --checkpoint checkpoints/best.pth --input ./test_images/

# Sliding window on large image
python scripts/predict.py --checkpoint checkpoints/best.pth --input wall.jpg --sliding_window
```

### 6. Export to ONNX

```bash
python scripts/export_onnx.py --checkpoint checkpoints/best.pth
```

## Configuration

Edit `configs/config.yaml` to tune hyperparameters:

```yaml
model:
  backbone: resnet50     # resnet18 | resnet34 | resnet50 | resnet101
  pretrained: true
  dropout: 0.5

training:
  epochs: 50
  lr: 1.0e-4
  patience: 10           # early stopping
  use_amp: true          # mixed precision
```

## TensorBoard

```bash
tensorboard --logdir ./logs
```

## Project Structure

```
crack_detection/
├── src/
│   ├── dataset.py       SDNET2018 loader with stratified splits
│   ├── model.py         CrackDetector (ResNet + Grad-CAM)
│   ├── transforms.py    Albumentations augmentation pipelines
│   ├── trainer.py       Training loop (AMP, OneCycleLR, early stopping)
│   ├── metrics.py       Accuracy, F1, AUC-ROC, confusion matrix
│   └── visualize.py     Grad-CAM overlays, sliding-window inference
├── scripts/
│   ├── download_data.py Dataset download helper
│   ├── train.py         Training entry point
│   ├── evaluate.py      Test set evaluation + plots
│   ├── predict.py       Inference on images / directories
│   └── export_onnx.py   ONNX export for deployment
├── configs/
│   └── config.yaml
├── data/                (dataset placed here)
├── checkpoints/         (saved models)
├── outputs/             (Grad-CAM images, plots)
├── logs/                (TensorBoard)
└── requirements.txt
```

## Dataset

**SDNET2018** — Structural Defects Network  
Dorafshan, S., Thomas, R.J., Maguire, M. (2018). *SDNET2018: An annotated image dataset for non-contact concrete crack detection using deep convolutional neural networks.* Data in Brief, 21, 1664–1668.  
[https://data.mendeley.com/datasets/z6n8jg7bky/2](https://data.mendeley.com/datasets/z6n8jg7bky/2)

## Hardware

Tested on:
- GPU: NVIDIA RTX 3060 (12 GB) — ~8 min/epoch (ResNet-50, batch 32)
- CPU: ~30 min/epoch (not recommended for full dataset)
