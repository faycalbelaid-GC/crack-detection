"""Export a trained checkpoint to ONNX for deployment.

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/best.pth
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path

import torch

from src.model import CrackDetector


def load_model(checkpoint_path: str, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device)
    cfg = state["cfg"]
    model = CrackDetector(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["model"]["num_classes"],
        dropout=0.0,
        pretrained=False,
    )
    model.load_state_dict(state["model_state"])
    return model.to(device).eval(), cfg


class InferenceWrapper(torch.nn.Module):
    """Wraps CrackDetector to output only crack probabilities (ONNX-friendly)."""

    def __init__(self, model: CrackDetector):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return torch.softmax(out["crack_logits"], dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None, help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    device = torch.device("cpu")
    model, cfg = load_model(args.checkpoint, device)
    wrapper = InferenceWrapper(model)

    image_size = cfg["data"]["image_size"]
    dummy = torch.randn(1, 3, image_size, image_size)

    out_path = args.output or args.checkpoint.replace(".pth", ".onnx")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        input_names=["image"],
        output_names=["probabilities"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "probabilities": {0: "batch_size"},
        },
        opset_version=args.opset,
    )
    print(f"ONNX model saved to {out_path}")

    # Verify
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        x_np = dummy.numpy()
        result = sess.run(None, {"image": x_np})
        print(f"ONNX inference OK — output shape: {result[0].shape}")
    except ImportError:
        print("onnxruntime not installed; skipping validation (pip install onnxruntime)")


if __name__ == "__main__":
    main()
