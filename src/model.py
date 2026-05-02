"""ResNet-based crack detector with Grad-CAM support."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2, 2048),
}


class CrackDetector(nn.Module):
    """
    Fine-tuned ResNet for binary crack classification.

    Optionally predicts surface type (deck / pavement / wall) as an
    auxiliary task, which acts as a regulariser and improves feature
    learning on small datasets.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 2,
        num_surfaces: int = 3,
        dropout: float = 0.5,
        pretrained: bool = True,
        multitask: bool = True,
    ):
        super().__init__()
        assert backbone in BACKBONES, f"Unsupported backbone: {backbone}. Choose from {list(BACKBONES)}"

        factory, weights, feat_dim = BACKBONES[backbone]
        base = factory(weights=weights if pretrained else None)

        # Keep feature extractor, drop the original FC head
        self.features = nn.Sequential(*list(base.children())[:-2])  # → (B, C, H, W)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

        self.multitask = multitask
        if multitask:
            self.surface_head = nn.Sequential(
                nn.Dropout(dropout / 2),
                nn.Linear(feat_dim, num_surfaces),
            )

        self._feat_dim = feat_dim
        self.backbone_name = backbone

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat_map = self.features(x)                    # (B, C, H, W)
        pooled = self.pool(feat_map).flatten(1)        # (B, C)

        out: Dict[str, torch.Tensor] = {
            "crack_logits": self.classifier(pooled),
            "feat_map": feat_map,
        }
        if self.multitask:
            out["surface_logits"] = self.surface_head(pooled)
        return out

    def freeze_backbone(self, freeze: bool = True) -> None:
        for p in self.features.parameters():
            p.requires_grad = not freeze

    def unfreeze_layer(self, layer_idx: int) -> None:
        children = list(self.features.children())
        for p in children[layer_idx].parameters():
            p.requires_grad = True


class CrackLoss(nn.Module):
    """Cross-entropy for crack + optional surface auxiliary loss."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        aux_weight: float = 0.3,
        multitask: bool = True,
    ):
        super().__init__()
        self.crack_ce = nn.CrossEntropyLoss(weight=class_weights)
        self.surface_ce = nn.CrossEntropyLoss()
        self.aux_weight = aux_weight
        self.multitask = multitask

    def forward(self, outputs: Dict, crack_labels: torch.Tensor, surface_labels: torch.Tensor) -> Dict:
        crack_loss = self.crack_ce(outputs["crack_logits"], crack_labels)
        total = crack_loss

        losses = {"crack": crack_loss}
        if self.multitask and "surface_logits" in outputs:
            surface_loss = self.surface_ce(outputs["surface_logits"], surface_labels)
            total = crack_loss + self.aux_weight * surface_loss
            losses["surface"] = surface_loss

        losses["total"] = total
        return losses


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Usage:
        cam = GradCAM(model)
        heatmap = cam(image_tensor, class_idx=1)   # class 1 = cracked
        cam.remove_hooks()
    """

    def __init__(self, model: CrackDetector, target_layer: Optional[nn.Module] = None):
        self.model = model
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None

        layer = target_layer or self._get_default_layer()
        self._fwd_hook = layer.register_forward_hook(self._save_activations)
        self._bwd_hook = layer.register_full_backward_hook(self._save_gradients)

    def _get_default_layer(self) -> nn.Module:
        # Last convolutional block of the feature extractor
        return list(self.model.features.children())[-1]

    def _save_activations(self, module, input, output) -> None:
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    @torch.enable_grad()
    def __call__(self, x: torch.Tensor, class_idx: int = 1) -> torch.Tensor:
        """
        Returns a (H, W) heatmap in [0, 1], same spatial size as input.
        """
        self.model.eval()
        x = x.unsqueeze(0) if x.ndim == 3 else x
        x = x.requires_grad_(True)

        outputs = self.model(x)
        logits = outputs["crack_logits"]

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Global-average-pool the gradients
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)

        # Resize to input resolution
        h, w = x.shape[-2:]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze()  # (H, W)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.cpu()

    def remove_hooks(self) -> None:
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def build_model(cfg: dict) -> CrackDetector:
    return CrackDetector(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        pretrained=cfg["model"]["pretrained"],
    )
