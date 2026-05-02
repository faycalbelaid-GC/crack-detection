"""SDNET2018 crack detection dataset loader.

Dataset structure:
    SDNET2018/
    ├── D/          Bridge Decks
    │   ├── CD/     Cracked Deck
    │   └── UD/     Uncracked Deck
    ├── P/          Pavements
    │   ├── CP/     Cracked Pavement
    │   └── UP/     Uncracked Pavement
    └── W/          Walls
        ├── CW/     Cracked Wall
        └── UW/     Uncracked Wall
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split


SURFACE_TYPES = {"D": 0, "P": 1, "W": 2}
SURFACE_NAMES = {0: "Deck", 1: "Pavement", 2: "Wall"}
CLASS_NAMES = {0: "Non-cracked", 1: "Cracked"}
IMG_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}


class SDNET2018Dataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        val_size: float = 0.15,
        test_size: float = 0.15,
        seed: int = 42,
        surface_filter: Optional[List[str]] = None,
    ):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.root = Path(root)
        self.split = split
        self.transform = transform

        all_samples = self._scan_dataset(surface_filter)
        if not all_samples:
            raise RuntimeError(f"No images found in {root}. Check dataset path.")

        labels = [s[1] for s in all_samples]
        train_val, test = train_test_split(
            all_samples, test_size=test_size, stratify=labels, random_state=seed
        )
        adjusted_val = val_size / (1.0 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=adjusted_val,
            stratify=[s[1] for s in train_val],
            random_state=seed,
        )

        self.samples: List[Tuple[str, int, int]] = {"train": train, "val": val, "test": test}[split]
        self._log_stats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_dataset(self, surface_filter: Optional[List[str]]) -> List[Tuple[str, int, int]]:
        samples: List[Tuple[str, int, int]] = []
        for surface in ["D", "P", "W"]:
            if surface_filter and surface not in surface_filter:
                continue
            surface_path = self.root / surface
            if not surface_path.exists():
                continue
            surface_label = SURFACE_TYPES[surface]
            for crack_prefix, crack_label in [("C", 1), ("U", 0)]:
                subdir = surface_path / f"{crack_prefix}{surface}"
                if not subdir.exists():
                    continue
                for p in subdir.iterdir():
                    if p.suffix in IMG_EXTENSIONS:
                        samples.append((str(p), crack_label, surface_label))
        return samples

    def _log_stats(self) -> None:
        n_cracked = sum(1 for s in self.samples if s[1] == 1)
        n_uncracked = len(self.samples) - n_cracked
        print(
            f"[{self.split:5s}] {len(self.samples):6d} images  "
            f"| cracked: {n_cracked}  non-cracked: {n_uncracked}"
        )

    # ------------------------------------------------------------------
    # Sampling utilities
    # ------------------------------------------------------------------

    def get_class_weights(self) -> torch.Tensor:
        n = len(self.samples)
        n_pos = sum(s[1] for s in self.samples)
        n_neg = n - n_pos
        w_neg = n / (2.0 * n_neg) if n_neg > 0 else 1.0
        w_pos = n / (2.0 * n_pos) if n_pos > 0 else 1.0
        return torch.tensor([w_neg, w_pos], dtype=torch.float32)

    def get_sampler(self) -> WeightedRandomSampler:
        class_w = self.get_class_weights().numpy()
        weights = [float(class_w[s[1]]) for s in self.samples]
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path, crack_label, surface_label = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        return {
            "image": image,
            "crack_label": torch.tensor(crack_label, dtype=torch.long),
            "surface_label": torch.tensor(surface_label, dtype=torch.long),
            "path": img_path,
        }


def create_dataloaders(
    data_dir: str,
    train_transform,
    val_transform,
    batch_size: int = 32,
    num_workers: int = 4,
    use_sampler: bool = True,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    kwargs = dict(val_size=val_size, test_size=test_size, seed=seed)
    train_ds = SDNET2018Dataset(data_dir, split="train", transform=train_transform, **kwargs)
    val_ds = SDNET2018Dataset(data_dir, split="val", transform=val_transform, **kwargs)
    test_ds = SDNET2018Dataset(data_dir, split="test", transform=val_transform, **kwargs)

    sampler = train_ds.get_sampler() if use_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
