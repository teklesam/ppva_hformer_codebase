"""
KermanyDataset: PyTorch Dataset for the Kermany 2018 paediatric CXR collection.

Kermany, D. S. et al. (2018). Identifying Medical Diagnoses and Treatable
    Diseases by Image-Based Deep Learning. Cell, 172(5), 1122-1131.
    DOI: 10.1016/j.cell.2018.02.010

Dataset structure (as downloaded from Kaggle/Mendeley):
    chest_xray/
        train/
            NORMAL/      (1349 images)
            PNEUMONIA/   (3883 images)
        test/
            NORMAL/      (234 images)
            PNEUMONIA/   (390 images)

Each image is a greyscale JPEG of variable resolution (roughly 400-1400px).
Images are resized to target_size x target_size for training.

Noise synthesis:
    Foi Poisson-Gaussian noise is applied on-the-fly at batch load time.
    The 'val' split uses fixed noise seeds for reproducibility.

Labels:
    0 = NORMAL, 1 = PNEUMONIA
    Labels are not used during denoising training but are stored so the
    same DataLoader can feed the downstream diagnostic classification task.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

from .noise_simulation import add_foi_noise_preset, PRESETS


NoiseLevelType = Literal["low", "mid", "high", "random"]


class KermanyDataset(Dataset):
    """
    Loads Kermany chest X-rays, applies Foi noise on-the-fly, and returns
    (noisy, clean, label) tuples for training the denoising model.

    Parameters
    ----------
    root_dir    : path to chest_xray/ (contains train/ and test/)
    split       : 'train', 'val', or 'test'
    val_fraction: fraction of training images to hold out as val set
    target_size : resize all images to (target_size, target_size)
    noise_level : 'low', 'mid', 'high', or 'random' (randomise per image)
    augment     : random horizontal flip + small rotation (train split only)
    seed        : random seed for val split and noise augmentation
    """

    LABEL_MAP = {"NORMAL": 0, "PNEUMONIA": 1}

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["train", "val", "test"] = "train",
        val_fraction: float = 0.15,
        target_size: int = 256,
        noise_level: NoiseLevelType = "mid",
        augment: bool = True,
        seed: int = 42,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.target_size = target_size
        self.noise_level = noise_level
        self.augment = augment and split == "train"
        self.seed = seed

        self.paths: list[Path] = []
        self.labels: list[int] = []

        folder = "train" if split in ("train", "val") else "test"
        split_dir = self.root_dir / folder
        for class_name, label in self.LABEL_MAP.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Expected directory not found: {class_dir}")
            images = sorted(class_dir.glob("*.jpeg")) + sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
            self.paths.extend(images)
            self.labels.extend([label] * len(images))

        if split in ("train", "val"):
            rng = random.Random(seed)
            indices = list(range(len(self.paths)))
            rng.shuffle(indices)
            n_val = int(len(indices) * val_fraction)
            val_idx = set(indices[:n_val])
            train_idx = set(indices[n_val:])
            keep = val_idx if split == "val" else train_idx
            self.paths = [self.paths[i] for i in range(len(self.paths)) if i in keep]
            self.labels = [self.labels[i] for i in range(len(self.labels)) if i in keep]

        self._resize = transforms.Resize((target_size, target_size), antialias=True)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        img = Image.open(self.paths[idx]).convert("L")  # greyscale
        img_t: torch.Tensor = TF.to_tensor(img)          # [1, H, W], range [0,1]
        img_t = self._resize(img_t)                       # [1, target_size, target_size]

        if self.augment:
            if random.random() > 0.5:
                img_t = TF.hflip(img_t)
            angle = random.uniform(-5, 5)
            img_t = TF.rotate(img_t, angle)

        level = (
            random.choice(["low", "mid", "high"])
            if self.noise_level == "random"
            else self.noise_level
        )

        if self.split == "val":
            g = torch.Generator()
            g.manual_seed(self.seed + idx)
            noise = torch.randn(img_t.shape, generator=g)
            params = PRESETS[level]
            variance = params["a"] * img_t.clamp(min=0.0) + params["b"]
            noisy = (img_t + noise * variance.sqrt()).clamp(0.0, 1.0)
        else:
            noisy = add_foi_noise_preset(img_t, level=level)

        return noisy, img_t, self.labels[idx]


def make_loaders(
    root_dir: str | Path,
    batch_size: int = 16,
    target_size: int = 256,
    noise_level: NoiseLevelType = "mid",
    num_workers: int = 4,
    seed: int = 42,
) -> dict[str, torch.utils.data.DataLoader]:
    """Convenience factory -- returns train/val/test DataLoaders."""
    loaders = {}
    for split in ("train", "val", "test"):
        ds = KermanyDataset(
            root_dir=root_dir,
            split=split,
            target_size=target_size,
            noise_level=noise_level,
            augment=(split == "train"),
            seed=seed,
        )
        loaders[split] = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders
