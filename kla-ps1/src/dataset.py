"""
Datasets for KLA PS1.

Two modes:
  1. PairedDataset      -- official KLA pairs: degraded/ + ground_truth/ folders
                           with matching .npy filenames.
  2. SyntheticDataset   -- any folder of clean grayscale images (.npy/.png/.jpg);
                           degraded inputs are synthesized ON THE FLY with the
                           randomized black-box chain (src/degradations.py).
                           This is the OOD-generalization workhorse.

Both apply the 8-config geometric augmentation (flips + 90-degree rotations,
W2S Sec. 2.1) and random crops. Inputs are NOT clipped to [0,1].
"""

import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .degradations import degrade

IMG_EXTS = (".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _load_gray(path: str) -> np.ndarray:
    """Load an image as float32 (H, W). PNG/JPG are normalized to [0,1];
    .npy files are used raw (do not clip -- overshoot is signal)."""
    if path.endswith(".npy"):
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1) if arr.shape[-1] <= 4 else arr[0]
        return arr
    from PIL import Image

    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def _augment8(gt: torch.Tensor, deg: torch.Tensor):
    """Same random flip/rotation applied to both tensors (C, H, W)."""
    if random.random() < 0.5:
        gt, deg = torch.flip(gt, [-1]), torch.flip(deg, [-1])
    if random.random() < 0.5:
        gt, deg = torch.flip(gt, [-2]), torch.flip(deg, [-2])
    k = random.randint(0, 3)
    if k:
        gt, deg = torch.rot90(gt, k, [-2, -1]), torch.rot90(deg, k, [-2, -1])
    return gt, deg


class PairedDataset(Dataset):
    """Official KLA pairs. degraded_dir and gt_dir hold matching filenames."""

    def __init__(self, degraded_dir: str, gt_dir: str, patch_size: int = 128,
                 scale: int = 2, augment: bool = True):
        self.deg_paths = sorted(
            p for p in glob.glob(os.path.join(degraded_dir, "*")) if p.lower().endswith(IMG_EXTS)
        )
        self.gt_dir = gt_dir
        self.patch = patch_size
        self.scale = scale
        self.augment = augment
        if not self.deg_paths:
            raise FileNotFoundError(f"No images found in {degraded_dir}")

    def __len__(self):
        return len(self.deg_paths)

    def __getitem__(self, idx):
        dpath = self.deg_paths[idx]
        gpath = os.path.join(self.gt_dir, os.path.basename(dpath))
        deg = torch.from_numpy(_load_gray(dpath)).unsqueeze(0)
        gt = torch.from_numpy(_load_gray(gpath)).unsqueeze(0)

        # aligned crop: patch on degraded, patch*scale on GT.
        # Random position when augmenting (normal training); fixed center
        # crop when not (e.g. --sanity), so the target is truly static.
        _, h, w = deg.shape
        p = min(self.patch, h, w)
        if self.augment:
            top, left = random.randint(0, h - p), random.randint(0, w - p)
        else:
            top, left = (h - p) // 2, (w - p) // 2
        deg = deg[:, top:top + p, left:left + p]
        gt = gt[:, top * self.scale:(top + p) * self.scale,
                left * self.scale:(left + p) * self.scale]

        if self.augment:
            gt, deg = _augment8(gt, deg)
        return deg, gt


class SyntheticDataset(Dataset):
    """Clean images only; degradation synthesized on the fly (random order)."""

    def __init__(self, clean_dir: str, patch_size: int = 128, scale: int = 2,
                 augment: bool = True, repeat: int = 1):
        self.paths = sorted(
            p for p in glob.glob(os.path.join(clean_dir, "*")) if p.lower().endswith(IMG_EXTS)
        ) * repeat
        self.patch = patch_size  # LOW-RES patch size; GT patch = patch * scale
        self.scale = scale
        self.augment = augment
        if not self.paths:
            raise FileNotFoundError(f"No images found in {clean_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        gt = torch.from_numpy(_load_gray(self.paths[idx])).unsqueeze(0)
        gp = self.patch * self.scale
        _, h, w = gt.shape
        if h < gp or w < gp:  # pad small images reflectively
            gt = torch.nn.functional.pad(
                gt, (0, max(0, gp - w), 0, max(0, gp - h)), mode="reflect"
            )
            _, h, w = gt.shape
        top, left = random.randint(0, h - gp), random.randint(0, w - gp)
        gt = gt[:, top:top + gp, left:left + gp].clamp(0, 1)

        deg = degrade(gt, scale=self.scale)  # unclipped by design
        if self.augment:
            gt, deg = _augment8(gt, deg)
        return deg, gt