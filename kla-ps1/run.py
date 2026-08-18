#!/usr/bin/env python3
"""
KLA PS1 -- standalone evaluation/inference script (rule-compliant).

Usage:
    python run.py <input_dir> <output_dir> [--scale 2] [--weights models/model.pt]

Guarantees (per submission rules):
  - reads every .npy in <input_dir> (raw values, NOT clipped -- speckle
    overshoot is signal), writes restored .npy to <output_dir> with the SAME
    filename, shape (H*scale, W*scale), dtype float32, values in [0, 1],
    no NaN/Inf. Creates <output_dir> if missing. Fully offline: weights are
    loaded from the local models/ folder, nothing is downloaded.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import build_model  # noqa: E402

PAD_MULTIPLE = 8  # U-body has 3 downsamples -> spatial dims must divide by 8


def load_npy(path: str) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:  # tolerate (H, W, 1) / (1, H, W)
        arr = arr.squeeze()
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected 2D grayscale, got shape {arr.shape}")
    return arr


def pad_to_multiple(x: torch.Tensor, m: int):
    _, _, h, w = x.shape
    ph, pw = (m - h % m) % m, (m - w % m) % m
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, h, w


def main():
    parser = argparse.ArgumentParser(description="KLA PS1 restoration inference")
    parser.add_argument("input_dir", help="folder of degraded .npy images")
    parser.add_argument("output_dir", help="folder to write restored .npy images")
    parser.add_argument("--scale", type=int, default=2, choices=[1, 2, 4],
                        help="super-resolution factor (default 2)")
    parser.add_argument("--weights", default=None,
                        help="path to model weights (default models/model.pt)")
    parser.add_argument("--batch", type=int, default=4, help="batch size")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    weights = args.weights or os.path.join(here, "models", "model.pt")

    os.makedirs(args.output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    if not paths:
        print(f"No .npy files found in {args.input_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"

    model = build_model(scale=args.scale)
    if os.path.exists(weights):
        state = torch.load(weights, map_location="cpu")
        state = state.get("model", state)  # accept checkpoint dicts too
        model.load_state_dict(state)
        print(f"Loaded weights: {weights}")
    else:
        print(f"WARNING: weights not found at {weights}; using random init "
              f"(outputs will be near-bicubic).")
    model.to(device).eval()
    if use_fp16:
        model.half()

    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        # group same-shaped images into batches for throughput
        i = 0
        while i < len(paths):
            batch_paths = [paths[i]]
            shape0 = load_npy(paths[i]).shape
            j = i + 1
            while (j < len(paths) and len(batch_paths) < args.batch
                   and load_npy(paths[j]).shape == shape0):
                batch_paths.append(paths[j])
                j += 1

            arrs = [load_npy(p) for p in batch_paths]
            x = torch.from_numpy(np.stack(arrs)).unsqueeze(1).to(device)
            if use_fp16:
                x = x.half()
            x, h0, w0 = pad_to_multiple(x, PAD_MULTIPLE)

            y = model(x)
            y = y[:, :, : h0 * args.scale, : w0 * args.scale]
            y = y.float().cpu().numpy()

            for p, out in zip(batch_paths, y):
                out = np.nan_to_num(out.squeeze(0), nan=0.0, posinf=1.0, neginf=0.0)
                out = np.clip(out, 0.0, 1.0).astype(np.float32)  # (H, W) in [0,1]
                np.save(os.path.join(args.output_dir, os.path.basename(p)), out)
                n_done += 1
            i = j

    dt = time.time() - t0
    print(f"Restored {n_done} images in {dt:.2f}s ({dt / max(n_done, 1):.3f}s/img) "
          f"on {device.type.upper()}")


if __name__ == "__main__":
    main()
