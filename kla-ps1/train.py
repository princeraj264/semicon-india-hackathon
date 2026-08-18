#!/usr/bin/env python3
"""
KLA PS1 -- reproducible training script.

Usage:
    # mandatory sanity check first (overfit 2 images; PSNR should exceed ~40 dB):
    python train.py --data data/train_clean --sanity

    # synthetic randomized-degradation training on clean images:
    python train.py --data data/train_clean --epochs 200 --scale 2

    # fine-tune on official KLA pairs:
    python train.py --paired data/degraded data/ground_truth \
        --resume models/model.pt --epochs 50 --lr 1e-5

Saves best checkpoint (val PSNR) to models/model.pt -- the file run.py loads.
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.dataset import PairedDataset, SyntheticDataset  # noqa: E402
from src.losses import CompositeLoss, psnr, ssim_metric  # noqa: E402
from src.model import build_model  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser(description="Train UBRN for KLA PS1")
    p.add_argument("--data", help="folder of CLEAN images (synthetic degradation mode)")
    p.add_argument("--paired", nargs=2, metavar=("DEGRADED_DIR", "GT_DIR"),
                   help="official paired data folders")
    p.add_argument("--scale", type=int, default=2, choices=[1, 2, 4])
    p.add_argument("--dim", type=int, default=32, help="base channel width")
    p.add_argument("--patch", type=int, default=64, help="low-res patch size")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--resume", help="checkpoint to resume/fine-tune from")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-lpips", action="store_true", help="disable LPIPS loss term")
    p.add_argument("--sanity", action="store_true",
                   help="overfit 2 images (mandatory pipeline sanity check)")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- data ----------------
    if args.paired:
        ds = PairedDataset(args.paired[0], args.paired[1],
                           patch_size=args.patch, scale=args.scale)
    elif args.data:
        ds = SyntheticDataset(args.data, patch_size=args.patch,
                              scale=args.scale, repeat=4)
    else:
        p.error("provide --data (clean images) or --paired DEGRADED_DIR GT_DIR")

    if args.sanity:
        ds = Subset(ds, list(range(min(2, len(ds)))))
        print("[sanity] overfitting 2 samples -- run ~300 epochs and expect "
              "PSNR > 40 dB. If it plateaus low, the loader/normalization/loss "
              "has a bug.")

    n_val = max(1, len(ds) // 20) if not args.sanity else 1
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    batch_size = min(args.batch, max(1, len(train_ds)))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True,
                          drop_last=len(train_ds) > batch_size)
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=2)

    # ---------------- model / loss / optim ----------------
    model = build_model(scale=args.scale, dim=args.dim).to(device)
    if args.resume and os.path.exists(args.resume):
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state.get("model", state))
        print(f"Resumed from {args.resume}")
    n_params = sum(x.numel() for x in model.parameters())
    print(f"UBRN scale=x{args.scale} dim={args.dim}: {n_params / 1e6:.2f}M params")

    criterion = CompositeLoss(use_lpips=not args.no_lpips).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    os.makedirs("models", exist_ok=True)
    best_psnr = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for deg, gt in train_dl:
            deg, gt = deg.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(deg)
                loss = criterion(pred, gt)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            running += loss.item()
        sched.step()

        # ---------------- validation ----------------
        model.eval()
        v_psnr, v_ssim = 0.0, 0.0
        with torch.no_grad():
            for deg, gt in val_dl:
                deg, gt = deg.to(device), gt.to(device)
                pred = model(deg)
                v_psnr += psnr(pred, gt)
                v_ssim += ssim_metric(pred, gt)
        v_psnr /= len(val_dl)
        v_ssim /= len(val_dl)

        marker = ""
        if v_psnr > best_psnr:
            best_psnr = v_psnr
            torch.save({"model": model.state_dict(),
                        "scale": args.scale, "dim": args.dim,
                        "epoch": epoch, "psnr": v_psnr},
                       "models/model.pt")
            marker = "  <- saved models/model.pt"
        print(f"epoch {epoch:4d}/{args.epochs} | loss {running / len(train_dl):.4f} | "
              f"val PSNR {v_psnr:.2f} dB | val SSIM {v_ssim:.4f} | "
              f"{time.time() - t0:.1f}s{marker}")

    print(f"Done. Best val PSNR: {best_psnr:.2f} dB -> models/model.pt")


if __name__ == "__main__":
    main()
