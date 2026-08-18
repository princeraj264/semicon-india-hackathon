# KLA PS1 — AI-Based Restoration of Degraded Images

**SEMICON India Hackathon 2026 · Track 1 (KLA) · Problem Statement 1**

A single **Unified Blind Restoration Network (UBRN)** that jointly reverses
speckle noise, Gaussian noise, and downsampling applied in **random order**,
restoring degraded grayscale semiconductor inspection images to clean,
full-resolution ground truth.

Full architecture rationale: `docs/KLA_PS1_Architecture.pdf`.

---

## Quick start — inference (what reviewers run)

```bash
pip install -r requirements.txt
python run.py <input_dir> <output_dir>            # default x2 super-resolution
python run.py <input_dir> <output_dir> --scale 4  # x4 variant
```

- Reads every `.npy` in `<input_dir>` (raw values, **not** clipped — speckle
  overshoot is signal).
- Writes restored `.npy` to `<output_dir>` with the **same filename**,
  shape `(H*scale, W*scale)`, dtype `float32`, values in `[0, 1]`,
  guaranteed **no NaN/Inf**.
- Creates `<output_dir>` automatically. Fully **offline**: weights load from
  `models/model.pt`, nothing is downloaded.
- FP16 + shape-grouped batching on GPU for throughput.

## Training (reproducible from scratch)

```bash
# 1. MANDATORY sanity check — overfit 2 images (expect PSNR > 40 dB):
python train.py --data data/train_clean --sanity

# 2. Randomized-degradation pretraining on any clean grayscale images:
python train.py --data data/train_clean --epochs 200 --scale 2

# 3. Fine-tune on the official KLA pairs:
python train.py --paired data/degraded data/ground_truth \
    --resume models/model.pt --epochs 50 --lr 1e-5
```

Best checkpoint (val PSNR) is saved to `models/model.pt` — the exact file
`run.py` loads. Seeded (`--seed 42`) for reproducibility.

## Repository layout

```
run.py               standalone inference script (reviewers run this as-is)
train.py             training script (sanity / synthetic / paired modes)
src/model.py         UBRN: noise estimator + Restormer U-body + PixelShuffle head
src/degradations.py  randomized black-box degradation synthesizer
src/losses.py        Charbonnier + SSIM + LPIPS + FFT composite loss (training only)
src/dataset.py       paired + on-the-fly synthetic datasets, 8-way augmentation
models/              trained weights (model.pt)
docs/                architecture PDF
requirements.txt     pinned dependencies (pip freeze)
```

## Architecture summary

| Stage | Design | Source paper |
|---|---|---|
| Noise conditioning | 3-layer CNN noise-level map, concatenated as input channel (soft conditioning, no routing) | FFDNet |
| Single blind model for mixed degradations | one network, all tasks | DnCNN |
| Speckle handling | end-to-end in-network; **no log-transform** (mixed additive noise makes `log` produce NaN) | ID-CNN, Dalsasso et al. |
| Body | 4-level U-shape of MDTA + GDFN blocks (channel attention, linear in resolution) | Restormer |
| Global skip + reconstruction | long skip from shallow features; sub-pixel (PixelShuffle) upsampling head | SwinIR |
| Joint denoise + SR training | trained on the true joint degradation, never sequential | W2S |
| Loss | `L1(Charbonnier) + 0.2·SSIM + 0.1·LPIPS + 0.05·FFT` | SwinIR / W2S / KLA frequency hint |

~9M parameters (dim=32) — inside the accuracy/throughput "golden ratio" zone.

## Output guarantees

Every saved output passes: `np.nan_to_num` → `np.clip(0, 1)` → `float32 (H, W)`.

## Tech stack

Python 3.10+, PyTorch 2.x, NumPy. GPU: trained/benchmarked with FP16 autocast;
runs on CPU as fallback. LPIPS is a **training-only** optional dependency —
`run.py` never imports it.
