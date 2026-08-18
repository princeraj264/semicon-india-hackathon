# KLA PS1 — AI-Based Restoration of Degraded Images

**SEMICON India Hackathon 2026 · Track 1 (KLA) · Problem Statement 1**

A single **Unified Blind Restoration Network (UBRN)** that jointly reverses
speckle noise, Gaussian noise, and downsampling applied in **random order**,
restoring degraded grayscale semiconductor inspection images to clean,
full-resolution ground truth.

Full architecture rationale: `docs/KLA_PS1_Architecture.pdf`.

---

## Official dataset

[Google Drive folder](https://drive.google.com/drive/folders/1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6)

| Split | Contents | Shape | Range |
|---|---|---|---|
| `train.zip` (876 MB) | 3,200 pairs: `train/GT/*.npy` + `train/NoisyLR/*.npy` (matching filenames) | GT `256×256`, NoisyLR `128×128` | GT `[0, 1]`; NoisyLR **unclipped** (observed `[-0.09, 1.88]`) |
| `Test_NoisyLR.zip` (22 MB) | 400 degraded images: `NoisyLR/*.npy` | `128×128` | **unclipped** (observed `[-0.22, 2.16]`) |

Scale factor is **fixed ×2** (128 → 256). Do **not** clip inputs — the
out-of-range values from speckle are signal (see architecture PDF, F2).

```bash
pip install gdown
gdown 1SNPXs_E9GHQuHiiElXOsmnzOxT4PFubx -O train.zip        # train (876 MB)
gdown 1Ayd88_vLwVh-0of3BzL3v94DF7tTutK9 -O Test_NoisyLR.zip # test (22 MB)
unzip -q train.zip -d data/ && unzip -q Test_NoisyLR.zip -d data/
# -> data/train/{GT,NoisyLR}/  and  data/NoisyLR/ (test)
```

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
# 1. MANDATORY sanity check — overfit 2 official pairs (expect PSNR > 40 dB):
python train.py --kla data/train --sanity --epochs 300

# 2. PRIMARY: train on the 3,200 official pairs, mixing in 25% synthetic
#    re-degradations of GT for OOD robustness (test set is part-OOD):
python train.py --kla data/train --epochs 200 --synth-mix 0.25

# 3. (optional) pure synthetic pretraining on any clean grayscale images:
python train.py --data data/train/GT --epochs 100 --scale 2
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
