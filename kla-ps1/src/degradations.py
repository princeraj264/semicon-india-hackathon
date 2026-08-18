"""
Randomized black-box degradation synthesizer (KLA PS1).

Mirrors the competition's degradation model:
  - Gaussian noise  (additive, zero-mean, variable sigma)
  - Speckle noise   (multiplicative -> pushes values beyond [0,1]; DO NOT clip)
  - Gaussian blur   (optional, mild)
  - Downsampling    (x2 or x4, bicubic/area)

Degradations are applied in RANDOM ORDER with random strengths, per KLA's
"do not read into the order" guidance. Training on the true joint, randomized
degradation is the key OOD-generalization lever (W2S / Real-ESRGAN insight).

IMPORTANT: outputs are intentionally NOT clipped to [0,1] -- the out-of-range
values from speckle are signal the network must learn from.
"""

import random

import numpy as np
import torch
import torch.nn.functional as F


def add_gaussian_noise(x: torch.Tensor, sigma_range=(0.01, 0.1)) -> torch.Tensor:
    sigma = random.uniform(*sigma_range)
    return x + torch.randn_like(x) * sigma


def add_speckle_noise(x: torch.Tensor, sigma_range=(0.05, 0.25)) -> torch.Tensor:
    """Multiplicative noise: y = x * (1 + n), n ~ N(0, sigma^2). Scales with brightness."""
    sigma = random.uniform(*sigma_range)
    return x * (1.0 + torch.randn_like(x) * sigma)


def gaussian_blur(x: torch.Tensor, kernel_range=(3, 5), sigma_range=(0.3, 1.2)) -> torch.Tensor:
    k = random.choice(range(kernel_range[0], kernel_range[1] + 1, 2))
    sigma = random.uniform(*sigma_range)
    ax = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = (g[:, None] @ g[None, :]).to(x.device)
    kernel = kernel.expand(x.shape[-3], 1, k, k)
    pad = k // 2
    x4 = x.unsqueeze(0) if x.dim() == 3 else x
    out = F.conv2d(F.pad(x4, (pad,) * 4, mode="reflect"), kernel, groups=x4.shape[1])
    return out.squeeze(0) if x.dim() == 3 else out


def downsample(x: torch.Tensor, scale: int) -> torch.Tensor:
    mode = random.choice(["bicubic", "area", "bilinear"])
    x4 = x.unsqueeze(0) if x.dim() == 3 else x
    kwargs = {} if mode == "area" else {"align_corners": False}
    out = F.interpolate(x4, scale_factor=1.0 / scale, mode=mode, **kwargs)
    return out.squeeze(0) if x.dim() == 3 else out


def degrade(clean: torch.Tensor, scale: int = 2, blur_prob: float = 0.5):
    """Apply the randomized degradation chain to a clean image.

    Args:
        clean: (1, H, W) or (B, 1, H, W) tensor in [0, 1].
        scale: downsampling factor (2 or 4).
        blur_prob: probability of including a mild blur in the chain.

    Returns:
        degraded tensor at 1/scale resolution, values NOT clipped.
    """
    ops = ["gaussian", "speckle"]
    if random.random() < blur_prob:
        ops.append("blur")
    # downsample position is random within the chain (random-order black box)
    ops.insert(random.randint(0, len(ops)), "down")
    random.shuffle(ops)
    # ensure exactly one downsample and it is respected wherever it lands
    x = clean
    for op in ops:
        if op == "gaussian":
            x = add_gaussian_noise(x)
        elif op == "speckle":
            x = add_speckle_noise(x)
        elif op == "blur":
            x = gaussian_blur(x)
        elif op == "down":
            x = downsample(x, scale)
    return x  # intentionally unclipped


def degrade_numpy(clean: np.ndarray, scale: int = 2) -> np.ndarray:
    """Convenience wrapper for numpy (H, W) float arrays."""
    t = torch.from_numpy(clean.astype(np.float32)).unsqueeze(0)
    d = degrade(t, scale=scale)
    return d.squeeze(0).numpy()
