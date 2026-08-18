"""
Unified Blind Restoration Network (UBRN) for KLA PS1.

Architecture (see docs/KLA_PS1_Architecture.pdf):
  raw degraded input (unclipped)
    -> NoiseEstimator (3-layer CNN)  -- concat as extra channel   [FFDNet]
    -> shallow 3x3 conv                                            [SwinIR]
    -> 4-level U-shaped body of MDTA + GDFN transformer blocks     [Restormer]
    -> global long skip from shallow features                      [SwinIR]
    -> PixelShuffle x2 / x4 reconstruction head                    [SwinIR / W2S]
    -> 3x3 output conv

Single model handles Gaussian + speckle + downsampling jointly
(random order), per DnCNN's single-model multi-task result and
W2S's finding that joint denoise+SR must be trained jointly.
"""

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Layer norm (bias-free, channel-wise) -- Restormer style
# --------------------------------------------------------------------------
class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        # x: (B, C, H, W) -> norm over C
        b, c, h, w = x.shape
        y = x.permute(0, 2, 3, 1)  # B H W C
        sigma = y.var(-1, keepdim=True, unbiased=False)
        y = y / torch.sqrt(sigma + 1e-5) * self.weight
        return y.permute(0, 3, 1, 2)


# --------------------------------------------------------------------------
# MDTA: Multi-Dconv-head Transposed Attention (Restormer, arXiv:2111.09881)
# Attention across CHANNELS (C x C map) -> linear cost in resolution.
# --------------------------------------------------------------------------
class MDTA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=False
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        head_c = c // self.num_heads
        q = q.reshape(b, self.num_heads, head_c, h * w)
        k = k.reshape(b, self.num_heads, head_c, h * w)
        v = v.reshape(b, self.num_heads, head_c, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # (B, heads, C', C')
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B, heads, C', HW)
        out = out.reshape(b, c, h, w)
        return self.project_out(out)


# --------------------------------------------------------------------------
# GDFN: Gated-Dconv Feed-forward Network (Restormer)
# --------------------------------------------------------------------------
class GDFN(nn.Module):
    def __init__(self, dim, expansion=2.66):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, padding=1,
            groups=hidden * 2, bias=False,
        )
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = BiasFreeLayerNorm(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = BiasFreeLayerNorm(dim)
        self.ffn = GDFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# --------------------------------------------------------------------------
# Down / Up sampling inside the U-body (Restormer style)
# --------------------------------------------------------------------------
class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


# --------------------------------------------------------------------------
# Noise estimator: tiny 3-layer CNN -> 1-channel noise-level map [FFDNet]
# Soft conditioning replaces hard DnCNN/FFDNet routing.
# --------------------------------------------------------------------------
class NoiseEstimator(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------
class UBRN(nn.Module):
    """Unified Blind Restoration Network.

    Args:
        dim: base channel width (default 32 -> ~9M params).
        num_blocks: transformer blocks per U-level (encoder side).
        heads: attention heads per level.
        scale: super-resolution factor of the reconstruction head (2 or 4).
    """

    def __init__(self, dim=32, num_blocks=(2, 3, 3, 4), heads=(1, 2, 4, 8), scale=2):
        super().__init__()
        assert scale in (1, 2, 4), "scale must be 1, 2, or 4"
        self.scale = scale

        self.noise_estimator = NoiseEstimator()
        # input = image + noise map = 2 channels
        self.shallow = nn.Conv2d(2, dim, kernel_size=3, padding=1, bias=False)

        d1, d2, d3, d4 = dim, dim * 2, dim * 4, dim * 8

        self.enc1 = nn.Sequential(*[TransformerBlock(d1, heads[0]) for _ in range(num_blocks[0])])
        self.down1 = Downsample(d1)
        self.enc2 = nn.Sequential(*[TransformerBlock(d2, heads[1]) for _ in range(num_blocks[1])])
        self.down2 = Downsample(d2)
        self.enc3 = nn.Sequential(*[TransformerBlock(d3, heads[2]) for _ in range(num_blocks[2])])
        self.down3 = Downsample(d3)

        self.latent = nn.Sequential(*[TransformerBlock(d4, heads[3]) for _ in range(num_blocks[3])])

        self.up3 = Upsample(d4)
        self.fuse3 = nn.Conv2d(d4, d3, kernel_size=1, bias=False)
        self.dec3 = nn.Sequential(*[TransformerBlock(d3, heads[2]) for _ in range(num_blocks[2])])
        self.up2 = Upsample(d3)
        self.fuse2 = nn.Conv2d(d3, d2, kernel_size=1, bias=False)
        self.dec2 = nn.Sequential(*[TransformerBlock(d2, heads[1]) for _ in range(num_blocks[1])])
        self.up1 = Upsample(d2)
        self.fuse1 = nn.Conv2d(d2, d1, kernel_size=1, bias=False)
        self.dec1 = nn.Sequential(*[TransformerBlock(d1, heads[0]) for _ in range(num_blocks[0])])

        # Reconstruction head with sub-pixel upsampling [SwinIR / W2S]
        head = []
        s = scale
        while s > 1:
            head += [
                nn.Conv2d(d1, d1 * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            s //= 2
        self.upsampler = nn.Sequential(*head) if head else nn.Identity()
        self.output = nn.Conv2d(d1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        """x: (B, 1, H, W) raw degraded image (NOT clipped). Returns (B, 1, sH, sW)."""
        noise_map = self.noise_estimator(x)
        feat = self.shallow(torch.cat([x, noise_map], dim=1))
        shallow_feat = feat  # global long skip [SwinIR]

        e1 = self.enc1(feat)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        lat = self.latent(self.down3(e3))

        d3 = self.dec3(self.fuse3(torch.cat([self.up3(lat), e3], dim=1)))
        d2 = self.dec2(self.fuse2(torch.cat([self.up2(d3), e2], dim=1)))
        d1 = self.dec1(self.fuse1(torch.cat([self.up1(d2), e1], dim=1)))

        d1 = d1 + shallow_feat  # low frequencies bypass the deep body
        out = self.output(self.upsampler(d1))

        # residual around bicubic upsample of the input: model learns the correction
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False) \
            if self.scale > 1 else x
        return out + base


def build_model(scale=2, dim=32):
    return UBRN(dim=dim, scale=scale)


if __name__ == "__main__":
    m = build_model(scale=2)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"[v0] UBRN params: {n_params / 1e6:.2f}M")
    x = torch.randn(1, 1, 128, 128)
    y = m(x)
    print(f"[v0] in {tuple(x.shape)} -> out {tuple(y.shape)}")
