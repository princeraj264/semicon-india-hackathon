"""
Composite restoration loss (training only -- never imported by run.py):

    L = L_Charbonnier + 0.2 * L_SSIM + 0.1 * L_LPIPS + 0.05 * L_FFT

  - Charbonnier: robust L1 variant (SwinIR Eq. 7) -- stable early training.
  - SSIM loss: directly optimizes an official metric.
  - LPIPS: perceptual metric officially scored by KLA. Optional import; if the
    lpips package (and its local weights) are unavailable, weight is set to 0.
  - FFT L1: frequency-domain loss for periodic semiconductor structures
    (KLA's frequency-domain hint).

NOTE (offline rule): LPIPS/VGG weights must be bundled locally for training.
run.py never touches this module, so inference stays dependency-light.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


def _gaussian_window(size: int, sigma: float, device, dtype):
    ax = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)


class SSIMLoss(nn.Module):
    """1 - SSIM, computed with an 11x11 Gaussian window (grayscale)."""

    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma

    def forward(self, pred, target):
        w = _gaussian_window(self.window_size, self.sigma, pred.device, pred.dtype)
        pad = self.window_size // 2
        mu_p = F.conv2d(pred, w, padding=pad)
        mu_t = F.conv2d(target, w, padding=pad)
        mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
        sig_p = F.conv2d(pred * pred, w, padding=pad) - mu_p2
        sig_t = F.conv2d(target * target, w, padding=pad) - mu_t2
        sig_pt = F.conv2d(pred * target, w, padding=pad) - mu_pt
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu_pt + c1) * (2 * sig_pt + c2)) / (
            (mu_p2 + mu_t2 + c1) * (sig_p + sig_t + c2)
        )
        return 1.0 - ssim.mean()


class FFTLoss(nn.Module):
    """L1 in the frequency domain -- preserves structural periodicities."""

    def forward(self, pred, target):
        pf = torch.fft.rfft2(pred, norm="ortho")
        tf = torch.fft.rfft2(target, norm="ortho")
        return F.l1_loss(torch.view_as_real(pf), torch.view_as_real(tf))


class CompositeLoss(nn.Module):
    def __init__(self, w_char=1.0, w_ssim=0.2, w_lpips=0.1, w_fft=0.05, use_lpips=True):
        super().__init__()
        self.char = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.fft = FFTLoss()
        self.w_char, self.w_ssim, self.w_fft = w_char, w_ssim, w_fft
        self.w_lpips = 0.0
        self.lpips = None
        if use_lpips and w_lpips > 0:
            try:
                import lpips  # optional; needs local weights for offline use

                self.lpips = lpips.LPIPS(net="alex", verbose=False)
                for p in self.lpips.parameters():
                    p.requires_grad_(False)
                self.w_lpips = w_lpips
            except Exception as e:  # noqa: BLE001
                print(f"[losses] LPIPS unavailable ({e}); continuing without it.")

    def forward(self, pred, target):
        loss = self.w_char * self.char(pred, target)
        loss = loss + self.w_ssim * self.ssim(pred, target)
        loss = loss + self.w_fft * self.fft(pred, target)
        if self.lpips is not None:
            # LPIPS expects 3-channel in [-1, 1]
            p3 = pred.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
            t3 = target.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
            loss = loss + self.w_lpips * self.lpips(p3, t3).mean()
        return loss


# ---------------------------- metrics (validation) ----------------------------
@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1)).item()
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


@torch.no_grad()
def ssim_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    return 1.0 - SSIMLoss()(pred.clamp(0, 1), target.clamp(0, 1)).item()
