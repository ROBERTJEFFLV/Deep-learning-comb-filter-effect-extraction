"""Lightweight stpACC extraction for mono audio."""

from __future__ import annotations

import numpy as np
from scipy import signal


def compute_stpacc_frame(
    window: np.ndarray,
    downsample_bins: int,
) -> np.ndarray:
    power_spec = np.abs(np.fft.rfft(window)) ** 2
    acc = np.fft.irfft(power_spec)
    acc = acc[: len(acc) // 2]
    acc = np.clip(acc, 1e-12, None)
    acc = acc / max(1e-12, float(np.max(np.abs(acc))))
    stp = np.convolve(acc**2, np.hanning(8), mode="same")
    stp = stp / max(1e-12, float(np.max(np.abs(stp))))
    if len(stp) == downsample_bins:
        return stp.astype(np.float32)
    return signal.resample(stp, num=downsample_bins).astype(np.float32)

