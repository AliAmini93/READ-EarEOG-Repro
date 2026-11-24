"""Filtering and amplitude scaling (article Eqs. 3-4).

All functions are pure: they take arrays and return new arrays.
"""
from __future__ import annotations

import numpy as np
import scipy.signal

MAD_TO_SD = 1.4826   # robust SD = 1.4826 x median absolute deviation (consistent for a normal distribution)


def bandpass(x, lo_hz: float, hi_hz: float, fs: float, order: int = 4, axis: int = -1) -> np.ndarray:
    """Zero-phase Butterworth band-pass (forward-backward, second-order sections)."""
    sos = scipy.signal.butter(order, [lo_hz, hi_hz], btype='bandpass', fs=fs, output='sos')
    return scipy.signal.sosfiltfilt(sos, np.asarray(x, float), axis=axis)


def causal_bandpass(x, lo_hz: float, hi_hz: float, fs: float, order: int = 4) -> np.ndarray:
    """Causal (forward-only) Butterworth band-pass of each row, initialised at the first sample's steady state."""
    x = np.atleast_2d(np.asarray(x, float))
    sos = scipy.signal.butter(order, [lo_hz, hi_hz], btype='bandpass', fs=fs, output='sos')
    zi = scipy.signal.sosfilt_zi(sos)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i], _ = scipy.signal.sosfilt(sos, x[i], zi=zi * x[i, 0])
    return out


def robust_sd(x) -> float:
    """1.4826 x median(|x - median(x)|)."""
    x = np.asarray(x, float)
    return float(MAD_TO_SD * np.median(np.abs(x - np.median(x))))


def core_slice(n: int, fs: float, pad_s: float) -> slice:
    """Samples more than ``pad_s`` seconds from either end of a recording of ``n`` samples."""
    return slice(int(pad_s * fs), n - int(pad_s * fs))


def scale_rsd(x, fs: float, band_hz=(1.0, 20.0), pad_s: float = 10.0, order: int = 4) -> float:
    """Robust SD used for scaling (Eq. 4): robust SD of the ``band_hz`` band-pass of ``x`` over the core samples."""
    x = np.asarray(x, float)
    b = bandpass(x, band_hz[0], band_hz[1], fs, order)[core_slice(x.size, fs, pad_s)]
    return robust_sd(b)


def _nonzero(rsd: float) -> float:
    if not np.isfinite(rsd) or rsd <= 0:
        raise ValueError(f'cannot scale: robust SD is {rsd} (flat or non-finite signal)')
    return rsd


def scale(x, fs: float, band_hz=(1.0, 20.0), pad_s: float = 10.0, order: int = 4) -> np.ndarray:
    """z = x / rSD (Eq. 4); raises ValueError for a flat or non-finite signal."""
    return np.asarray(x, float) / _nonzero(scale_rsd(x, fs, band_hz, pad_s, order))
