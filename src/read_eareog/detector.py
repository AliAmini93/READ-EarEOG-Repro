"""Step statistic and detections (article Eqs. 5-7 and the causal variant, Eq. 11).

``z`` is the scaled derivation (Eq. 4) sampled at ``fs``; ``t`` holds the eye-tracker-clock time of every sample.
"""
from __future__ import annotations

import numpy as np
import scipy.signal

from .config import PROTOCOL, Protocol


def _lags(window_s, fs: float):
    """Window limits in samples, rounded to the nearest sample (ties to even, Python ``round``)."""
    return int(round(window_s[0] * fs)), int(round(window_s[1] * fs))


def step_statistic(z, fs: float, after_s=(0.10, 0.30), before_s=(-0.25, -0.05)) -> np.ndarray:
    """s(t) = mean z over [t + a0, t + a1) - mean z over [t + b0, t + b1) for every sample (Eq. 6).

    Window limits that fall between samples are rounded to the nearest sample; where a window leaves the recording
    the value is NaN. At 500 Hz the windows are exactly 100 samples each.
    """
    z = np.asarray(z, float)
    n = z.size
    c = np.concatenate([[0.0], np.cumsum(z)])
    t = np.arange(n)

    def wmean(a, b):
        lo, hi = t + a, t + b
        ok = (lo >= 0) & (hi <= n)
        out = np.full(n, np.nan)
        out[ok] = (c[hi[ok]] - c[lo[ok]]) / max(b - a, 1)
        return out
    return wmean(*_lags(after_s, fs)) - wmean(*_lags(before_s, fs))


def detect(s, theta: float, mask, fs: float, refractory_s: float = 0.6) -> np.ndarray:
    """Sample indices of the detections (Eq. 7): local maxima of s inside the mask with s >= theta; when two are
    closer than int(refractory_s * fs) samples, the larger is kept (scipy ``find_peaks`` priority rule)."""
    s = np.asarray(s, float)
    s2 = np.where(np.asarray(mask, bool) & np.isfinite(s), s, -np.inf)
    peaks, _ = scipy.signal.find_peaks(s2, height=theta, distance=max(1, int(refractory_s * fs)))
    return peaks


def detection_times(z, t, mask, fs: float, theta: float, protocol: Protocol = PROTOCOL) -> np.ndarray:
    """Detection times (eye-tracker clock) of the scaled derivation ``z`` at threshold ``theta``."""
    s = step_statistic(z, fs, protocol.after_window_s, protocol.before_window_s)
    return np.asarray(t, float)[detect(s, theta, mask, fs, protocol.refractory_s)]


def causal_statistic(z, fs: float, after_s=(-0.15, 0.0), before_s=(-0.60, -0.40)) -> np.ndarray:
    """Past-only step statistic (Eq. 11): mean z over (t - 0.15 s, t] minus mean over (t - 0.60 s, t - 0.40 s].

    Window limits are truncated to whole samples (``int``), as in the original analysis; values before the first
    complete window are NaN.
    """
    z = np.asarray(z, float)
    n = z.size
    c = np.concatenate([[0.0], np.cumsum(z)])
    t = np.arange(n)

    def wm(a, b):
        lo, hi = t + a, t + b
        ok = lo >= 0
        out = np.full(n, np.nan)
        out[ok] = (c[hi[ok] + 1] - c[lo[ok] + 1]) / (b - a) if b - a > 0 else np.nan
        return out
    return wm(-int(-after_s[0] * fs), int(after_s[1] * fs)) - wm(-int(-before_s[0] * fs), -int(-before_s[1] * fs))


def upward_crossings(s, theta: float, mask, fs: float, refractory_s: float = 0.6) -> np.ndarray:
    """Causal detections: samples where s first reaches theta from below (both samples inside the mask), keeping a
    crossing only if it is at least refractory_s after the previously kept one."""
    s = np.asarray(s, float)
    ok = np.asarray(mask, bool) & np.isfinite(s)
    up = np.flatnonzero(ok[1:] & ok[:-1] & (s[:-1] < theta) & (s[1:] >= theta)) + 1
    kept, last = [], -1e18
    for i in up:
        if i - last >= refractory_s * fs:
            kept.append(i)
            last = i
    return np.array(kept, int)
