"""Saccade-locked steps, horizontal and vertical gains and single-event d' (article Eqs. 5 and 10).

Gains and step noise are computed on the filtered but UNSCALED derivation (recorder units), so that derivations of
the same recorder share one scale; d' does not depend on this choice. Gains of different recorders are compared only
through d'.
"""
from __future__ import annotations

import numpy as np

from .detector import step_statistic
from .signal import MAD_TO_SD


def epoch_steps(x, t, onsets, durations, blinks, rsd: float, reject: float = 40.0,
                blink_exclusion_s=(-0.5, 0.7)) -> np.ndarray:
    """Ocular step of every saccade (Eq. 5): mean x over [on + dur + 0.05, on + dur + 0.25) minus mean x over
    [on - 0.25, on - 0.05) (window limits located on the sample-time axis t). NaN (rejected) if the epoch
    [on - 0.5, on + 0.7] leaves the recording, contains a blink onset, or |x - pre-saccade mean| exceeds
    reject x rsd anywhere in it."""
    x, t = np.asarray(x, float), np.asarray(t, float)
    c = np.concatenate([[0.0], np.cumsum(x)])

    def idx(v):
        return np.searchsorted(t, v)

    def wmean(a, b):
        i0, i1 = idx(a), idx(b)
        return (c[i1] - c[i0]) / np.maximum(i1 - i0, 1)
    on, du = np.asarray(onsets, float), np.asarray(durations, float)
    post = wmean(on + du + 0.05, on + du + 0.25)
    pre = wmean(on - 0.25, on - 0.05)
    st = post - pre
    blinks = np.asarray(blinks, float)
    bl = np.sort(blinks[np.isfinite(blinks)]) if blinks.size else np.array([])
    e0, e1 = blink_exclusion_s
    for k in range(on.size):
        i0, i1 = idx(on[k] + e0), idx(on[k] + e1)
        if i0 <= 0 or i1 >= x.size:
            st[k] = np.nan
            continue
        if bl.size and np.any((bl >= on[k] + e0) & (bl <= on[k] + e1)):
            st[k] = np.nan
            continue
        if np.max(np.abs(x[i0:i1] - pre[k])) > reject * rsd:
            st[k] = np.nan
    return st


def gain_saccades(onset, dx, dy, lo: float, hi: float, min_px: float = 20.0) -> np.ndarray:
    """Saccades entering the gain regression: onset in W and max(|dx|, |dy|) >= min_px (boolean mask)."""
    onset, dx, dy = (np.asarray(v, float) for v in (onset, dx, dy))
    return (onset >= lo) & (onset <= hi) & (np.maximum(np.abs(dx), np.abs(dy)) >= min_px)


def gains(steps, dx, dy, min_epochs: int = 10):
    """Ordinary least squares step = a + g_H dx/100 + g_V dy/100 over finite steps (Eq. 10).
    Returns (g_H, g_V, n); gains are per 100 px. NaN gains if fewer than ``min_epochs`` epochs."""
    steps, dx, dy = (np.asarray(v, float) for v in (steps, dx, dy))
    ok = np.isfinite(steps) & np.isfinite(dx) & np.isfinite(dy)
    if ok.sum() < min_epochs:
        return np.nan, np.nan, int(ok.sum())
    X = np.column_stack([np.ones(ok.sum()), dx[ok] / 100.0, dy[ok] / 100.0])
    beta, *_ = np.linalg.lstsq(X, steps[ok], rcond=None)
    return float(beta[1]), float(beta[2]), int(ok.sum())


def quiet_times(fix_onset, fix_duration, blinks, lo: float, hi: float, every: float = 0.02,
                blink_radius: float = 0.5) -> np.ndarray:
    """Times every ``every`` s at which [t - 0.25, t + 0.30] lies inside one fixation within W and no blink onset is
    within +-blink_radius s (used for the step noise)."""
    blinks = np.asarray(blinks, float)
    bl = np.sort(blinks[np.isfinite(blinks)]) if blinks.size else np.array([])
    out = []
    for o, d in zip(fix_onset, fix_duration):
        a, b = max(o + 0.25, lo), min(o + d - 0.30, hi)
        if b <= a:
            continue
        ts = np.arange(a, b, every)
        if bl.size and ts.size:
            ts = ts[~np.array([np.any(np.abs(bl - v) <= blink_radius) for v in ts])]
        out.append(ts)
    return np.concatenate(out) if out else np.array([])


def step_noise(x, t, fs: float, quiet, min_samples: int = 21) -> float:
    """Robust SD of the step statistic of the unscaled derivation at the quiet fixation times (sigma_s)."""
    s = step_statistic(np.asarray(x, float), fs)
    qi = np.clip(np.searchsorted(np.asarray(t, float), quiet), 0, s.size - 1)
    qs = s[qi]
    qs = qs[np.isfinite(qs)]
    return float(MAD_TO_SD * np.median(np.abs(qs - np.median(qs)))) if qs.size >= min_samples else np.nan


def dprime(gain_per_100px: float, displacement_px: float, sigma: float) -> float:
    """Single-event d' = |gain x displacement / 100| / sigma_s (Eq. 10)."""
    return abs(gain_per_100px * displacement_px / 100) / sigma if sigma else np.nan


def gain_db(gain, reference_gain) -> float:
    """Gain relative to a reference derivation of the same recorder, in dB: 20 log10(|gain| / |reference|)."""
    return float(20 * np.log10(abs(gain) / abs(reference_gain)))
