"""Chance levels and conventional detectors (article Section "Chance levels and conventional detectors").

Random generators are passed in explicitly (``numpy.random.Generator``) so that every baseline is reproducible.
"""
from __future__ import annotations

import bisect

import numpy as np

from .scoring import match, match_asymmetric
from .signal import robust_sd


def random_f1(rng, n_detections: int, lo: float, hi: float, labels, refractory_s: float = 0.6,
              draws: int = 100, tolerance: float = 0.2) -> float:
    """Rate-matched random baseline: mean F1 of ``n_detections`` times drawn uniformly in [lo, hi] subject to the
    same minimum spacing (hard-core process), over ``draws`` draws. NaN if impossible."""
    span = (hi - lo) - (n_detections - 1) * refractory_s
    if n_detections == 0 or span <= 0:
        return np.nan
    vals = []
    for _ in range(draws):
        u = np.sort(rng.uniform(0, span, n_detections)) + np.arange(n_detections) * refractory_s + lo
        vals.append(match(u, labels, tolerance).f1)
    return float(np.mean(vals))


def random_f1_asymmetric(rng, n_detections: int, lo: float, hi: float, labels, refractory_s: float = 0.6,
                         draws: int = 100, before: float = 0.1, after: float = 0.5) -> float:
    """Rate-matched random baseline for the causal variant's asymmetric matching window."""
    span = (hi - lo) - (n_detections - 1) * refractory_s
    if n_detections == 0 or span <= 0:
        return np.nan
    return float(np.mean([match_asymmetric(np.sort(rng.uniform(0, span, n_detections))
                                           + np.arange(n_detections) * refractory_s + lo, labels, before, after)[2]
                          for _ in range(draws)]))


def periodic_f1(rng, n_detections: int, lo: float, hi: float, labels, draws: int = 100, tolerance: float = 0.2) -> float:
    """Periodic baseline: ``n_detections`` equally spaced times with a random phase."""
    if n_detections == 0:
        return np.nan
    step = (hi - lo) / n_detections
    return float(np.mean([match(lo + rng.uniform(0, step) + np.arange(n_detections) * step, labels, tolerance).f1
                          for _ in range(draws)]))


def velocity_candidates(z, t, mask, fs: float, lam: float, merge_gap_s: float = 0.03,
                        before_s=(-0.15, -0.05), after_s=(0.05, 0.15)):
    """Velocity-threshold candidates: segments where the derivative of z exceeds lam x its robust SD over the mask,
    merged if separated by less than merge_gap_s; amplitude = mean z 50-150 ms after the segment end minus mean z
    150-50 ms before its start; candidate time = segment start. Returns (times, amplitudes)."""
    z = np.asarray(z, float)
    mask = np.asarray(mask, bool)
    v = np.gradient(z) * fs
    on = v > lam * robust_sd(v[mask])
    idx = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    segs = list(zip(idx[0::2], idx[1::2] - 1))
    merged = []
    for s0, s1 in segs:
        if merged and s0 - merged[-1][1] < int(merge_gap_s * fs):
            merged[-1] = (merged[-1][0], s1)
        else:
            merged.append((s0, s1))
    times, amp = [], []
    for s0, s1 in merged:
        a0, a1 = s0 + int(before_s[0] * fs), s0 + int(before_s[1] * fs)
        b0, b1 = s1 + int(after_s[0] * fs), s1 + int(after_s[1] * fs)
        if a0 < 0 or b1 > z.size or not mask[s0]:
            continue
        times.append(t[s0])
        amp.append(z[b0:b1].mean() - z[a0:a1].mean())
    return np.array(times), np.array(amp)


def refractory_select(times, score, threshold: float, refractory_s: float = 0.6, top: int | None = None) -> np.ndarray:
    """Greedy selection by decreasing score: keep candidates with score >= threshold (or the ``top`` best) that are
    at least refractory_s from every kept candidate. Returns sorted kept times."""
    times, score = np.asarray(times, float), np.asarray(score, float)
    order = np.argsort(-score, kind='stable')
    acc = []
    for i in order:
        if top is None and score[i] < threshold:
            break
        ti = times[i]
        j = bisect.bisect_left(acc, ti)
        if (j < len(acc) and acc[j] - ti < refractory_s) or (j > 0 and ti - acc[j - 1] < refractory_s):
            continue
        acc.insert(j, ti)
        if top is not None and len(acc) >= top:
            break
    return np.array(acc, float)


def label_free_threshold(amplitudes, seed: int = 20261121):
    """Label-free amplitude threshold adapted from a two-component Gaussian mixture (Kunze et al.-style):
    tau = mu_low + 0.5 (mu_high - mu_low); also returns the sensitivity variant 0.5 (mu_high - mu_low).

    Requires scikit-learn (optional dependency)."""
    from sklearn.mixture import GaussianMixture
    a = np.asarray(amplitudes, float)
    a = a[np.isfinite(a)]
    if a.size < 10:
        return np.nan, np.nan
    gm = GaussianMixture(2, n_init=5, random_state=seed).fit(a.reshape(-1, 1))
    lo_, hi_ = np.sort(gm.means_.ravel())
    return float(lo_ + 0.5 * (hi_ - lo_)), float(0.5 * (hi_ - lo_))
