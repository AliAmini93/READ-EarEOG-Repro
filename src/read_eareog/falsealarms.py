"""False-alarm null times and concordance with a second derivation (article Supplement, formal definitions).

A false-alarm-eligible null time is a time that could have been a false alarm: farther than 0.2 s from every label
and at least 0.6 s from every true-positive detection. Concordance compares a second derivation's response at the
false alarms with its response at null times and at true positives:

    kappa = mean_i(R_FA,i - R_null,i) / mean_i(R_TP,i - R_null,i)

(ratio of cohort means; 0 = null level, 1 = true-positive level).
"""
from __future__ import annotations

import bisect

import numpy as np


def min_dist(times, events) -> np.ndarray:
    """Distance from each time to the nearest event (inf if there are no events)."""
    t = np.atleast_1d(np.asarray(times, float))
    ev = np.sort(np.asarray(events, float))
    if ev.size == 0:
        return np.full(t.size, np.inf)
    j = np.searchsorted(ev, t)
    left = ev[np.clip(j - 1, 0, ev.size - 1)]
    right = ev[np.clip(j, 0, ev.size - 1)]
    return np.minimum(np.abs(t - left), np.abs(t - right))


def eligible_null_times(rng, n: int, lo: float, hi: float, labels, true_positives, label_radius: float = 0.2,
                        tp_radius: float = 0.6, draws: int = 20, grid: float = 0.002, refractory_s: float = 0.6):
    """``draws`` sets of up to ``n`` null times each, drawn without replacement from a 2-ms grid on [lo, hi]
    restricted to distance > label_radius from labels and >= tp_radius from true positives, with the detector's
    minimum spacing between times of the same draw."""
    cand = np.arange(lo, hi, grid)
    ok = (min_dist(cand, labels) > label_radius) & (min_dist(cand, true_positives) >= tp_radius)
    pool = cand[ok]
    out = []
    if n <= 0 or pool.size == 0:
        return out
    for _ in range(draws):
        acc = []
        for i in rng.permutation(pool.size):
            v = pool[i]
            k = bisect.bisect_left(acc, v)
            if (k < len(acc) and acc[k] - v < refractory_s) or (k > 0 and v - acc[k - 1] < refractory_s):
                continue
            acc.insert(k, v)
            if len(acc) >= n:
                break
        out.append(np.array(acc, float))
    return out


def window_max(s, t, times, half: float = 0.1) -> np.ndarray:
    """Maximum of s over samples with t in [time - half, time + half) for each time (NaN if none is finite)."""
    s, t = np.asarray(s, float), np.asarray(t, float)
    out = np.full(len(times), np.nan)
    for k, v in enumerate(times):
        i0, i1 = np.searchsorted(t, v - half), np.searchsorted(t, v + half)
        seg = s[i0:i1]
        if seg.size and np.isfinite(seg).any():
            out[k] = float(np.nanmax(seg))
    return out


def graded_response(s, t, times, half: float = 0.1) -> float:
    """Graded response of a second derivation: median over events of the window maximum of its step statistic."""
    if len(times) == 0:
        return np.nan
    v = window_max(s, t, times, half)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan


def binary_response(s, t, times, theta: float, half: float = 0.1) -> float:
    """Binary response: share of events at which the second derivation's statistic reaches its own threshold within
    [time - half, time + half) (no peak picking or thinning)."""
    if len(times) == 0:
        return np.nan
    return float(np.mean(window_max(s, t, times, half) >= theta))
