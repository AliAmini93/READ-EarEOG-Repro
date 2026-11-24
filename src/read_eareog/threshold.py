"""Leave-one-recording-out threshold selection (article Eq. 9)."""
from __future__ import annotations

import numpy as np

from .detector import detect
from .scoring import match


def f1_curve(s, t, mask, labels, fs: float, thetas, tolerance: float = 0.2, refractory_s: float = 0.6) -> np.ndarray:
    """F1 of one recording at every threshold of the grid."""
    t = np.asarray(t, float)
    return np.array([match(t[detect(s, th, mask, fs, refractory_s)], labels, tolerance).f1 for th in thetas])


def loro_thresholds(curves: dict, thetas) -> dict:
    """For each recording, the grid threshold maximising the mean F1 of the OTHER recordings (first maximum;
    NaN entries ignored). ``curves`` maps recording -> F1 array over ``thetas``."""
    out = {}
    for k in curves:
        others = [curves[q] for q in curves if q != k]
        out[k] = float(thetas[int(np.nanargmax(np.nanmean(others, 0)))])
    return out


def loro_joint(curves: dict, grid: list) -> dict:
    """Leave-one-recording-out selection over an arbitrary parameter grid (list of parameter tuples)."""
    out = {}
    for k in curves:
        out[k] = grid[int(np.nanargmax(np.nanmean([curves[q] for q in curves if q != k], 0)))]
    return out
