"""Cohort statistics (article Section "Statistics" and Supplement, formal definitions).

The unit of analysis is the recording. Bootstrap intervals are percentile intervals over recordings.
"""
from __future__ import annotations

import numpy as np

SEED = 20261121


def sign_flip_exact(d) -> float:
    """Exact two-sided sign-flip test of mean(d) = 0 by enumeration of all 2^n sign patterns (n <= 22).
    The smallest attainable p is 2^(1-n) (the observed pattern and its global reversal)."""
    d = np.asarray(d, float)
    n = d.size
    if n == 0:
        return np.nan
    if n > 22:
        raise ValueError('exact enumeration limited to n <= 22')
    ref = abs(d.mean()) - 1e-12                       # absolute tolerance for floating-point ties, as in the original
    count, chunk = 0, 1 << 16
    for start in range(0, 2 ** n, chunk):              # enumerate in chunks to bound memory (n = 22: 4.2 million patterns)
        idx = np.arange(start, min(start + chunk, 2 ** n))
        signs = ((idx[:, None] >> np.arange(n)) & 1).astype(np.int8) * 2 - 1
        count += int(np.sum(np.abs((signs @ d) / n) >= ref))
    return count / 2 ** n


def bootstrap_mean_ci(d, n_boot: int = 10000, seed: int = SEED):
    """95 % percentile bootstrap interval of the mean (resampling recordings)."""
    d = np.asarray(d, float)
    if d.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, d.size, (n_boot, d.size))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def bootstrap_median_ci(v, n_boot: int = 10000, seed: int = SEED) -> dict:
    """Median with 95 % percentile bootstrap interval (finite values only), rounded to 3 decimals."""
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if v.size < 2:
        return {'n': int(v.size)}
    rng = np.random.default_rng(seed)
    m = np.median(v[rng.integers(0, v.size, (n_boot, v.size))], axis=1)
    return {'n': int(v.size), 'median': round(float(np.median(v)), 3),
            'ci95': [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]}


def ratio_of_means_ci(num, den, n_boot: int = 10000, seed: int = SEED) -> dict:
    """mean(num) / mean(den) with a 95 % percentile bootstrap interval over recordings (pairs with both finite)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    ok = np.isfinite(num) & np.isfinite(den)
    num, den = num[ok], den[ok]
    if num.size < 2 or den.mean() == 0:
        return {'n': int(num.size)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, (n_boot, num.size))
    r = num[idx].mean(1) / den[idx].mean(1)
    return {'n': int(num.size), 'ratio': round(float(num.mean() / den.mean()), 3),
            'ci95': [round(float(np.nanpercentile(r, 2.5)), 3), round(float(np.nanpercentile(r, 97.5)), 3)]}


def paired_summary(d, n_boot: int = 10000, seed: int = SEED) -> dict:
    """Cohort summary of per-recording values (e.g. paired differences): mean, 95 % bootstrap CI, Cohen's d_z
    (SD with n - 1), exact sign-flip p and the number of recordings with the majority sign (rounded as reported)."""
    d = np.asarray([x for x in d if np.isfinite(x)], float)
    if d.size == 0:
        return {'n': 0}
    sgn = np.sign(d.mean())
    lo, hi = bootstrap_mean_ci(d, n_boot, seed)
    return {'n': int(d.size), 'mean': round(float(d.mean()), 3), 'ci95': [round(lo, 3), round(hi, 3)],
            'd_z': round(float(d.mean() / d.std(ddof=1)), 3) if d.size > 1 and d.std(ddof=1) > 0 else None,
            'p_exact_signflip': round(sign_flip_exact(d), 6),
            'n_majority_sign': int((np.sign(d) == sgn).sum()),
            'majority_sign': 'positive' if sgn > 0 else 'negative'}


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p values: p_H(r) = max over k <= r of min(1, (m - k + 1) p_(k))."""
    ps = np.asarray(pvalues, float)
    order = np.argsort(ps)
    adj, run = np.empty(ps.size), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (ps.size - r) * ps[i]))
        adj[i] = run
    return adj
