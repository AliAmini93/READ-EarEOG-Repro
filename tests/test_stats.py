import numpy as np

from read_eareog.stats import bootstrap_mean_ci, holm, paired_summary, ratio_of_means_ci, sign_flip_exact


def test_sign_flip_floor():
    for n in (5, 10, 18):
        assert np.isclose(sign_flip_exact(np.ones(n)), 2.0 ** (1 - n))


def test_sign_flip_symmetric_null():
    assert sign_flip_exact(np.array([1.0, -1.0, 2.0, -2.0])) == 1.0


def test_holm_definition():
    rng = np.random.default_rng(6)
    p = rng.uniform(0, 0.2, 24)
    srt = np.sort(p)
    want = [max(min(1.0, (24 - k) * srt[k]) for k in range(r + 1)) for r in range(24)]
    assert np.allclose(holm(p)[np.argsort(p)], want)


def test_bootstrap_reproducible():
    d = np.linspace(-1, 2, 15)
    assert bootstrap_mean_ci(d) == bootstrap_mean_ci(d)
    lo, hi = bootstrap_mean_ci(d)
    assert lo < d.mean() < hi


def test_summaries():
    s = paired_summary(np.array([0.1, 0.2, 0.3, -0.05]))
    assert s['n'] == 4 and s['majority_sign'] == 'positive' and s['n_majority_sign'] == 3
    r = ratio_of_means_ci([1, 2, 3], [2, 4, 6])
    assert r['ratio'] == 0.5
