import numpy as np

from read_eareog.falsealarms import eligible_null_times, graded_response, min_dist, window_max


def test_null_times_respect_exclusions():
    rng = np.random.default_rng(7)
    labels = np.arange(5, 100, 3.0)
    tps = labels[::2] + 0.05
    draws = eligible_null_times(rng, 20, 0, 100, labels, tps)
    assert len(draws) == 20
    for d in draws:
        assert d.size <= 20
        assert np.all(min_dist(d, labels) > 0.2) and np.all(min_dist(d, tps) >= 0.6)
        assert np.all(np.diff(np.sort(d)) >= 0.6 - 1e-9)


def test_window_is_half_open():
    t = np.arange(0, 10, 0.125)                 # dyadic grid: exact window edges
    s = np.zeros_like(t)
    s[t == 5.125] = 9.0
    assert window_max(s, t, [5.0], 0.125)[0] == 0.0      # 5.125 is the excluded right edge
    assert window_max(s, t, [5.25], 0.125)[0] == 9.0
    assert np.isnan(graded_response(s, t, []))


def test_binary_response():
    from read_eareog.falsealarms import binary_response
    t = np.arange(0, 10, 0.125)
    s = np.zeros_like(t)
    s[t == 5.0] = 2.0
    assert binary_response(s, t, [5.0, 8.0], theta=1.0) == 0.5
