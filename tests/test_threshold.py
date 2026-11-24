import numpy as np

from read_eareog.threshold import loro_thresholds


def test_own_curve_is_left_out():
    th = np.array([1.0, 2.0, 3.0])
    curves = {'a': np.array([0.0, 0.0, 1.0]), 'b': np.array([0.9, 0.1, 0.0]), 'c': np.array([0.8, 0.2, 0.0])}
    # a sees mean(b, c) = [0.85, 0.15, 0.0]; b sees mean(a, c) = [0.4, 0.1, 0.5]; c sees mean(a, b) = [0.45, 0.05, 0.5]
    assert loro_thresholds(curves, th) == {'a': 1.0, 'b': 3.0, 'c': 3.0}   # b's own curve would have chosen 1.0


def test_first_maximum_and_nan():
    th = np.array([1.0, 2.0])
    out = loro_thresholds({'a': np.array([0.5, 0.5]), 'b': np.array([np.nan, 0.5]), 'c': np.array([0.5, 0.5])}, th)
    assert out['b'] == 1.0
