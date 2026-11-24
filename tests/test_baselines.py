import numpy as np

from read_eareog.baselines import (periodic_f1, random_f1, random_f1_asymmetric, refractory_select,
                                   velocity_candidates)
from read_eareog.pipeline import pooled_share_ci
from read_eareog.threshold import loro_joint


def test_chance_baselines_are_low_and_seeded():
    labels = np.arange(10, 290, 3.0)
    for f in (random_f1, periodic_f1, random_f1_asymmetric):
        a = f(np.random.default_rng(1), 90, 5, 295, labels)
        assert a == f(np.random.default_rng(1), 90, 5, 295, labels) and 0 < a < 0.5
    assert np.isnan(random_f1(np.random.default_rng(1), 0, 5, 295, labels))
    assert np.isnan(random_f1(np.random.default_rng(1), 1000, 5, 295, labels))     # cannot place 1000 at 0.6 s


def test_velocity_candidates_find_steps():
    fs = 500
    z = np.zeros(60 * fs)
    for s in (10, 20, 30):
        z[s * fs:] += 4.0
    z += np.random.default_rng(2).normal(0, 0.05, z.size)
    t = np.arange(z.size) / fs
    times, amp = velocity_candidates(z, t, np.ones(z.size, bool), fs, lam=5.0)
    big = times[amp > 2]
    assert np.allclose(big, [10, 20, 30], atol=0.02)


def test_refractory_select():
    t = np.array([1.0, 1.3, 2.0, 5.0])
    s = np.array([1.0, 3.0, 2.0, 0.5])
    assert list(refractory_select(t, s, 0.8)) == [1.3, 2.0]
    assert list(refractory_select(t, s, 0.0, top=1)) == [1.3]


def test_joint_loro_and_pooled_share():
    grid = [(1, 0.5), (2, 0.5)]
    curves = {'a': np.array([1.0, 0.0]), 'b': np.array([0.0, 1.0]), 'c': np.array([0.0, 1.0])}
    # a sees mean(b, c) = [0, 1]; b and c see [0.5, 0.5] and take the first maximum
    assert loro_joint(curves, grid) == {'a': (2, 0.5), 'b': (1, 0.5), 'c': (1, 0.5)}
    share, ci = pooled_share_ci([1, 2, 3], [10, 10, 10], n_boot=200)
    assert share == 0.2 and ci[0] <= 0.2 <= ci[1]
