import numpy as np

from read_eareog.signal import bandpass, robust_sd, scale, scale_rsd


def test_bandpass_removes_offset():
    x = 5.0 + np.sin(2 * np.pi * 3 * np.arange(20000) / 500)
    y = bandpass(x, 0.1, 15.0, 500)
    assert abs(np.mean(y[2000:-2000])) < 0.05


def test_robust_sd_of_normal_is_about_one():
    rng = np.random.default_rng(0)
    assert abs(robust_sd(rng.normal(size=200000)) - 1.0) < 0.01


def test_scaled_signal_has_unit_rsd():
    rng = np.random.default_rng(1)
    x = rng.normal(size=60 * 500) * 7.0
    z = scale(x, 500)
    assert abs(scale_rsd(z, 500) - 1.0) < 1e-12


def test_flat_signal_cannot_be_scaled():
    import pytest
    with pytest.raises(ValueError, match='flat'):
        scale(np.zeros(30000), 500)


def test_causal_bandpass_is_causal():
    from read_eareog.signal import causal_bandpass
    rng = np.random.default_rng(4)
    x = rng.normal(size=(1, 5000))
    y = x.copy()
    y[0, 3000:] += 50.0
    assert np.array_equal(causal_bandpass(x, 0.1, 15, 500)[0, :3000], causal_bandpass(y, 0.1, 15, 500)[0, :3000])
