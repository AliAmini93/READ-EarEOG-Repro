import numpy as np

from read_eareog.detector import causal_statistic, detect, step_statistic, upward_crossings


def test_ideal_step_gives_step_height_at_onset():
    z = np.zeros(2000)
    z[1000:] = 3.0
    s = step_statistic(z, 500)
    assert s[1000] == 3.0                       # A = [t+50, t+150) samples, B = [t-125, t-25)
    assert np.isnan(s[:125]).all() and np.isnan(s[-150 + 1:]).all()


def test_windows_rounded_at_low_rates():
    # at 64 Hz the after window [0.10, 0.30) s becomes lags 6..18 (round(6.4) = 6, round(19.2) = 19)
    z = np.zeros(640)
    z[300 + 6] = 13.0
    s = step_statistic(z, 64)
    assert np.isclose(s[300], 1.0)
    z = np.zeros(640)
    z[300 + 19] = 13.0
    assert np.isclose(step_statistic(z, 64)[300], 0.0)


def test_detection_threshold_and_spacing():
    s = np.zeros(3000)
    s[[400, 700, 999, 1300]] = [2.3, 3.0, 2.9, 2.29999]
    mask = np.ones(3000, bool)
    assert list(detect(s, 2.3, mask, 500)) == [400, 700]     # >= theta; 999 is 299 samples after 700


def test_mask_excludes_peaks():
    s = np.zeros(1000)
    s[100] = 5.0
    mask = np.ones(1000, bool)
    mask[50:150] = False
    assert detect(s, 1.0, mask, 500).size == 0


def test_causal_statistic_uses_past_only():
    rng = np.random.default_rng(2)
    z = rng.normal(size=5000)
    a = causal_statistic(z, 500)
    z2 = z.copy()
    z2[3001:] += 100.0
    b = causal_statistic(z2, 500)
    assert np.array_equal(a[:3001], b[:3001], equal_nan=True)


def test_upward_crossings_refractory():
    s = np.zeros(2000)
    s[100:110] = 2.0
    s[300:310] = 2.0          # 0.4 s later: suppressed
    s[500:510] = 2.0          # 0.8 s after the first: kept
    assert list(upward_crossings(s, 1.0, np.ones(2000, bool), 500)) == [100, 500]
