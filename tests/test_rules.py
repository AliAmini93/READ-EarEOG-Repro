"""Targeted tests: each pins one rule of the article to a known value, so that a plausible change of the rule fails."""
import numpy as np
import pytest
import scipy.signal

from read_eareog import pipeline as PL
from read_eareog.baselines import random_f1
from read_eareog.config import Protocol
from read_eareog.labels import fixation_runs, line_classes
from read_eareog.oculomotor import epoch_steps, quiet_times
from read_eareog.scoring import false_alarms_per_minute, match
from read_eareog.signal import bandpass, causal_bandpass, robust_sd, scale_rsd


def test_false_alarms_are_per_minute():
    m = match([1.0, 5.0, 9.0], [1.05])
    assert false_alarms_per_minute(m, 2.0) == 1.0          # 2 false alarms in 2 minutes


def test_cohort_summaries_use_medians(cohort):
    res = PL.run_primary(cohort)
    rows = res['recordings']
    assert res['cohort']['false_alarms_per_min_median'] == round(float(np.median([r['false_alarms_per_min'] for r in rows])), 2)
    assert res['cohort']['theta_median'] == float(np.median([r['theta'] for r in rows]))
    assert res['cohort']['f1']['median'] == round(float(np.median([r['f1'] for r in rows])), 3)


def test_random_baseline_seed_and_spacing(cohort):
    res = PL.run_primary(cohort)
    P = Protocol()
    for k, (rec, row) in enumerate(zip(cohort, res['recordings'])):
        lab = PL.labels_of(rec)
        want = random_f1(np.random.default_rng(P.seed + 2000 + k), row['n_detections'], *rec.window, lab)
        assert row['f1_random'] == want
    # hard-core spacing: with n points filling the window at exactly 0.6 s, every draw is the same regular grid
    lab = np.arange(0.0, 6.0, 0.6)
    assert random_f1(np.random.default_rng(1), 10, 0.0, 5.4, lab, draws=5) == 1.0


def test_filter_order_and_scaling_band():
    x = np.random.default_rng(1).normal(size=20000)
    sos = scipy.signal.butter(4, [0.1, 15.0], btype='bandpass', fs=500, output='sos')
    assert np.array_equal(bandpass(x, 0.1, 15.0, 500), scipy.signal.sosfiltfilt(sos, x))
    core = slice(5000, 15000)
    assert scale_rsd(x, 500) == robust_sd(bandpass(x, 1.0, 20.0, 500)[core])


def test_causal_filter_starts_in_steady_state():
    x = np.full((1, 2000), 7.0)                                 # constant input: no transient if initialised
    assert np.max(np.abs(causal_bandpass(x, 0.1, 15.0, 500))) < 1e-9


def test_epoch_window_starts_50_ms_after_saccade_end():
    fs = 500.0
    t = np.arange(0, 20, 1 / fs)
    x = np.where(t >= 10.06, 2.0, 0.0)                           # step 10 ms after the window start
    st = epoch_steps(x, t, [10.0], [0.04], blinks=[], rsd=1.0)  # window [10.09, 10.29): fully after the step
    assert np.isclose(st[0], 2.0)
    x2 = np.where(t >= 10.10, 2.0, 0.0)                          # step inside the post window
    assert 0 < epoch_steps(x2, t, [10.0], [0.04], blinks=[], rsd=1.0)[0] < 2.0


def test_quiet_times_blink_radius():
    q = quiet_times([10.0], [2.0], blinks=[9.4], lo=0, hi=100)     # 0.6 s before the first candidate: kept
    assert q[0] == pytest.approx(10.25)
    q = quiet_times([10.0], [2.0], blinks=[9.8], lo=0, hi=100)     # within 0.5 s of 10.25 ... 10.30
    assert q[0] > 10.30 - 1e-9


def test_attribution_blink_radius():
    from read_eareog.synthetic import synthetic_recording
    rec = synthetic_recording(8, minutes=1.0)
    lo = rec.window[0]
    sac_free = [t for t in np.arange(lo + 5, lo + 55, 0.01)
                if np.min(np.abs(rec.saccades['onset'] - t)) > 0.25 and np.min(np.abs(rec.blinks - t)) > 0.5]
    t0 = sac_free[0]
    rec.blinks = np.sort(np.r_[rec.blinks, t0 + 0.29])
    assert PL.attribute_detections([t0], rec, 29.5)['BLINK_ONLY'] == 1
    rec.blinks = np.sort(np.r_[rec.blinks[rec.blinks != t0 + 0.29], t0 + 0.31])
    assert PL.attribute_detections([t0], rec, 29.5)['NOTHING'] == 1


def _page(post_y, landing_x, pre_end=9.995):
    fix = [(0.2 * i, 0.15, 100.0 + 60 * i, 900.0) for i in range(20)]      # x range 100 ... 1240
    fix += [(pre_end - 0.6, 0.2, 900, 300), (pre_end - 0.4, 0.2, 1000, 300), (pre_end - 0.2, 0.2, 1100, 300)]
    fix += [(10.05, 0.2, landing_x, post_y), (10.3, 0.2, landing_x + 60, post_y), (10.55, 0.2, landing_x + 120, post_y)]
    sac = [(10.0, 0.05, landing_x - 1100.0, post_y - 300.0)]
    fix.sort()
    F = {k: np.array([f[i] for f in fix]) for i, k in enumerate(('onset', 'duration', 'x', 'y'))}
    S = {k: np.array([s[i] for s in sac]) for i, k in enumerate(('onset', 'duration', 'dx', 'dy'))}
    return S, F


def test_line_class_thresholds():
    pitch = 30.0
    cls = lambda py, lx, **kw: line_classes(*_page(py, lx, **kw), 0.0, 30.0, pitch)['cls'][0]  # noqa: E731
    assert cls(300 + 1.0 * pitch, 200) == 'NL'
    assert cls(300 - 0.6 * pitch, 200) == 'UP'                 # u = -0.6: earlier line
    assert cls(300 - 0.4 * pitch, 200) == 'SAME'               # |u| < 0.5
    # landing fraction 0.35 of the 2nd-98th percentile range of fixation x (about 123 ... 1240)
    assert cls(300 + pitch, 480) == 'NL' and cls(300 + pitch, 560) == 'OTHERLEFT'


def test_ten_millisecond_tolerance():
    # the last fixation before the saccade ends 9 ms after its onset: still counted (launch x = 1100)
    S, F = _page(330.0, 200, pre_end=10.009)
    assert fixation_runs(F, S, [0])[3][0] == 1100.0
    # 11 ms after: not counted, the previous fixation becomes the launch fixation
    S, F = _page(330.0, 200, pre_end=10.011)
    assert fixation_runs(F, S, [0])[3][0] == 1000.0


def test_step_windows_rounded_at_125_hz():
    # A = [0.10, 0.30) s at 125 Hz: lags round(12.5) = 12 (96 ms) to round(37.5) = 38, i.e. last sample 37 (296 ms)
    from read_eareog.detector import step_statistic
    for lag, inside in ((12, True), (11, False), (37, True), (38, False)):
        z = np.zeros(1000)
        z[500 + lag] = 1.0
        assert bool(step_statistic(z, 125)[500] > 0) is inside, lag


def test_ocular_step_post_window_ends_250_ms_after_saccade_end():
    # onset 10.0 s, duration 0: post window [10.05, 10.25) s, pre window [9.75, 9.95) s
    fs = 500.0
    t = np.arange(0, 20, 1 / fs)
    after_end = epoch_steps(np.where(t >= 10.25, 1.0, 0.0), t, [10.0], [0.0], blinks=[], rsd=1e9)[0]
    inside = epoch_steps(np.where(t >= 10.20, 1.0, 0.0), t, [10.0], [0.0], blinks=[], rsd=1e9)[0]
    assert after_end == 0.0 and np.isclose(inside, 0.25)


def test_causal_window_lengths():
    from read_eareog.detector import causal_statistic
    z = np.zeros(2000)
    z[1000 - 75] = 1.0                       # 150 ms back at 500 Hz: just outside (t - 0.15 s, t]
    assert causal_statistic(z, 500)[1000] == 0.0
    z = np.zeros(2000)
    z[1000 - 74] = 1.0                       # inside: contributes 1/75 to the recent mean
    assert np.isclose(causal_statistic(z, 500)[1000], 1 / 75)


def test_causal_calibration_period_is_60_s():
    from dataclasses import replace
    from read_eareog.synthetic import synthetic_recording
    rec = PL.prepare(synthetic_recording(70, minutes=2.0))
    lo = rec.window[0]
    z1 = PL.causal_signal(rec)
    ear = rec.ear.copy()
    late = rec.t > lo + 60.0 + 0.002
    ear[:, late] *= 5.0
    z2 = PL.causal_signal(replace(rec, ear=ear))
    early = rec.t <= lo + 59.0
    assert np.allclose(z1[early], z2[early])          # data after 60 s do not affect the scale
    ear = rec.ear.copy()
    ear[:, (rec.t > lo + 59.0) & (rec.t <= lo + 60.0)] *= 50.0
    z3 = PL.causal_signal(replace(rec, ear=ear))
    assert not np.allclose(z1[rec.t <= lo + 50], z3[rec.t <= lo + 50])   # data up to 60 s do
