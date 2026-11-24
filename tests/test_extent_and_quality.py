"""Pinning tests for the analysed extent, the channel-quality rule, the two-per-side rule and related helpers."""
from dataclasses import replace

import numpy as np
import pytest

from read_eareog import pipeline as PL
from read_eareog.derivations import orientation_from_gains
from read_eareog.oculomotor import gain_saccades
from read_eareog.recording import channel_qc, crop_to_window, flat_run_fraction
from read_eareog.synthetic import synthetic_recording


@pytest.fixture(scope='module')
def rec():
    return synthetic_recording(50, minutes=1.0)


def test_crop_uses_floor_and_ceil(rec):
    r = replace(rec, t=rec.t + 0.0011)                      # W limits fall between samples
    c = crop_to_window(r, 10.0)
    lo, hi = r.window
    pos = np.interp([lo, hi], r.t, np.arange(r.t.size))
    assert c.t[0] == r.t[int(np.floor(pos[0] - 5000))] and c.t[-1] == r.t[int(np.ceil(pos[1] + 5000)) - 1]
    with pytest.raises(ValueError, match='extend'):
        crop_to_window(replace(r, window=(r.t[0] + 5.0, r.window[1])), 10.0)


def test_quality_thresholds(rec):
    c = crop_to_window(rec)
    base = channel_qc(c)
    assert not any(base.values())
    i = c.ear_labels.index('EL3')
    for factor, flagged in ((19.0, False), (21.0, True), (1 / 19.0, False), (1 / 21.0, True)):
        ear = c.ear.copy()
        ear[i] *= factor
        assert channel_qc(replace(c, ear=ear))['EL3'] is flagged, factor


def test_flat_run_limit(rec):
    c = crop_to_window(rec)
    n = c.t.size - 2 * 5000                                  # samples over W
    i = c.ear_labels.index('ER2')
    for frac, flagged in ((0.009, False), (0.012, True)):
        ear = c.ear.copy()
        k = int(frac * n)
        ear[i, 6000:6000 + k] = 0.5
        assert channel_qc(replace(c, ear=ear))['ER2'] is flagged, frac
    assert flat_run_fraction(np.r_[np.zeros(250), np.arange(1.0, 751.0)]) == 0.25      # a run of exactly 250
    assert flat_run_fraction(np.r_[np.zeros(249), np.arange(1.0, 752.0)]) == 0.0       # 249: below the run length


def test_nan_in_flagged_channel_does_not_disable_qc(rec):
    c = crop_to_window(rec)
    ear = c.ear.copy()
    ear[c.ear_labels.index('Fpz'), 100] = np.nan
    ear[c.ear_labels.index('EL4')] *= 40.0
    flags = channel_qc(replace(c, ear=ear, bad={'Fpz': True}))
    assert flags['EL4'] and flags['Fpz']


def test_two_electrodes_per_ear(rec):
    c = crop_to_window(rec)
    one_left = {'EL1': True, 'EL3': True, 'EL4': True}
    two_left = {'EL1': True, 'EL3': True}
    assert PL.derivation(replace(c, bad=one_left)) is None and PL.causal_signal(replace(c, bad=one_left)) is None
    assert PL.derivation(replace(c, bad=two_left)) is not None


def test_false_alarms_per_minute_end_to_end():
    recs = [synthetic_recording(60 + i, minutes=2.0) for i in range(3)]
    res = PL.run_primary(recs)
    for r, row in zip(recs, res['recordings']):
        lab = PL.labels_of(r)
        from read_eareog.scoring import match
        m = match(row['detections'], lab)
        assert row['false_alarms_per_min'] == m.fp / ((r.window[1] - r.window[0]) / 60.0)
        assert r.minutes == 2.0


def test_orientation_sign_and_gain_saccades():
    o = orientation_from_gains({'a': -1.5, 'b': -1.2, 'c': 0.8})
    assert o == {'a': 1.0, 'b': 1.0, 'c': 1.0}             # others' median gain <= 0 -> leftward steps positive
    assert orientation_from_gains({'a': 1.0, 'b': 2.0, 'c': -1.0})['c'] == -1.0
    sel = gain_saccades([1, 6, 7, 8, 9], [-19.9, 25, -5, 3, 100], [3, 0, 20, 1, 0], 5, 8.5)
    assert list(sel) == [False, True, True, False, False]


def test_causal_scale_uses_calibration_period_only(rec):
    c = PL.prepare(rec)
    z1 = PL.causal_signal(c)
    ear = c.ear.copy()
    late = c.t > c.window[0] + 61.0
    ear[:, late] *= 3.0                                      # change the data after the calibration minute
    z2 = PL.causal_signal(replace(c, ear=ear))
    early = c.t <= c.window[0] + 60.0
    assert np.allclose(z1[early], z2[early])


def test_label_free_threshold():
    pytest.importorskip('sklearn')
    from read_eareog.baselines import label_free_threshold
    rng = np.random.default_rng(0)
    amp = np.r_[rng.normal(0.5, 0.2, 300), rng.normal(4.0, 0.3, 100)]
    tau, half = label_free_threshold(amp)
    assert 2.0 < tau < 2.5 and 1.5 < half < 2.0
    assert np.isnan(label_free_threshold(amp[:5])[0])
