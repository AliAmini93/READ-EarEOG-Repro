import numpy as np
import pytest

from read_eareog.recording import Recording, fit_clock, load_recording, sample_times, save_recording
from read_eareog.synthetic import synthetic_recording


def test_roundtrip(tmp_path):
    rec = synthetic_recording(2, minutes=1.0)
    save_recording(rec, tmp_path / 'r')
    back = load_recording(tmp_path / 'r')
    assert np.array_equal(back.ear, rec.ear) and back.ear_labels == rec.ear_labels and back.window == rec.window
    assert np.allclose(back.saccades['dx'], rec.saccades['dx']) and np.allclose(back.fixations['y'], rec.fixations['y'])


def test_validation():
    rec = synthetic_recording(3, minutes=1.0)
    with pytest.raises(ValueError):
        Recording(name='x', ear=rec.ear, ear_labels=rec.ear_labels[:-1], fs=rec.fs, t=rec.t, window=rec.window,
                  saccades=rec.saccades, fixations=rec.fixations, blinks=rec.blinks)
    with pytest.raises(ValueError):
        Recording(name='x', ear=rec.ear, ear_labels=rec.ear_labels, fs=rec.fs, t=rec.t, window=(0, 1e6),
                  saccades=rec.saccades, fixations=rec.fixations, blinks=rec.blinks)


def test_clock_fit():
    ticks = np.array([100, 5100, 10100, 20100])
    times = 3.0 + ticks / 500.002
    slope, intercept, resid = fit_clock(ticks, times)
    assert resid < 1e-9 and np.allclose(sample_times(3, slope, intercept), 3.0 + np.arange(3) / 500.002)


def test_synthetic_flag_survives_reload(tmp_path):
    rec = synthetic_recording(4, minutes=1.0)
    save_recording(rec, tmp_path / 's')
    assert load_recording(tmp_path / 's').synthetic


def test_rejects_non_finite_and_wrong_rate():
    rec = synthetic_recording(5, minutes=1.0)
    ear = rec.ear.copy()
    ear[0, 100] = np.nan
    kw = dict(name='x', ear_labels=rec.ear_labels, t=rec.t, window=rec.window, saccades=rec.saccades,
              fixations=rec.fixations, blinks=rec.blinks)
    with pytest.raises(ValueError, match='non-finite'):
        Recording(ear=ear, fs=rec.fs, **kw)
    Recording(ear=ear, fs=rec.fs, bad={rec.ear_labels[0]: True}, **kw)      # allowed when flagged bad
    with pytest.raises(ValueError, match='does not match fs'):
        Recording(ear=rec.ear, fs=250.0, **kw)


def test_events_are_sorted_and_labels_decoded():
    rec = synthetic_recording(6, minutes=1.0)
    order = np.random.default_rng(0).permutation(rec.fixations['onset'].size)
    shuffled = {k: v[order] for k, v in rec.fixations.items()}
    r2 = Recording(name='y', ear=rec.ear, ear_labels=[x.encode() for x in rec.ear_labels], fs=rec.fs, t=rec.t,
                   window=rec.window, saccades=rec.saccades, fixations=shuffled, blinks=rec.blinks)
    assert np.array_equal(r2.fixations['onset'], rec.fixations['onset']) and r2.ear_labels == rec.ear_labels
