import numpy as np

from read_eareog.oculomotor import dprime, epoch_steps, gain_db, gains


def test_gains_recover_known_slopes():
    rng = np.random.default_rng(5)
    dx, dy = rng.uniform(-900, 300, 200), rng.uniform(-50, 50, 200)
    steps = 0.3 + 1.5 * dx / 100 - 0.2 * dy / 100 + rng.normal(0, 1e-6, 200)
    gH, gV, n = gains(steps, dx, dy)
    assert n == 200 and np.isclose(gH, 1.5, atol=1e-5) and np.isclose(gV, -0.2, atol=1e-4)
    assert np.isnan(gains(steps[:5], dx[:5], dy[:5])[0])


def test_epoch_steps_measure_a_clean_step_and_reject_blinks():
    fs = 500.0
    t = np.arange(0, 20, 1 / fs)
    x = np.where(t >= 10.04, 2.0, 0.0)
    st = epoch_steps(x, t, [10.0, 10.0], [0.04, 0.04], blinks=[], rsd=1.0)
    assert np.allclose(st, 2.0)
    assert np.isnan(epoch_steps(x, t, [10.0], [0.04], blinks=[10.3], rsd=1.0)[0])
    assert np.isnan(epoch_steps(x, t, [10.0], [0.04], blinks=[], rsd=0.01)[0])     # amplitude rejection


def test_dprime_and_db():
    assert dprime(-2.0, 500.0, 5.0) == 2.0
    assert np.isclose(gain_db(0.5, 1.0), -6.0206, atol=1e-4)


def test_quiet_times_and_step_noise():
    from read_eareog.oculomotor import quiet_times, step_noise
    q = quiet_times([10.0, 20.0], [1.0, 0.3], blinks=[10.6], lo=0.0, hi=100.0)
    assert q.size == 0          # every candidate of the first fixation is within 0.5 s of the blink; the second is too short
    q = quiet_times([10.0], [1.0], blinks=[], lo=0.0, hi=100.0)
    assert np.allclose(q, np.arange(10.25, 10.70, 0.02))
    fs = 500.0
    t = np.arange(0, 60, 1 / fs)
    x = np.random.default_rng(3).normal(size=t.size)
    sig = step_noise(x, t, fs, np.arange(5, 55, 0.5))
    assert 0.1 < sig < 0.3          # SD of a difference of two 100-sample means of unit noise is about 0.14
