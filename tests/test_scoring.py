import numpy as np

from read_eareog.scoring import event_hits, match, match_asymmetric


def test_tolerance_is_inclusive():
    m = match([0.2, 1.2000001], [0.0, 1.0])
    assert m.tp == 1 and m.fp == 1 and m.fn == 1


def test_one_to_one():
    m = match([1.0, 1.05], [1.02])
    assert m.tp == 1 and m.fp == 1 and m.fn == 0
    assert np.isclose(m.f1, 2 / 3)


def test_f1_definition():
    m = match([1, 2, 3, 10], [1.1, 2.1, 5, 6, 7])
    assert m.tp == 2 and np.isclose(m.f1, 2 * 2 / (2 * 2 + 2 + 3))


def test_empty_inputs():
    assert np.isnan(match([], []).f1)
    assert match([1.0], []).f1 == 0.0


def test_asymmetric_window():
    p, r, f1, lat = match_asymmetric([1.4], [1.0])       # label 0.4 s before detection: inside [d-0.5, d+0.1]
    assert f1 == 1.0 and np.isclose(lat[0], 0.4)
    assert match_asymmetric([1.0], [1.2])[2] == 0.0       # label 0.2 s after detection: outside


def test_event_hits_not_one_to_one():
    assert list(event_hits([1.0], [0.9, 1.1, 2.0])) == [True, True, False]
