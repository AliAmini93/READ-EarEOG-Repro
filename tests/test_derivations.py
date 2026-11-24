import numpy as np

from read_eareog.config import Protocol
from read_eareog.derivations import ear_derivations, leave_one_out_sign

P = Protocol()
LABELS = list(P.left_ear) + list(P.right_ear) + ['M1', 'M2', 'Fpz']


def test_classes_and_identities():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(len(LABELS), 500))
    D = ear_derivations(X, LABELS)
    L, R = X[:4].mean(0), X[4:8].mean(0)
    m1, m2 = X[8], X[9]
    assert np.allclose(D['cross_head'][0][1], L - R)
    assert len(D['single_cross_pairs']) == 16 and len(D['within_ear_pairs']) == 12
    assert np.allclose(D['two_earpieces_ear_mastoid'][0][1], (L - R) - (m1 - m2))
    assert np.allclose(D['forehead_mastoids'][0][1], X[10] - (m1 + m2) / 2)


def test_bad_channels_dropped():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(len(LABELS), 100))
    D = ear_derivations(X, LABELS, {'EL3': True, 'M1': True})
    assert np.allclose(D['cross_head'][0][1], X[[0, 2, 3]].mean(0) - X[4:8].mean(0))
    assert D['mastoid_pair'][0][1] is None and len(D['single_cross_pairs']) == 12


def test_orientation_from_others_only():
    s = leave_one_out_sign({'a': -5.0, 'b': 1.0, 'c': 1.0})
    assert s == {'a': 1.0, 'b': -1.0, 'c': -1.0}          # b and c see a median of -2.0
    assert leave_one_out_sign({'a': 1.0, 'b': -1.0, 'c': -1.0})['a'] == -1.0
