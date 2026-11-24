import numpy as np
import pytest

from read_eareog.config import REPO_CONFIG, Protocol


def test_file_equals_builtin_defaults():
    assert Protocol.load(REPO_CONFIG) == Protocol() == Protocol.load()


def test_threshold_grid():
    th = Protocol().thetas
    assert th.size == 56 and th[0] == 0.5 and th[-1] == 6.0 and np.allclose(np.diff(th), 0.1)


def test_unknown_parameter_rejected(tmp_path):
    p = tmp_path / 'bad.toml'
    p.write_text('[detector]\nrefractory = 0.5\n')
    with pytest.raises(ValueError, match='unknown parameter'):
        Protocol.load(p)


def test_override(tmp_path):
    p = tmp_path / 'p.toml'
    p.write_text('[scoring]\ntolerance_s = 0.1\n')
    assert Protocol.load(p).tolerance_s == 0.1


def test_type_and_length_checks(tmp_path):
    p = tmp_path / 't.toml'
    for body, msg in (('[detector]\ntheta_min = "0.5"\n', 'number'), ('[detector]\nafter_window_s = [0.1]\n', 'list'),
                      ('[statistics]\nseed = 1.5\n', 'integer'), ('[electrodes]\nleft_ear = [1, 2]\n', 'list')):
        p.write_text(body)
        with pytest.raises(ValueError, match=msg):
            Protocol.load(p)
