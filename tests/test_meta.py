"""Packaging and consistency checks (no installation is performed)."""
import inspect
import tomllib
import zipfile
from pathlib import Path

import pytest

import read_eareog
from read_eareog import baselines, detector, falsealarms, labels, oculomotor, scoring
from read_eareog.config import Protocol

ROOT = Path(__file__).resolve().parents[1]


def test_version_single_source():
    meta = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    assert meta['project']['version'] == read_eareog.__version__


def test_function_defaults_equal_protocol():
    """Functions called with their defaults use exactly the article's parameters."""
    P = Protocol()

    def d(f, name):
        return inspect.signature(f).parameters[name].default
    assert d(detector.step_statistic, 'after_s') == P.after_window_s
    assert d(detector.step_statistic, 'before_s') == P.before_window_s
    assert d(detector.detect, 'refractory_s') == P.refractory_s
    assert d(detector.causal_statistic, 'after_s') == P.causal_after_window_s
    assert d(detector.causal_statistic, 'before_s') == P.causal_before_window_s
    assert d(scoring.match, 'tolerance') == P.tolerance_s
    assert (d(scoring.match_asymmetric, 'before'), d(scoring.match_asymmetric, 'after')) == (P.match_before_s,
                                                                                              P.match_after_s)
    assert d(baselines.velocity_candidates, 'merge_gap_s') == P.merge_gap_s
    assert d(baselines.velocity_candidates, 'before_s') == P.before_amplitude_s
    assert d(baselines.velocity_candidates, 'after_s') == P.after_amplitude_s
    assert d(baselines.random_f1, 'draws') == P.random_draws
    assert d(falsealarms.eligible_null_times, 'draws') == P.null_draws
    assert d(oculomotor.epoch_steps, 'reject') == P.epoch_reject_rsd
    assert d(oculomotor.epoch_steps, 'blink_exclusion_s') == P.blink_exclusion_s
    assert d(oculomotor.quiet_times, 'every') == P.quiet_step_s
    assert d(labels.fixation_runs, 'k') == P.fixations_per_side
    assert d(labels.fixation_runs, 'span') == P.fixation_span_s


def test_wheel_contents(tmp_path):
    build_meta = pytest.importorskip('setuptools.build_meta')
    import os
    import shutil
    import subprocess
    # build from a copy of the TRACKED files only, so that no untracked data (e.g. recordings placed in the checkout)
    # is ever copied, and the checkout stays clean
    try:
        tracked = subprocess.run(['git', 'ls-files'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('git is needed to list the tracked files')
    src = tmp_path / 'src_copy'
    for rel in tracked:
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, src / rel)
    cwd = os.getcwd()
    try:
        os.chdir(src)
        name = build_meta.build_wheel(str(tmp_path / 'dist'))
    finally:
        os.chdir(cwd)
    names = zipfile.ZipFile(tmp_path / 'dist' / name).namelist()
    assert 'read_eareog/pipeline.py' in names and 'read_eareog/__main__.py' in names
    assert not [n for n in names if n.startswith('read_eareog/data/') or n.endswith('figures.py')]   # code only
