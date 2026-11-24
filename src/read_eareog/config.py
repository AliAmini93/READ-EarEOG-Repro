"""Analysis parameters as an immutable object.

The built-in defaults are the pre-specified values of the article; ``configs/protocol.toml`` in the repository states
the same values (a test asserts that they are identical). ``Protocol.load(path)`` reads a TOML file with that layout:
missing keys fall back to the defaults, unknown keys and values of the wrong type or length raise an error, so that a
misspelt or malformed parameter cannot pass silently. ``Protocol.load()`` without a path returns the defaults, so the
package also works when installed outside a repository checkout.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np

REPO_CONFIG = Path(__file__).resolve().parents[2] / 'configs' / 'protocol.toml'   # present in a checkout only


@dataclass(frozen=True)
class Protocol:
    # electrodes
    left_ear: tuple = ('EL1', 'EL3', 'EL4', 'EL5')
    right_ear: tuple = ('ER1', 'ER2', 'ER3', 'ER4')
    mastoid_left: str = 'M1'
    mastoid_right: str = 'M2'
    forehead: str = 'Fpz'
    # filter and scaling
    band_hz: tuple = (0.1, 15.0)
    order: int = 4
    scale_band_hz: tuple = (1.0, 20.0)
    pad_s: float = 10.0
    # labels
    dx_max_px: float = -300.0
    abs_dy_max_px: float = 150.0
    # detector
    after_window_s: tuple = (0.10, 0.30)
    before_window_s: tuple = (-0.25, -0.05)
    refractory_s: float = 0.6
    theta_min: float = 0.5
    theta_max: float = 6.0
    theta_step: float = 0.1
    # scoring
    tolerance_s: float = 0.2
    # causal variant
    calibration_s: float = 60.0
    causal_after_window_s: tuple = (-0.15, 0.0)
    causal_before_window_s: tuple = (-0.60, -0.40)
    match_before_s: float = 0.1
    match_after_s: float = 0.5
    # velocity baseline
    lambda_grid: tuple = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
    merge_gap_s: float = 0.03
    before_amplitude_s: tuple = (-0.15, -0.05)
    after_amplitude_s: tuple = (0.05, 0.15)
    # oculomotor quantities
    epoch_reject_rsd: float = 40.0
    blink_exclusion_s: tuple = (-0.5, 0.7)
    line_pitch_px: float = 29.5
    quiet_step_s: float = 0.02
    # line classes
    fixations_per_side: int = 3
    fixation_span_s: float = 1.5
    left_landing_fraction: float = 0.35
    # statistics
    seed: int = 20261121
    n_boot: int = 10000
    random_draws: int = 100
    null_draws: int = 20
    source: str = field(default='built-in defaults (the article values)', compare=False)

    @property
    def thetas(self) -> np.ndarray:
        """Threshold grid theta_min, theta_min + step, ..., theta_max (rounded to 2 decimals)."""
        return np.round(np.arange(self.theta_min, self.theta_max + 1e-4, self.theta_step), 2)

    # TOML section/key -> attribute
    _MAP = {('electrodes', 'left_ear'): 'left_ear', ('electrodes', 'right_ear'): 'right_ear',
            ('electrodes', 'mastoid_left'): 'mastoid_left', ('electrodes', 'mastoid_right'): 'mastoid_right',
            ('electrodes', 'forehead'): 'forehead',
            ('filter', 'band_hz'): 'band_hz', ('filter', 'order'): 'order', ('filter', 'scale_band_hz'): 'scale_band_hz',
            ('filter', 'pad_s'): 'pad_s',
            ('labels', 'dx_max_px'): 'dx_max_px', ('labels', 'abs_dy_max_px'): 'abs_dy_max_px',
            ('detector', 'after_window_s'): 'after_window_s', ('detector', 'before_window_s'): 'before_window_s',
            ('detector', 'refractory_s'): 'refractory_s', ('detector', 'theta_min'): 'theta_min',
            ('detector', 'theta_max'): 'theta_max', ('detector', 'theta_step'): 'theta_step',
            ('scoring', 'tolerance_s'): 'tolerance_s',
            ('causal', 'calibration_s'): 'calibration_s', ('causal', 'after_window_s'): 'causal_after_window_s',
            ('causal', 'before_window_s'): 'causal_before_window_s', ('causal', 'match_before_s'): 'match_before_s',
            ('causal', 'match_after_s'): 'match_after_s',
            ('velocity_baseline', 'lambda_grid'): 'lambda_grid', ('velocity_baseline', 'merge_gap_s'): 'merge_gap_s',
            ('velocity_baseline', 'before_amplitude_s'): 'before_amplitude_s',
            ('velocity_baseline', 'after_amplitude_s'): 'after_amplitude_s',
            ('oculomotor', 'epoch_reject_rsd'): 'epoch_reject_rsd', ('oculomotor', 'blink_exclusion_s'): 'blink_exclusion_s',
            ('oculomotor', 'line_pitch_px'): 'line_pitch_px', ('oculomotor', 'quiet_step_s'): 'quiet_step_s',
            ('line_classes', 'fixations_per_side'): 'fixations_per_side',
            ('line_classes', 'fixation_span_s'): 'fixation_span_s',
            ('line_classes', 'left_landing_fraction'): 'left_landing_fraction',
            ('statistics', 'seed'): 'seed', ('statistics', 'n_boot'): 'n_boot',
            ('statistics', 'random_draws'): 'random_draws', ('statistics', 'null_draws'): 'null_draws'}

    @classmethod
    def load(cls, path: str | Path | None = None) -> 'Protocol':
        """Read a protocol TOML file; without a path, return the built-in (article) defaults."""
        if path is None:
            return cls()
        path = Path(path)
        with open(path, 'rb') as fh:
            doc = tomllib.load(fh)
        kw = {}
        for section, table in doc.items():
            if not isinstance(table, dict):
                raise ValueError(f'unexpected top-level key {section!r} in {path.name}')
            for key, value in table.items():
                attr = cls._MAP.get((section, key))
                if attr is None:
                    raise ValueError(f'unknown parameter [{section}] {key} in {path.name}')
                kw[attr] = _checked(attr, value, getattr(cls(), attr), path.name)
        return replace(cls(), source=str(path.name), **kw)

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _checked(attr, value, default, fname):
    """Value converted to the default's type; raises ValueError on a wrong type or length."""
    def num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if isinstance(default, tuple):
        if not isinstance(value, list) or not value:
            raise ValueError(f'{fname}: {attr} must be a non-empty list')
        if all(isinstance(d, str) for d in default):
            ok = all(isinstance(v, str) for v in value)
        else:
            ok = all(num(v) for v in value) and (attr == 'lambda_grid' or len(value) == len(default))
        if not ok:
            raise ValueError(f'{fname}: {attr} must be a list like {list(default)}')
        return tuple(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ValueError(f'{fname}: {attr} must be a string')
        return value
    if isinstance(default, int):
        if not (isinstance(value, int) and not isinstance(value, bool)):
            raise ValueError(f'{fname}: {attr} must be an integer')
        return value
    if not num(value):
        raise ValueError(f'{fname}: {attr} must be a number')
    return float(value)


PROTOCOL = Protocol()
