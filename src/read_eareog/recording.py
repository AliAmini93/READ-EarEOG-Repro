"""Input contract for one recording, a loader for a documented directory format and clock alignment.

The package never downloads or ships recordings. A user who has legitimately obtained a dataset converts each
recording into the format below (see docs/DATA_ACCESS.md) and loads it with :func:`load_recording`.

Directory format (one directory per recording; all times in seconds on the eye-tracker clock):

    ear.npz         data   float array (channels x samples), ear-recorder signals in recorder units
                    labels str array (channels,), channel labels (e.g. EL1 ... ER4, M1, M2, Fpz)
                    fs     scalar, sampling rate in Hz
                    t      float array (samples,), eye-tracker-clock time of every ear sample
    saccades.csv    header onset,duration,start_x,start_y,end_x,end_y  (screen px, y downwards)
    fixations.csv   header onset,duration,x,y
    blinks.csv      header onset
    meta.json       {"window": [lo, hi], "bad_channels": [...], "synthetic": false}
                    (W = first task marker + 2 s to last task marker - 2 s on the eye-tracker clock, see
                    docs/DATA_ACCESS.md; "synthetic" marks generated example data)

Event tables are sorted by onset when a Recording is created (the line classes assume time order).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Recording:
    name: str                 # an anonymous identifier chosen by the user, e.g. "rec01"
    ear: np.ndarray           # (channels, samples)
    ear_labels: list
    fs: float
    t: np.ndarray             # eye-tracker-clock time of every ear sample
    window: tuple             # analysis window W = (lo, hi)
    saccades: dict            # onset, duration, dx, dy (arrays)
    fixations: dict           # onset, duration, x, y (arrays)
    blinks: np.ndarray        # blink onsets
    bad: dict = field(default_factory=dict)
    synthetic: bool = False   # True for generated data (never a real recording)

    def __post_init__(self):
        self.ear = np.atleast_2d(np.asarray(self.ear, float))
        self.ear_labels = [x.decode() if isinstance(x, bytes) else str(x) for x in self.ear_labels]
        self.t = np.asarray(self.t, float)
        self.blinks = np.sort(np.asarray(self.blinks, float))
        self.saccades = _sorted_table({k: np.asarray(v, float) for k, v in self.saccades.items()}, self.name, 'saccades')
        self.fixations = _sorted_table({k: np.asarray(v, float) for k, v in self.fixations.items()}, self.name, 'fixations')
        self.validate()

    def validate(self):
        n_ch, n = self.ear.shape
        if len(self.ear_labels) != n_ch:
            raise ValueError(f'{self.name}: {len(self.ear_labels)} labels for {n_ch} channels')
        if self.t.shape != (n,):
            raise ValueError(f'{self.name}: t must have one time per ear sample')
        if not np.all(np.diff(self.t) > 0):
            raise ValueError(f'{self.name}: t must be strictly increasing')
        if self.fs <= 0:
            raise ValueError(f'{self.name}: fs must be positive')
        step = float(np.median(np.diff(self.t)))
        if abs(step * self.fs - 1.0) > 0.01:
            raise ValueError(f'{self.name}: median sample spacing of t ({step:.6g} s) does not match fs = {self.fs:g} Hz')
        bad_values = [lab for lab, row in zip(self.ear_labels, self.ear)
                      if not self.bad.get(lab, False) and not np.all(np.isfinite(row))]
        if bad_values:
            raise ValueError(f'{self.name}: non-finite samples in channel(s) {bad_values}; '
                             'flag them as bad or repair them before the analysis')
        lo, hi = self.window
        if not (self.t[0] <= lo < hi <= self.t[-1]):
            raise ValueError(f'{self.name}: window W must lie inside the ear recording')
        for key in ('onset', 'duration', 'dx', 'dy'):
            if key not in self.saccades:
                raise ValueError(f'{self.name}: saccades need {key}')
        for key in ('onset', 'duration', 'x', 'y'):
            if key not in self.fixations:
                raise ValueError(f'{self.name}: fixations need {key}')

    @property
    def mask(self) -> np.ndarray:
        """Ear samples inside W."""
        return (self.t >= self.window[0]) & (self.t <= self.window[1])

    @property
    def minutes(self) -> float:
        return (self.window[1] - self.window[0]) / 60.0


def _sorted_table(table: dict, name: str, what: str) -> dict:
    lengths = {k: v.shape for k, v in table.items()}
    if len(set(lengths.values())) > 1 or any(len(s) != 1 for s in lengths.values()):
        raise ValueError(f'{name}: {what} columns must be 1-D and of equal length ({lengths})')
    if 'onset' in table and table['onset'].size:
        order = np.argsort(table['onset'], kind='stable')
        table = {k: v[order] for k, v in table.items()}
    return table


def _read_csv(path: Path, columns) -> dict:
    with open(path, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for c in columns:
        if rows and c not in rows[0]:
            raise ValueError(f'{path.name}: missing column {c}')
        out[c] = np.array([float(r[c]) if r[c] not in ('', 'nan', 'NaN') else np.nan for r in rows], float)
    return out


def load_recording(directory: str | Path, name: str | None = None) -> Recording:
    """Load one recording from the documented directory format."""
    d = Path(directory)
    with np.load(d / 'ear.npz', allow_pickle=False) as z:
        data, labels, fs, t = z['data'], [str(x) for x in z['labels']], float(z['fs']), z['t']
    s = _read_csv(d / 'saccades.csv', ('onset', 'duration', 'start_x', 'start_y', 'end_x', 'end_y'))
    f = _read_csv(d / 'fixations.csv', ('onset', 'duration', 'x', 'y'))
    b = _read_csv(d / 'blinks.csv', ('onset',))['onset']
    meta = json.loads((d / 'meta.json').read_text(encoding='utf-8'))
    sac = {'onset': s['onset'], 'duration': s['duration'], 'dx': s['end_x'] - s['start_x'], 'dy': s['end_y'] - s['start_y']}
    return Recording(name=name or d.name, ear=data, ear_labels=labels, fs=fs, t=t, window=tuple(meta['window']),
                     saccades=sac, fixations=f, blinks=b, bad={c: True for c in meta.get('bad_channels', [])},
                     synthetic=bool(meta.get('synthetic', False)))


def save_recording(rec: Recording, directory: str | Path) -> None:
    """Write a Recording in the documented directory format (used for the synthetic example)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / 'ear.npz', data=rec.ear, labels=np.array(rec.ear_labels), fs=rec.fs, t=rec.t)
    s = rec.saccades
    with open(d / 'saccades.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['onset', 'duration', 'start_x', 'start_y', 'end_x', 'end_y'])
        for row in zip(s['onset'], s['duration'], np.zeros_like(s['dx']), np.zeros_like(s['dy']), s['dx'], s['dy']):
            w.writerow([repr(float(v)) for v in row])
    with open(d / 'fixations.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['onset', 'duration', 'x', 'y'])
        for row in zip(*(rec.fixations[k] for k in ('onset', 'duration', 'x', 'y'))):
            w.writerow([repr(float(v)) for v in row])
    with open(d / 'blinks.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['onset'])
        for v in rec.blinks:
            w.writerow([repr(float(v))])
    meta = {'window': list(rec.window), 'bad_channels': [c for c, v in rec.bad.items() if v],
            'synthetic': bool(rec.synthetic)}
    (d / 'meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')


def crop_to_window(rec: Recording, pad_s: float = 10.0) -> Recording:
    """The analysed extent: W extended by ``pad_s`` seconds on each side, as in the article (filters and the robust SD
    for scaling are computed on this extent; the robust SD over its central part, i.e. over W).

    The first sample is floor(s(lo) - pad_s fs) and the end is ceil(s(hi) + pad_s fs), where s(.) is the fractional
    sample position of an eye-tracker time. Raises ValueError if the recording does not cover this extent."""
    from dataclasses import replace
    lo, hi = rec.window
    pos = np.arange(rec.t.size, dtype=float)
    s_lo, s_hi = np.interp(lo, rec.t, pos), np.interp(hi, rec.t, pos)
    s0 = int(np.floor(s_lo - pad_s * rec.fs))
    s1 = int(np.ceil(s_hi + pad_s * rec.fs))
    if s0 < 0 or s1 > rec.t.size:
        raise ValueError(f'{rec.name}: the recording must extend at least {pad_s:g} s beyond W on both sides')
    return replace(rec, ear=rec.ear[:, s0:s1], t=rec.t[s0:s1])


def flat_run_fraction(x, run: int = 250) -> float:
    """Fraction of samples lying in runs of at least ``run`` identical consecutive values."""
    x = np.asarray(x)
    if x.size == 0:
        return 1.0
    change = np.flatnonzero(np.diff(x) != 0)
    edges = np.concatenate([[-1], change, [x.size - 1]])
    lens = np.diff(edges)
    return float(lens[lens >= run].sum() / x.size)


def channel_qc(rec: Recording, pad_s: float = 10.0, rel: float = 20.0, max_flat: float = 0.01) -> dict:
    """Channel quality rule of the article on a cropped recording: a channel is unusable if the robust SD of its 1-20 Hz
    band (over W) lies outside 1/rel to rel times the median over channels, or if more than ``max_flat`` of its samples
    lie in runs of at least 250 identical values. Returns {label: True if unusable}."""
    from .signal import MAD_TO_SD, bandpass, core_slice
    core = core_slice(rec.t.size, rec.fs, pad_s)
    b = bandpass(rec.ear, 1.0, 20.0, rec.fs)[:, core]
    rsd = MAD_TO_SD * np.median(np.abs(b - np.median(b, axis=1, keepdims=True)), axis=1)
    flat = np.array([flat_run_fraction(rec.ear[i, core]) for i in range(rec.ear.shape[0])])
    med = float(np.median(rsd[np.isfinite(rsd)]))          # finite channels only (identical when all are finite)
    return {c: bool(not np.isfinite(rsd[i]) or rsd[i] < med / rel or rsd[i] > rel * med or flat[i] > max_flat) for i, c in enumerate(rec.ear_labels)}


def fit_clock(device_tick_samples, tracker_tick_times):
    """Least-squares linear map from device sample index to eye-tracker time using shared timing events (e.g.
    audio ticks recorded by both systems). Returns (slope, intercept, max abs residual in s)."""
    x = np.asarray(device_tick_samples, float)
    y = np.asarray(tracker_tick_times, float)
    A = np.column_stack([x, np.ones_like(x)])
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - (slope * x + intercept)
    return float(slope), float(intercept), float(np.max(np.abs(resid)))


def sample_times(n_samples: int, slope: float, intercept: float) -> np.ndarray:
    """Eye-tracker-clock time of every device sample from a :func:`fit_clock` map."""
    return slope * np.arange(n_samples) + intercept
