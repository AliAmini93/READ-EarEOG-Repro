"""Eye-tracker labels (article Eqs. 1-2).

Inputs are the eye tracker's parsed events in seconds on the eye-tracker clock and screen pixels (x to the right,
y downwards):

* saccades: ``onset``, ``duration``, ``dx`` (end x - start x), ``dy`` (end y - start y)
* fixations: ``onset``, ``duration``, ``x``, ``y``
"""
from __future__ import annotations

import numpy as np

from .config import PROTOCOL, Protocol

CLASSES = ('NL', 'SKIP', 'SAME', 'UP', 'OTHERLEFT', 'PAGE', 'NONLEFT')
CLASS_NAMES = {'NL': 'next-line transition', 'SKIP': 'downward skip (>= 2 lines)',
               'SAME': 'same-line regression', 'UP': 'earlier-line regression', 'OTHERLEFT': 'other leftward saccade',
               'PAGE': 'page-level jump', 'NONLEFT': 'not leftward'}


def return_sweep_labels(onset, dx, dy, lo: float, hi: float, protocol: Protocol = PROTOCOL) -> np.ndarray:
    """Sorted onsets of saccades in W = [lo, hi] with dx < -300 px and |dy| < 150 px (Eq. 1)."""
    onset, dx, dy = (np.asarray(v, float) for v in (onset, dx, dy))
    inw = (onset >= lo) & (onset <= hi)
    return np.sort(onset[inw & (dx < protocol.dx_max_px) & (np.abs(dy) < protocol.abs_dy_max_px)])


def fixation_runs(fix: dict, sac: dict, index, k: int = 3, span: float = 1.5):
    """Median y of up to ``k`` fixations ending within ``span`` s before each saccade and of up to ``k`` fixations
    starting within ``span`` s after it (10-ms tolerance at the saccade boundaries); x of the first fixation after
    (landing) and of the last fixation before (launch). Returns four arrays (NaN where no fixation)."""
    on, end = fix['onset'], fix['onset'] + fix['duration']
    pre_y, post_y, land_x, launch_x = [], [], [], []
    for i in index:
        t0, t1 = sac['onset'][i], sac['onset'][i] + sac['duration'][i]
        b = np.flatnonzero((end <= t0 + 0.01) & (end >= t0 - span))[-k:]
        a = np.flatnonzero((on >= t1 - 0.01) & (on <= t1 + span))[:k]
        pre_y.append(np.median(fix['y'][b]) if b.size else np.nan)
        post_y.append(np.median(fix['y'][a]) if a.size else np.nan)
        land_x.append(fix['x'][a[0]] if a.size else np.nan)
        launch_x.append(fix['x'][b[-1]] if b.size else np.nan)
    return np.array(pre_y), np.array(post_y), np.array(land_x), np.array(launch_x)


def line_classes(sac: dict, fix: dict, lo: float, hi: float, pitch_px: float, protocol: Protocol = PROTOCOL) -> dict:
    """Fixation-defined line class of every saccade in W (Eq. 2).

    u = (median y after - median y before) / pitch, in lines. Leftward means dx < -100 px; landing on the left means
    the landing x <= x2 + 0.35 (x98 - x2), with x2, x98 the 2nd and 98th percentiles of fixation x in W.
    Page-level jumps (u <= -8 or dy < -400 px) take precedence and apply to any saccade.
    Returns a dict of arrays: onset, duration, dx, dy, u, cls, land_left, label (Eq. 1 criterion).
    """
    sac = {k: np.asarray(v, float) for k, v in sac.items()}
    fix = {k: np.asarray(v, float) for k, v in fix.items()}
    inw = np.flatnonzero((sac['onset'] >= lo) & (sac['onset'] <= hi) & np.isfinite(sac['dx']) & np.isfinite(sac['dy']))
    fw = (fix['onset'] >= lo) & (fix['onset'] <= hi)
    x02, x98 = np.percentile(fix['x'][fw], [2, 98])
    pre, post, landx, _ = fixation_runs(fix, sac, inw, k=protocol.fixations_per_side, span=protocol.fixation_span_s)
    u = (post - pre) / pitch_px
    dx, dy = sac['dx'][inw], sac['dy'][inw]
    left_land = landx <= x02 + protocol.left_landing_fraction * (x98 - x02)
    cls = np.full(inw.size, 'NONLEFT', dtype=object)
    lef = dx < -100
    fin = np.isfinite(u)
    page = (u <= -8) | (dy < -400)
    cls[lef & fin & (u >= 0.5) & (u < 1.5) & left_land] = 'NL'
    cls[lef & fin & (u >= 1.5) & (u < 8) & left_land] = 'SKIP'
    cls[lef & fin & (np.abs(u) < 0.5)] = 'SAME'
    cls[lef & fin & (u <= -0.5) & (u > -8)] = 'UP'
    cls[(cls == 'NONLEFT') & lef] = 'OTHERLEFT'
    cls[page] = 'PAGE'
    return {'onset': sac['onset'][inw], 'duration': sac['duration'][inw], 'dx': dx, 'dy': dy, 'u': u, 'cls': cls,
            'land_left': left_land,
            'label': (dx < protocol.dx_max_px) & (np.abs(dy) < protocol.abs_dy_max_px)}
