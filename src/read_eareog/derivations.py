"""Ear-recorder derivations (article Eq. 3 and Section "Ear derivations").

A derivation is the mean of a set of "positive" channels minus the mean of a set of "negative" channels of the
band-passed ear recording. Channels flagged as bad are dropped; a derivation whose positive side, or non-empty
negative side, has no usable channel is ``None``.

Emulated two-earpiece combinations are re-combinations of the same common-reference recording; they are not
measurements with separate earpieces.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .config import PROTOCOL, Protocol


def combine(data, labels, bad, positive, negative):
    """mean(data[positive]) - mean(data[negative]) over usable channels (``None`` if a needed side is empty)."""
    labels = list(labels)
    ip = [labels.index(c) for c in positive if c in labels and not bad.get(c, False)]
    ineg = [labels.index(c) for c in negative if c in labels and not bad.get(c, False)]
    if not ip or (negative and not ineg):
        return None
    data = np.asarray(data, float)
    return data[ip].mean(0) - (data[ineg].mean(0) if ineg else 0.0)


def ear_derivations(data, labels, bad=None, protocol: Protocol = PROTOCOL) -> dict:
    """All pre-declared ear derivations, grouped by class (unoriented).

    Returns a dict ``class name -> list of (member name, signal or None)``:

    ``cross_head``            mean(left ear) - mean(right ear) (the primary derivation)
    ``single_cross_pairs``    each left electrode minus each right electrode (16 with four per side)
    ``within_ear_pairs``      every pair of electrodes within the same ear (one side only)
    ``ear_to_own_mastoid``    mean(left ear) - M1 and M2 - mean(right ear) (one side only)
    ``two_earpieces_ear_mastoid``  (left ear - M1) - (right ear - M2) (emulated two separate earpieces)
    ``mastoid_pair``          M1 - M2
    ``ear_common_mode``       mean(all ear electrodes) - mean(M1, M2)
    ``forehead_mastoids``     Fpz - mean(M1, M2) (vertical positive control)
    """
    p = protocol
    bad = dict(bad or {})
    gl = [c for c in p.left_ear if not bad.get(c, False)]
    gr = [c for c in p.right_ear if not bad.get(c, False)]
    m1, m2 = p.mastoid_left, p.mastoid_right
    x = lambda pos, neg: combine(data, labels, bad, pos, neg)  # noqa: E731
    D = {'cross_head': [('left-right', x(gl, gr))],
         'single_cross_pairs': [(f'{a}-{b}', x([a], [b])) for a in gl for b in gr],
         'within_ear_pairs': ([(f'{a}-{b}', x([a], [b])) for a, b in combinations(gl, 2)]
                              + [(f'{a}-{b}', x([a], [b])) for a, b in combinations(gr, 2)]),
         'ear_to_own_mastoid': [(f'left-{m1}', x(gl, [m1])), (f'{m2}-right', x([m2], gr))],
         'mastoid_pair': [(f'{m1}-{m2}', x([m1], [m2]))],
         'ear_common_mode': [('ears-mastoids', x(gl + gr, [m1, m2]))],
         'forehead_mastoids': [(f'{p.forehead}-mastoids', x([p.forehead], [m1, m2]))]}
    a, b = D['ear_to_own_mastoid'][0][1], D['ear_to_own_mastoid'][1][1]
    D['two_earpieces_ear_mastoid'] = [('(left-M1)-(right-M2)', a + b if a is not None and b is not None else None)]
    return D


def two_ear_earpieces(left_pair, right_pair, o_left: float, o_right: float):
    """Emulated two separate earpieces, one per ear: oriented left within-ear pair plus oriented right within-ear pair."""
    return o_left * np.asarray(left_pair, float) + o_right * np.asarray(right_pair, float)


def orientation_from_gains(horizontal_gains: dict) -> dict:
    """Orientation o_r of a member from the other recordings' horizontal gains g_H: +1 if the median over the other
    recordings of -g_H is >= 0 (leftward steps then positive), else -1 (article, Section "Ear derivations")."""
    return leave_one_out_sign({k: -g for k, g in horizontal_gains.items()})


def leave_one_out_sign(values: dict) -> dict:
    """Orientation of each recording's member from the OTHER recordings only.

    ``values`` maps recording -> a scalar such as ``-g_H`` (minus the horizontal gain). The result maps recording ->
    +1 if the median over the other recordings (finite values) is >= 0 or there are none, else -1.
    """
    out = {}
    for k in values:
        o = np.array([values[q] for q in values if q != k and np.isfinite(values[q])])
        out[k] = 1.0 if o.size == 0 or np.median(o) >= 0 else -1.0
    return out
