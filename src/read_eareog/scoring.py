"""Event-level scoring (article Eq. 8).

Times are in seconds on the eye-tracker clock.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Match:
    precision: float
    recall: float
    f1: float
    label_matched: np.ndarray       # per sorted label: matched to a detection
    detection_matched: np.ndarray   # per sorted detection: true positive

    @property
    def tp(self) -> int:
        return int(self.detection_matched.sum())

    @property
    def fp(self) -> int:
        return int(self.detection_matched.size - self.tp)

    @property
    def fn(self) -> int:
        return int(self.label_matched.size - self.tp)


def match(detections, labels, tolerance: float = 0.2) -> Match:
    """One-to-one matching in time order: each detection takes the nearest still-unmatched label within
    +-tolerance (ties to the earlier label). F1 = 2TP / (2TP + FP + FN); NaN when undefined.

    With detections at least 0.6 s apart and a 0.2-s tolerance, this greedy rule yields a maximum matching.
    """
    det = np.sort(np.asarray(detections, float))
    lab = np.sort(np.asarray(labels, float))
    used, hit = np.zeros(lab.size, bool), np.zeros(det.size, bool)
    for i, t in enumerate(det):
        j = np.searchsorted(lab, t)
        cands = [k for k in (j - 1, j) if 0 <= k < lab.size and not used[k] and abs(lab[k] - t) <= tolerance]
        if cands:
            k = min(cands, key=lambda k: abs(lab[k] - t))
            used[k], hit[i] = True, True
    tp = int(hit.sum())
    fp, fn = det.size - tp, lab.size - tp
    prec = tp / det.size if det.size else np.nan
    rec = tp / lab.size if lab.size else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return Match(prec, rec, f1, used, hit)


def false_alarms_per_minute(m: Match, minutes: float) -> float:
    """False positives per minute of the analysis window."""
    return m.fp / minutes


def match_asymmetric(detections, labels, before: float = 0.1, after: float = 0.5):
    """One-to-one matching for the causal variant: a label onset may lie in [d - after, d + before] of detection d.
    Returns (precision, recall, F1, latencies d - label of the matched pairs)."""
    det = np.sort(np.asarray(detections, float))
    lab = np.sort(np.asarray(labels, float))
    used = np.zeros(lab.size, bool)
    lat = []
    for t in det:
        c = np.flatnonzero(~used & (lab >= t - after) & (lab <= t + before))
        if c.size:
            k = c[np.argmin(np.abs(lab[c] - t))]
            used[k] = True
            lat.append(t - lab[k])
    tp = len(lat)
    fp, fn = det.size - tp, lab.size - tp
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return (tp / det.size if det.size else np.nan), (tp / lab.size if lab.size else np.nan), f1, np.array(lat)


def event_hits(detections, events, tolerance: float = 0.2) -> np.ndarray:
    """Event-centred hit indicator (not one-to-one): is there any detection within +-tolerance of each event?"""
    det = np.sort(np.asarray(detections, float))
    events = np.asarray(events, float)
    if det.size == 0:
        return np.zeros(events.size, bool)
    if det.size == 1:
        return np.abs(det[0] - events) <= tolerance
    j = np.clip(np.searchsorted(det, events), 1, det.size - 1)
    d = np.minimum(np.abs(det[j] - events), np.abs(det[j - 1] - events))
    return d <= tolerance
