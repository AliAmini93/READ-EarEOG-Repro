"""The primary analysis as one function: labels -> derivation -> scaling -> step statistic -> leave-one-recording-out
threshold -> one-to-one scoring -> rate-matched random baseline -> cohort summary.

    from read_eareog import Protocol, load_recording, run_primary
    result = run_primary([load_recording(d) for d in dirs], Protocol.load())

The result holds per-recording values (keep them private if the recordings are) and cohort summaries.
"""
from __future__ import annotations

import numpy as np

from ._version import __version__
from .baselines import random_f1, random_f1_asymmetric
from .config import PROTOCOL, Protocol
from .derivations import combine, ear_derivations
from .detector import causal_statistic, detect, step_statistic, upward_crossings
from .labels import line_classes, return_sweep_labels
from .recording import Recording, channel_qc, crop_to_window
from .scoring import false_alarms_per_minute, match, match_asymmetric
from .signal import bandpass, causal_bandpass, robust_sd, scale
from .stats import bootstrap_median_ci, paired_summary
from .threshold import f1_curve, loro_thresholds

MIN_LABELS = 10   # recordings with fewer labels are not scored
MIN_PER_SIDE = 2  # the cross-head derivation needs at least two usable electrodes in each ear


def prepare(rec: Recording, protocol: Protocol = PROTOCOL, qc: bool = True) -> Recording:
    """The analysed extent (W +- pad_s, :func:`crop_to_window`) with the article's channel-quality flags added to any
    flags already set on the recording."""
    r = crop_to_window(rec, protocol.pad_s)
    if qc:
        flags = channel_qc(r, protocol.pad_s)
        r.bad = {c: bool(rec.bad.get(c, False) or flags.get(c, False)) for c in r.ear_labels}
    return r


def filtered_channels(rec: Recording, protocol: Protocol = PROTOCOL) -> np.ndarray:
    """Every ear channel band-passed (zero phase) as in the article."""
    return bandpass(rec.ear, protocol.band_hz[0], protocol.band_hz[1], rec.fs, protocol.order, axis=-1)


def derivation(rec: Recording, cls: str = 'cross_head', member: int = 0, protocol: Protocol = PROTOCOL):
    """One derivation (unscaled, recorder units) of the band-passed ear channels, or None if unavailable."""
    D = ear_derivations(filtered_channels(rec, protocol), rec.ear_labels, rec.bad, protocol)
    if cls == 'cross_head' and min(sum(not rec.bad.get(c, False) for c in side)
                                   for side in (protocol.left_ear, protocol.right_ear)) < MIN_PER_SIDE:
        return None
    if cls not in D:
        raise ValueError(f'unknown derivation class {cls!r}; choose one of {sorted(D)}')
    members = D[cls]
    return members[member][1] if member < len(members) else None


def labels_of(rec: Recording, protocol: Protocol = PROTOCOL) -> np.ndarray:
    lo, hi = rec.window
    return return_sweep_labels(rec.saccades['onset'], rec.saccades['dx'], rec.saccades['dy'], lo, hi, protocol)


def _check_names(recordings):
    names = [r.name for r in recordings]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f'recording names must be unique; duplicated: {dup}')
    return names


def _provenance(p: Protocol, names) -> dict:
    return {'package_version': __version__, 'protocol_source': p.source, 'protocol': p.as_dict(), 'input_order': names}


def run_primary(recordings, protocol: Protocol = PROTOCOL, cls: str = 'cross_head', member: int = 0,
                qc: bool = True) -> dict:
    """Primary event-level analysis of one derivation over a cohort of recordings.

    Each recording is first cropped to W +- pad_s and checked with the channel-quality rule (:func:`prepare`).
    The random baseline of the recording at list position k uses ``numpy.random.default_rng(seed + 2000 + k)``.
    Recordings without a usable derivation or with fewer than MIN_LABELS labels are listed in ``excluded``.
    """
    p = protocol
    names = _check_names(recordings)
    prep, curves, excluded = {}, {}, []
    for k, rec in enumerate(recordings):
        try:
            rec = prepare(rec, p, qc)
        except ValueError as exc:                        # e.g. the recording does not extend 10 s beyond W
            excluded.append({'recording': rec.name, 'reason': str(exc)})
            continue
        lab = labels_of(rec, p)
        x = derivation(rec, cls, member, p)
        if x is None or lab.size < MIN_LABELS:
            excluded.append({'recording': rec.name, 'reason': 'derivation unavailable (bad channels)' if x is None
                             else f'{lab.size} labels (< {MIN_LABELS})'})
            continue
        z = scale(x, rec.fs, p.scale_band_hz, p.pad_s, p.order)
        s = step_statistic(z, rec.fs, p.after_window_s, p.before_window_s)
        curves[rec.name] = f1_curve(s, rec.t, rec.mask, lab, rec.fs, p.thetas, p.tolerance_s, p.refractory_s)
        prep[rec.name] = (k, rec, s, lab)
    if len(curves) < 2:
        raise ValueError('at least two scorable recordings are needed for leave-one-recording-out thresholds; '
                         f'excluded: {excluded}')
    theta = loro_thresholds(curves, p.thetas)
    rows = []
    for name, (k, rec, s, lab) in prep.items():
        det = rec.t[detect(s, theta[name], rec.mask, rec.fs, p.refractory_s)]
        m = match(det, lab, p.tolerance_s)
        rng = np.random.default_rng(p.seed + 2000 + k)
        f1r = random_f1(rng, det.size, rec.window[0], rec.window[1], lab, p.refractory_s, p.random_draws, p.tolerance_s)
        rows.append({'recording': name, 'synthetic': rec.synthetic, 'theta': theta[name], 'precision': m.precision,
                     'recall': m.recall, 'f1': m.f1, 'f1_random': f1r, 'n_detections': int(det.size),
                     'n_labels': int(lab.size), 'false_alarms_per_min': false_alarms_per_minute(m, rec.minutes),
                     'labels_per_min': lab.size / rec.minutes, 'detections': det})
    f1 = np.array([r['f1'] for r in rows])
    cohort = {'derivation': cls, 'n_recordings': len(rows), 'f1': bootstrap_median_ci(f1, p.n_boot, p.seed),
              'f1_random_median': round(float(np.median([r['f1_random'] for r in rows])), 3),
              'f1_minus_random': paired_summary(f1 - np.array([r['f1_random'] for r in rows]), p.n_boot, p.seed),
              'false_alarms_per_min_median': round(float(np.median([r['false_alarms_per_min'] for r in rows])), 2),
              'theta_median': float(np.median([r['theta'] for r in rows]))}
    return {**_provenance(p, names), 'recordings': rows, 'excluded': excluded, 'cohort': cohort,
            'all_synthetic': bool(all(r['synthetic'] for r in rows))}


def causal_signal(rec: Recording, protocol: Protocol = PROTOCOL) -> np.ndarray:
    """Causal cross-head derivation scaled with the first ``calibration_s`` seconds of W only (Eq. 11 inputs):
    every channel causally band-passed (0.1-15 Hz, steady-state initialisation), cross-head derivation, causal
    1-20 Hz band, robust SD over [lo, lo + calibration_s]."""
    p = protocol
    left = [c for c in p.left_ear if not rec.bad.get(c, False)]
    right = [c for c in p.right_ear if not rec.bad.get(c, False)]
    if min(len(left), len(right)) < MIN_PER_SIDE:
        return None
    y = causal_bandpass(rec.ear, p.band_hz[0], p.band_hz[1], rec.fs, p.order)
    x = combine(y, rec.ear_labels, rec.bad, left, right)
    b = causal_bandpass(x[None, :], p.scale_band_hz[0], p.scale_band_hz[1], rec.fs, p.order)[0]
    lo = rec.window[0]
    cal = (rec.t >= lo) & (rec.t <= lo + p.calibration_s)
    rsd = robust_sd(b[cal])
    if not np.isfinite(rsd) or rsd <= 0:
        raise ValueError(f'{rec.name}: flat or non-finite causal derivation in the calibration period')
    return x / rsd


def run_causal(recordings, protocol: Protocol = PROTOCOL, qc: bool = True) -> dict:
    """Causal variant (Eq. 11): past-only statistic, upward crossings, scored after the calibration period with the
    symmetric (+-tolerance) and the latency-tolerant asymmetric matching, each with its own leave-one-recording-out
    threshold. Random baselines use seeds seed + 6000 + k (symmetric) and seed + 6500 + k (asymmetric)."""
    p = protocol
    names = _check_names(recordings)
    prep, cs, ca, excluded = {}, {}, {}, []
    for k, rec in enumerate(recordings):
        try:
            rec = prepare(rec, p, qc)
        except ValueError as exc:
            excluded.append({'recording': rec.name, 'reason': str(exc)})
            continue
        z = causal_signal(rec, p)
        lo, hi = rec.window
        lab = labels_of(rec, p)
        lab = lab[lab > lo + p.calibration_s]
        if z is None or lab.size < MIN_LABELS:
            excluded.append({'recording': rec.name, 'reason': 'derivation unavailable (bad channels)' if z is None
                             else f'{lab.size} labels after calibration (< {MIN_LABELS})'})
            continue
        s = causal_statistic(z, rec.fs, p.causal_after_window_s, p.causal_before_window_s)
        mask = (rec.t >= lo + p.calibration_s) & (rec.t <= hi)

        def dets(th, s=s, mask=mask, rec=rec):
            return rec.t[upward_crossings(s, th, mask, rec.fs, p.refractory_s)]
        cs[rec.name] = np.array([match(dets(th), lab, p.tolerance_s).f1 for th in p.thetas])
        ca[rec.name] = np.array([match_asymmetric(dets(th), lab, p.match_before_s, p.match_after_s)[2] for th in p.thetas])
        prep[rec.name] = (k, rec, dets, lab)
    if len(prep) < 2:
        raise ValueError('at least two scorable recordings are needed for leave-one-recording-out thresholds; '
                         f'excluded: {excluded}')
    ths, tha = loro_thresholds(cs, p.thetas), loro_thresholds(ca, p.thetas)
    rows = []
    for name, (k, rec, dets, lab) in prep.items():
        lo, hi = rec.window
        ds, da = dets(ths[name]), dets(tha[name])
        m = match(ds, lab, p.tolerance_s)
        _, _, f1a, lat = match_asymmetric(da, lab, p.match_before_s, p.match_after_s)
        minutes = (hi - lo - p.calibration_s) / 60
        rows.append({'recording': name, 'synthetic': rec.synthetic, 'theta_symmetric': ths[name],
                     'theta_asymmetric': tha[name], 'f1_symmetric': m.f1, 'f1_asymmetric': f1a,
                     'false_alarms_per_min_symmetric': false_alarms_per_minute(m, minutes),
                     'median_latency_ms': float(np.median(lat)) * 1000 if len(lat) else np.nan,
                     'f1_random_symmetric': random_f1(np.random.default_rng(p.seed + 6000 + k), ds.size,
                                                      lo + p.calibration_s, hi, lab, p.refractory_s, p.random_draws,
                                                      p.tolerance_s),
                     'f1_random_asymmetric': random_f1_asymmetric(np.random.default_rng(p.seed + 6500 + k), da.size,
                                                                  lo + p.calibration_s, hi, lab, p.refractory_s,
                                                                  p.random_draws, p.match_before_s, p.match_after_s)})
    cohort = {'n_recordings': len(rows),
              'f1_symmetric': bootstrap_median_ci([r['f1_symmetric'] for r in rows], p.n_boot, p.seed),
              'f1_asymmetric': bootstrap_median_ci([r['f1_asymmetric'] for r in rows], p.n_boot, p.seed),
              'median_latency_ms': float(np.nanmedian([r['median_latency_ms'] for r in rows]))}
    return {**_provenance(p, names), 'recordings': rows, 'excluded': excluded, 'cohort': cohort,
            'all_synthetic': bool(all(r['synthetic'] for r in rows))}


def attribute_detections(detections, rec: Recording, pitch_px: float, protocol: Protocol = PROTOCOL,
                         tolerance: float = 0.2, blink_radius: float = 0.3) -> dict:
    """Count detections by the fixation-defined class of the nearest parsed saccade within +-tolerance; otherwise
    'BLINK_ONLY' if a blink onset lies within blink_radius, else 'NOTHING'."""
    lo, hi = rec.window
    ev = line_classes(rec.saccades, rec.fixations, lo, hi, pitch_px, protocol)
    keys = ['NL', 'SKIP', 'SAME', 'UP', 'OTHERLEFT', 'PAGE', 'NONLEFT', 'BLINK_ONLY', 'NOTHING']
    cnt = dict.fromkeys(keys, 0)
    for t in np.asarray(detections, float):
        d = np.abs(ev['onset'] - t)
        if np.any(d <= tolerance):
            cnt[ev['cls'][int(np.argmin(d))]] += 1
        elif rec.blinks.size and np.min(np.abs(rec.blinks - t)) <= blink_radius:
            cnt['BLINK_ONLY'] += 1
        else:
            cnt['NOTHING'] += 1
    return cnt


def pooled_share_ci(numerators, denominators, n_boot: int = 10000, seed: int = 20261121):
    """Pooled share sum(num)/sum(den) with a 95 % percentile bootstrap over recordings (rounded to 3 decimals)."""
    num, den = np.asarray(numerators, float), np.asarray(denominators, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, (n_boot, num.size))
    bs = num[idx].sum(1) / den[idx].sum(1)
    return (round(float(num.sum() / den.sum()), 3),
            [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)])
