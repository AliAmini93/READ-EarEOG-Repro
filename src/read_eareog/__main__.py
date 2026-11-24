"""Command-line interface.

    python -m read_eareog run --data DIR [DIR ...] --out results.json [--causal] [--config FILE]
    python -m read_eareog check-config [--config FILE]      # print the analysis parameters
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from ._version import __version__
from .config import Protocol
from .pipeline import run_causal, run_primary
from .recording import load_recording


def _clean(o):
    """JSON-safe copy: arrays to lists, NumPy scalars to Python, NaN/inf to null."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _write(result: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(result), indent=1, allow_nan=False), encoding='utf-8')


def _parser():
    ap = argparse.ArgumentParser(prog='read_eareog', description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--version', action='version', version=__version__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='run the analysis on recordings in the documented format (docs/DATA_ACCESS.md)')
    r.add_argument('--data', type=Path, nargs='+', required=True, help='one directory per recording')
    r.add_argument('--out', type=Path, required=True, help='results JSON file (contains per-recording values)')
    r.add_argument('--causal', action='store_true', help='run the causal variant (Eq. 11) instead of the primary one')
    r.add_argument('--config', type=Path, help='protocol TOML file (default: the article values)')
    c = sub.add_parser('check-config', help='print the analysis parameters')
    c.add_argument('--config', type=Path, help='protocol TOML file (default: the article values)')
    return ap


def _summary(res: dict) -> str:
    c = res['cohort']
    if 'f1' in c:
        return (f"{c['n_recordings']} recordings: median F1 {c['f1'].get('median')} {c['f1'].get('ci95')}, "
                f"rate-matched random {c['f1_random_median']}, false alarms/min {c['false_alarms_per_min_median']}")
    return (f"{c['n_recordings']} recordings (causal): median F1 {c['f1_symmetric'].get('median')} (+-tolerance), "
            f"{c['f1_asymmetric'].get('median')} (latency-tolerant), median latency {c['median_latency_ms']:.0f} ms")


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    try:
        if a.cmd == 'check-config':
            p = Protocol.load(a.config)
            print(f'source: {p.source}')
            for k, v in p.as_dict().items():
                if k != 'source':
                    print(f'{k:24s} {v}')
            return 0
        p = Protocol.load(a.config)
        recs = [load_recording(d) for d in a.data]
        res = (run_causal if a.causal else run_primary)(recs, p)
        if res['all_synthetic']:
            from .synthetic import SYNTHETIC_NOTICE
            res['notice'] = SYNTHETIC_NOTICE
            print(SYNTHETIC_NOTICE)
        _write(res, a.out)
        print(_summary(res) + f'; results in {a.out}')
        for e in res['excluded']:
            print(f"excluded {e['recording']}: {e['reason']}")
        print('The output contains per-recording values: keep it under the same access conditions as the recordings.')
        return 0
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
