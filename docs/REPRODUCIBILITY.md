# Reproducibility

This repository supports two distinct levels of reproduction. They answer different questions and must not be
confused.

## 1. Functional reproducibility with synthetic test fixtures

*Question answered:* does every analysis step run, and does it do what the article says?

```bash
export PYTHONPATH=src
python -m pytest                                    # synthetic tests
```

The synthetic generator (`read_eareog.synthetic`) produces the small test fixtures: a reader who fixates along lines,
makes forward saccades, regressions of varied size and return sweeps, and an ear signal proportional to horizontal
gaze plus noise and drift. It is deliberately simple. Results on synthetic data say nothing about real ear recordings
and must never be reported as results.

Synthetic recordings carry the flag `"synthetic": true` in their `meta.json` (it survives saving and loading); when
every input is synthetic, `run_primary` and `run_causal` report `all_synthetic: true` and the command line adds the
notice `SYNTHETIC DATA` to the output.

## 2. End-to-end reproducibility (requires the original recordings)

*Question answered:* given the recordings, does the analysis give the article's numbers?

This requires the original recordings, which are **not** distributed with this repository and are not
downloadable from it (see [DATA_ACCESS.md](DATA_ACCESS.md)). With lawfully obtained recordings converted to the
documented format:

```bash
python -m read_eareog run --data rec01/ rec02/ ... --out results.json
python -m read_eareog run --causal --data rec01/ rec02/ ... --out causal.json
```

The results file records the package version, every parameter, the input order and any excluded recording with the
reason; missing values are written as `null`.

What reproduces, given the recordings:

* **exactly** (up to the clock-map note in DATA_ACCESS.md): each recording's labels, threshold curve, cross-validated
  threshold, detections, F1, precision, recall and false alarms per minute, hence every cohort median; the
  percentile-bootstrap interval of a median is order-independent for an odd number of recordings but can move by up
  to about 0.02 for an even number, because the resampling indices then depend on the order;
* **only in distribution**: quantities that depend on the order of the recordings, because the original order was
  derived from a keyed ordering that cannot be published: the rate-matched random F1 of each recording (seed
  20261121 + 2000 + list position) and its cohort median, and bootstrap intervals of means and paired differences
  (they move by a few thousandths when the order changes).

For an exact match, the following must also match:

* the analysis parameters in `configs/protocol.toml` (unchanged);
* the clock alignment of ear recorder and eye tracker (the article mapped eye-tracker time to each recorder's clock
  by linear fits to shared periodic audio ticks; `read_eareog.fit_clock` fits such a linear map);
* the analysis window W of each recording (first task marker + 2 s to last task marker − 2 s; DATA_ACCESS.md) and
  recordings that extend at least 10 s beyond it (the analysis crops to W ± 10 s itself);
* channel quality: the article's rule is applied automatically;
* NumPy's random generator: random baselines and bootstrap draws use `numpy.random.default_rng`; its streams are
  stable within NumPy 1.26 (the tested version) but are not guaranteed across major versions.

What cannot be reproduced from this repository: any number that depends on the recordings (all F1 values,
false-alarm rates, gains, d', concordance, statistical tests). The scalp-recorder analyses additionally need the
scalp recordings; the package provides the generic functions (derivation, scaling, step statistic, detection and
scoring work at any sampling rate) but no scalp-specific loader.

## Determinism

All random draws use explicit seeds (`statistics.seed = 20261121`). Bootstrap intervals use 10,000 resamples of
recordings. Two runs on the same inputs give identical results (tested).

## Environment

Python ≥ 3.11 (for `tomllib`), NumPy, SciPy; optional scikit-learn (label-free threshold) and pytest.
`requirements.txt` lists the minimum versions; the package was tested with Python 3.11.15, NumPy 1.26.4,
SciPy 1.16.0, scikit-learn 1.6.1 and pytest 9.1.1 (last test run: version 0.1.0rc3, whose analysis modules 0.1.0
contains unchanged).
