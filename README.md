# READ-EarEOG-Repro

Experimental code that accompanies the article **"Detecting return-sweep-like saccades with ear electrodes during
continuous-text reading: an exploratory event-level comparison with eye tracking and scalp electro-oculography"**
(manuscript in preparation; not yet submitted or peer reviewed — a link will be added here when a public version
exists).

> **Version 0.1.0 (tag `v0.1.0`), MIT licence.** This repository contains **code, analysis parameters and
> documentation only**: no recordings, eye-tracker files, stimuli, per-recording or cohort results, figures or figure
> values. Running the experiments requires compatible recordings obtained lawfully from their data controller.

## What the study asked

During reading, the eyes jump back to the start of the next line (a *return sweep*). These large leftward saccades
rotate the corneo-retinal dipole and change the potential difference between electrodes in the left and the right
ear. The study asked whether a wearable ear-electrode recorder can detect such saccades event by event, scored
against a simultaneously recorded eye tracker, and how it compares with scalp electrodes recorded at the same time.

The article's results are not reported or reproduced here: they depend on the recordings, which are not distributed
(see below).

## What this repository contains

| Path | Content |
|---|---|
| `src/read_eareog/` | the experimental procedures as a small Python package: filtering and scaling, ear derivations, eye-tracker labels and line classes, step statistic and detections, one-to-one scoring, leave-one-recording-out thresholds, chance levels and a velocity-threshold detector, saccade-locked gains and d', false-alarm null times and concordance, cohort statistics, the causal variant |
| `configs/protocol.toml` | every pre-specified analysis parameter (no data) |
| `tests/` | unit and integration tests on small **synthetic** fixtures (`read_eareog.synthetic`) |
| `docs/` | input format, reproducibility, method-to-code map, validation, privacy and limitations |
| `scripts/` | a pre-release scan for files and text that must not be published |
| `LICENSE`, `CITATION.cff` | MIT licence; how to cite the software |

## Installation and inputs

Requirements: Python ≥ 3.11 with NumPy and SciPy (scikit-learn only for the optional label-free threshold; pytest for
the tests). The package was tested with Python 3.11, NumPy 1.26 and SciPy 1.16. From a checkout of this repository,
either install it in an environment of your choice (`pip install .`) or use it in place (`export PYTHONPATH=src`).

Each recording must first be converted into one directory in the format of [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md)
(ear-recorder channels with eye-tracker-clock sample times, the analysis window W, and the eye tracker's parsed
saccades, fixations and blinks). No converter or downloader is included, because the source formats are
dataset-specific and the recordings are not redistributed. Name the directories with anonymous identifiers.

```bash
python -m read_eareog check-config                                       # print the analysis parameters
python -m read_eareog run --data rec01/ rec02/ ... --out results.json    # primary analysis
python -m read_eareog run --causal --data rec01/ ... --out causal.json  # causal variant
python -m read_eareog run --config my_protocol.toml --data ... --out x.json   # changed parameters (not a reproduction)
python -m pytest tests                                                   # synthetic tests
```

## Experiments of the article and how to run them

| Experiment (article Methods) | Entry point | Output |
|---|---|---|
| Primary event-level detection: cross-head ear derivation, 0.1–15 Hz zero-phase band-pass, robust-SD scaling, step statistic, leave-one-recording-out threshold, one-to-one scoring within ±0.2 s, false alarms per minute, rate-matched random chance level | `python -m read_eareog run` (`pipeline.run_primary`) | `results.json` |
| Causal variant: causal filtering, scaling over the first 60 s of W, past-only statistic, upward crossings, symmetric and latency-tolerant matching with their own thresholds and random baselines | `python -m read_eareog run --causal` (`pipeline.run_causal`) | `causal.json` |
| Montage comparison over ear derivation classes (single cross-head pairs, ear to own mastoid, emulated ear–mastoid two-earpiece combination, mastoid pair) | Python: `run_primary(recordings, protocol, cls=..., member=...)` with the class names of `derivations.ear_derivations`; members without a fixed sign (within-ear pairs) and the emulated ear-electrodes-only two-earpiece combinations are oriented with `derivations.orientation_from_gains` and formed with `derivations.two_ear_earpieces` before detection | Python dictionary with the layout of `results.json` |
| Saccade-locked ocular steps, horizontal and vertical gains, single-event d′, gains in dB | `oculomotor.gain_saccades`, `epoch_steps`, `gains`, `quiet_times`, `step_noise`, `dprime`, `gain_db` | Python values |
| Conventional detectors and further chance levels: velocity-threshold detector tuned leave-one-recording-out, label-free mixture threshold, periodic baseline | `baselines.velocity_candidates`, `baselines.refractory_select`, `threshold.loro_joint`, `baselines.label_free_threshold` (scikit-learn), `baselines.periodic_f1` | Python values |
| What the detections are: fixation-defined line classes, attribution of detections, event-centred hit rates | `labels.line_classes`, `pipeline.attribute_detections`, `pipeline.pooled_share_ci`, `scoring.event_hits` | Python values |
| False alarms: false-alarm-eligible null times, graded and binary concordance with a second derivation | `falsealarms.eligible_null_times`, `window_max`, `graded_response`, `binary_response` | Python values |
| Scalp-derivation comparison | the generic functions (`signal.bandpass`, `signal.scale`, `detector.step_statistic`, `detector.detect`, `threshold.loro_thresholds`, `scoring.match`) applied to a bipolar scalp derivation formed by the user (for example F7 − F8); no scalp-recorder loader and no comparator-selection function are included | Python values |
| Inferential summaries: paired mean differences with bootstrap CI, d_z and exact sign-flip test; bootstrap CI of medians; ratio of means; Holm adjustment | `stats.paired_summary`, `stats.bootstrap_median_ci`, `stats.ratio_of_means_ci`, `stats.holm`, `stats.sign_flip_exact` | Python values |

Equation-level correspondence and the analyses that are **not** packaged (among them the ocular-step comparison with
random times and its controls, the resampling sensitivity analyses, the development variants and the scalp
comparator selection): [docs/METHOD_TO_CODE_MAP.md](docs/METHOD_TO_CODE_MAP.md).

**Output files.** `results.json` records the package version, every parameter (`protocol`), the input order, the
excluded recordings with the reason, one row per recording (threshold, precision, recall, F1, rate-matched random F1,
numbers of detections and labels, false alarms per minute, detection times) and a cohort summary (median F1 with its
95 % bootstrap interval, median random F1, the paired F1-minus-random summary, median false alarms per minute, median
threshold). `causal.json` has the same layout with the causal quantities (thresholds and F1 for both matchings, false
alarms per minute, median latency, random baselines). Missing values are written as `null`. Both files contain
per-recording values: keep them under the same access conditions as the recordings.

Details on what can and cannot be reproduced: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## How the code relates to the original analysis

The original analysis was carried out with research code that also handled access control and privacy screening of
the raw recordings and is not distributed. This package re-implements the analysis steps reported in the article as
plain functions with explicit inputs and outputs. Each function listed as cross-checked in the method map was executed
side by side with the corresponding original function on synthetic inputs and produced identical outputs (31 checks, including end-to-end runs of the
primary analysis and of the causal variant; summary in [docs/VALIDATION.md](docs/VALIDATION.md)); the package has
**not** been run on the original recordings. The correspondence between the
article's equations and the functions is listed in [docs/METHOD_TO_CODE_MAP.md](docs/METHOD_TO_CODE_MAP.md).

**Use of AI assistance.** The package, its tests and its documentation were written by an AI coding assistant
(Claude, Anthropic) under the direction of the authors, as a re-implementation of the original analysis code, and
were checked against that code as described above. The original analysis code was also written by the AI assistant,
so this check shows that two implementations agree; it is not an independent verification of the analysis, and all
reviews so far were performed by AI instances, not by human reviewers.

## Limitations in one paragraph

Whether the analysed recordings come from distinct individuals is unverified; ear-electrode sites within the ear,
the scalp-cap placement, the eye-tracker mounting and the screen geometry are undocumented; separate earpieces were
not measured (two-earpiece montages are emulated from one common-reference recording). See
[docs/PRIVACY_AND_LIMITATIONS.md](docs/PRIVACY_AND_LIMITATIONS.md).

## Citation and licence

The code is released under the MIT licence ([LICENSE](LICENSE)); copyright © 2026 Ali Amini. To cite the software,
use [CITATION.cff](CITATION.cff). The licence covers this code only; it grants no rights to the recordings or to the
article.
