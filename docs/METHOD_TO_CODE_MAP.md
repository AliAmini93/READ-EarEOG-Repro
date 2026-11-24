# Method-to-code map

Equation numbers refer to the article's Methods (main text) and Supplement. "Cross-checked" means the function was
executed side by side with the original analysis function on synthetic inputs and gave identical output.

| Article | Quantity | Function | Cross-checked |
|---|---|---|---|
| Eq. 1 | return-sweep labels: Δx < −300 px, \|Δy\| < 150 px, onset in W | `labels.return_sweep_labels` | yes |
| Eq. 2 | fixation-defined line classes (next line, skip, same line, earlier line, page-level, other) | `labels.line_classes`, `labels.fixation_runs` | yes |
| Eq. 3 | ear derivations: cross-head mean(left) − mean(right); pairs; mastoids; emulated two-earpiece combinations | `derivations.ear_derivations`, `derivations.two_ear_earpieces` | yes |
| §3.3 | analysed extent W ± 10 s; channel-quality rule; two usable electrodes per ear | `recording.crop_to_window`, `recording.channel_qc`, `pipeline.prepare` | yes |
| §3.6 | zero-phase 0.1–15 Hz band-pass (order 4, 8 poles) | `signal.bandpass` | yes |
| Eq. 3 | leave-one-out orientation of members from the other recordings' gains | `derivations.orientation_from_gains`, `derivations.leave_one_out_sign` | yes |
| Eq. 4 | scaling by the robust SD of the 1–20 Hz band | `signal.scale`, `signal.scale_rsd` | yes |
| Eq. 5 | saccade-locked ocular step, epoch rejection | `oculomotor.epoch_steps` | yes |
| Eq. 6 | step statistic s(t), windows A = [0.10, 0.30) s, B = [−0.25, −0.05) s, limits rounded to samples | `detector.step_statistic` | yes (1000, 500, 250, 125, 64 Hz) |
| Eq. 7 | detections: local maxima ≥ θ, at least int(0.6 s · fs) samples apart | `detector.detect` | yes |
| Eq. 8 | one-to-one matching within ±0.2 s; precision, recall, F1; false alarms per minute | `scoring.match`, `scoring.false_alarms_per_minute` | yes |
| Eq. 9 | leave-one-recording-out threshold on the grid 0.5, 0.6, …, 6.0 | `threshold.loro_thresholds`, `threshold.f1_curve` | yes |
| Eq. 10 | saccades with max(\|Δx\|, \|Δy\|) ≥ 20 px in W; horizontal and vertical gains (OLS per 100 px); step noise at quiet fixation times; single-event d′; gains in dB | `oculomotor.gain_saccades`, `oculomotor.gains`, `oculomotor.quiet_times`, `oculomotor.step_noise`, `oculomotor.dprime`, `oculomotor.gain_db` | yes (selection and dB: formula only) |
| Eq. 11 | causal variant: causal filtering, scaling over the first 60 s of W, past-only statistic, upward crossings, scoring after the calibration period with symmetric and asymmetric matching | `pipeline.causal_signal`, `pipeline.run_causal`, `signal.causal_bandpass`, `detector.causal_statistic`, `detector.upward_crossings`, `scoring.match_asymmetric` | yes (end to end) |
| Methods, chance levels | rate-matched random, periodic and asymmetric random baselines | `baselines.random_f1`, `baselines.periodic_f1`, `baselines.random_f1_asymmetric` | yes |
| Methods, conventional detectors | velocity-threshold candidates and greedy selection; label-free mixture threshold | `baselines.velocity_candidates`, `baselines.refractory_select`, `baselines.label_free_threshold`, `threshold.loro_joint` | yes (`loro_joint`: same rule as `loro_thresholds`) |
| Results, detection identity | attribution of detections to the nearest parsed saccade; pooled shares | `pipeline.attribute_detections`, `pipeline.pooled_share_ci` | yes |
| Results, hit rates by size | event-centred hit indicator | `scoring.event_hits` | yes |
| Supplement S5 | exact sign-flip test, percentile bootstrap, Holm adjustment, cohort summaries | `stats.*` | yes |
| Supplement S5 | false-alarm-eligible null times; graded and binary concordance | `falsealarms.*` | yes |
| — | primary analysis end to end | `pipeline.run_primary` | yes (per-recording thresholds, detections, F1 and random F1) |

Not cross-checked (no original counterpart as a pure function): the recording loader and clock-alignment helpers
(`recording.*`), the synthetic test-fixture generator, and the layout of the cohort summary returned by
`run_primary`.

Analyses of the article that are **not** implemented as dedicated functions:

* the event-locked waveforms and the ocular-step analysis of §3.7 (comparison with random times, controls, AUROC,
  R² of gaze);
* the sensitivity analyses over sampling rate and resampling (the functions accept any sampling rate, but the
  polyphase resampling step is not packaged), and recall at matched false-alarm rates;
* the development variants (logistic model, within-recording calibration);
* the chance hit rate and the line-change analysis by size bin (the event-centred hit indicator is provided);
* the selection of the scalp comparator S* and all scalp-recorder handling (the generic derivation, scaling,
  detection and scoring functions work at any sampling rate);
* the gaze trace and gaze-change categories of the false-alarm analysis, and its head-motion and marker conditions
  (which need inertial and marker channels).

The original implementations of these analyses are part of research code that is interwoven with the access control
of the restricted recordings; no separable implementation exists, so they are omitted rather than re-implemented.

Details of the side-by-side check: [VALIDATION.md](VALIDATION.md).
