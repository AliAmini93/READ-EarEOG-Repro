# Validation against the original analysis code

The original analysis code is not distributed (it is interwoven with access control for the restricted recordings).
Before each release of this package a side-by-side check is run in the authors' private analysis environment (so far
by the AI assistant that wrote the package, under the authors' direction):
each package function is executed together with the corresponding original function on **synthetic inputs only**
(random signals, synthetic event tables and synthetic recordings; no recording is read), and the outputs are compared
for identity (floating-point sums to 1e-12 relative).

Result for version 0.1.0rc3: **31 of 31 checks identical.** Version 0.1.0 contains the same analysis modules
unchanged (only the figure module, its data, the demonstration command and the worked example were removed), so the
check was not repeated.

| Area | What was compared |
|---|---|
| parameters | threshold grid (56 values, exact); refractory period, tolerance, seed, line pitch, velocity λ grid, causal calibration; electrode groups |
| analysed extent and channel quality | cropping to W ± 10 s from fractional sample positions (including a clock offset and slope), the channel-quality rule with outlying and flat channels, and the two-electrodes-per-ear rule |
| filtering and scaling | zero-phase band-pass; robust SD for scaling at 500 and 1000 Hz; causal band-pass with steady-state initialisation at 500, 250 and 64 Hz |
| detector | step statistic at 1000, 500, 250, 125 and 64 Hz (rounding of window limits); detections in 90 signal/threshold cases at four rates; causal statistic and upward crossings at 500 and 64 Hz |
| scoring | one-to-one matching, asymmetric matching and event-centred hits in 300 random cases including detections exactly 0.2 s from labels; boundary at 0.2 s; event hits with a single detection |
| baselines and thresholds | rate-matched random, periodic and asymmetric random baselines with the same seeds; leave-one-recording-out thresholds (synthetic cohort with NaN entries); velocity-threshold candidates for 10 λ values at 500 and 64 Hz and greedy selection; label-free mixture threshold |
| derivations | all eight derivation classes including bad-channel handling; leave-one-out orientation sign |
| labels | return-sweep labels and fixation-defined line classes on synthetic recordings |
| oculomotor quantities | saccade-locked steps with blink and amplitude rejection; OLS gains; quiet fixation times and step noise; single-event d' |
| false alarms | eligible null times, window maxima, graded and binary responses, distances |
| statistics | exact sign-flip p (including n = 22), bootstrap intervals, ratio of means, cohort summary, pooled share, Holm adjustment (also against its definition for 3, 24 and 45 tests) |
| end to end | primary analysis on 5 synthetic recordings, each cropped and quality-checked as in the original (thresholds, detections, F1, random F1, detection attribution); causal variant on 4 synthetic recordings (calibration scaling, thresholds for both matchings, F1, latency, random baselines) |

What this does and does not show: identical outputs on synthetic inputs show that the package implements the same
computations as the code that produced the article's results. Both implementations were written by the same AI
assistant, so the check is not an independent verification of the analysis itself. They do not show that the package reproduces the
article's numbers, which requires the recordings; the package has not been run on them.

The package's own tests (`tests/`) are independent of the original code and check each function against known
answers on synthetic data. They were last run on 0.1.0rc3; in 0.1.0 the figure tests were removed and two tests were
adapted to the removed commands, without a new run.
