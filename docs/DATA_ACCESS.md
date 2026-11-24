# Data access and input format

## The original recordings are not distributed

The recordings analysed in the article (simultaneous eye tracking, wearable ear-electrode recordings and scalp
recordings during continuous-text reading) are **not** part of this repository and cannot be obtained through it.
Their availability, the conditions of access and the data controller are stated in the article's data-availability
statement (to be confirmed by the authors). Do not ask for, upload or commit recordings or any per-recording
output to this repository.

## What the analysis needs from each recording

| Item | Description | Notes |
|---|---|---|
| ear-recorder channels | signals of the ear electrodes, both mastoids and the forehead site, in recorder units | channel labels as in `configs/protocol.toml` (left ear EL1, EL3, EL4, EL5; right ear ER1–ER4; M1, M2, Fpz) |
| sampling rate | Hz | 500 Hz in the article |
| sample times | eye-tracker-clock time of every ear sample | requires clock alignment, e.g. shared audio ticks (`fit_clock`; see the note below) |
| analysis window W | from 2 s after the first to 2 s before the last task marker of the reading task in the eye tracker's event stream (the article's "reading period minus 2 s at each end"), eye-tracker clock | scoring, labels and detections are restricted to W; the ear recording must extend at least 10 s beyond W on both sides |
| saccades | onset, duration, start and end x/y in screen pixels (y downwards) | parsed by the eye tracker for the tracked (left) eye |
| fixations | onset, duration, x, y | used for line classes and fixation noise |
| blinks | onsets | used to reject epochs and control times |
| bad channels | optional list | excluded from derivations |

Return-sweep labels are computed from the saccades (Δx < −300 px and |Δy| < 150 px inside W). The amplitude scale of
the ear recorder does not matter for detection, because every derivation is divided by its own robust SD.

**Analysed extent and channel quality.** The analysis itself crops every recording to W extended by 10 s on each
side (first sample floor(s(lo) − 10 s), end ceil(s(hi) + 10 s), with s(·) the fractional sample position of an
eye-tracker time); filters are applied to this extent and the robust SD for scaling is taken over W. It then applies
the article's channel-quality rule (`recording.channel_qc`: robust SD of the 1–20 Hz band outside 1/20–20 times the
median over channels, or more than 1 % of samples in runs of at least 250 identical values); the cross-head derivation
needs at least two usable electrodes per ear. Provide all ear-recorder channels (the median is taken over them).

**Clock alignment.** The shared timing events are periodic audio ticks that both the ear recorder and the eye tracker
logged as events; extracting and pairing them is dataset-specific and not part of
this package. The article fitted the map between the eye-tracker clock and each recorder's clock in two steps:
audio-tick pairs were matched within 0.5 s after a coarse offset from the task markers, a linear map from eye-tracker
time to recorder time was fitted, and recorder seconds were converted to samples with the recorder's own sample
clock; the map was then inverted. `fit_clock` fits the equivalent one-step linear map from sample index to eye-tracker
time; on such data the two differ by well under 1 ms, which is far below the 0.2-s matching tolerance but can move a
detection by one sample.

## Directory format read by `load_recording`

One directory per recording; all times in seconds on the eye-tracker clock:

```
rec01/
  ear.npz          data   float array, channels x samples
                   labels str array, channel labels
                   fs     scalar, Hz
                   t      float array, eye-tracker-clock time of each sample
  saccades.csv     onset,duration,start_x,start_y,end_x,end_y
  fixations.csv    onset,duration,x,y
  blinks.csv       onset
  meta.json        {"window": [lo, hi], "bad_channels": [], "synthetic": false}
```

Event tables are sorted by onset when loaded; channels that contain non-finite samples must be flagged as bad (or
repaired), and the sample times must match the sampling rate (checked). Name directories with anonymous, unique
identifiers. `read_eareog.save_recording` writes a `Recording` in exactly this format and can serve as the last
step of a converter.

## Outputs

`run_primary` returns per-recording values (thresholds, detections, F1) and cohort summaries. Per-recording outputs
derived from restricted recordings are themselves restricted: keep them under the same access conditions and share
only cohort-level summaries.
