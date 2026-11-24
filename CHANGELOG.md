# Changelog

## 0.1.0 (2026-11-24) — first public release

Code-only public release of the experimental procedures, MIT licence.

- Package `read_eareog`: filtering and scaling, ear derivations (including emulated two-earpiece combinations),
  return-sweep labels and fixation-defined line classes, step statistic and detections, one-to-one scoring,
  leave-one-recording-out thresholds, chance levels, velocity-threshold and label-free detectors, saccade-locked gains
  and d', false-alarm null times and concordance, cohort statistics, causal variant, primary analysis pipeline.
- Command line: `run`, `run --causal`, `check-config`.
- Analysis parameters in `configs/protocol.toml`.
- Documentation: input format, reproducibility, method-to-code map, validation, privacy and limitations.
- `LICENSE` (MIT) and `CITATION.cff`.

Compared with the private pre-releases (0.1.0rc1–rc3, whose history is not published), the analysis modules are
unchanged. Removed: the figure module, the figure values shipped as package data, the figure tests and the `figures`
command, the `synthetic-demo` command and the worked example; Matplotlib is no longer a dependency. Two tests were
adapted to the removed commands. The tests were not re-run for this release (last run: 0.1.0rc3).
