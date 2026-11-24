"""read_eareog - detection of return-sweep-like saccades from ear-electrode recordings during reading.

Reader-facing re-implementation of the analysis described in the accompanying article (manuscript in preparation).
See README.md for scope, docs/METHOD_TO_CODE_MAP.md for the correspondence between the article's equations and the
functions below, and docs/PRIVACY_AND_LIMITATIONS.md for what this package can and cannot reproduce.
"""
from .config import PROTOCOL, Protocol
from .derivations import ear_derivations, leave_one_out_sign
from .detector import causal_statistic, detect, step_statistic, upward_crossings
from .labels import line_classes, return_sweep_labels
from ._version import __version__
from .pipeline import attribute_detections, derivation, run_causal, run_primary
from .recording import Recording, fit_clock, load_recording, sample_times, save_recording
from .scoring import event_hits, match, match_asymmetric
from .signal import bandpass, causal_bandpass, robust_sd, scale
from .stats import bootstrap_mean_ci, bootstrap_median_ci, holm, paired_summary, ratio_of_means_ci, sign_flip_exact
from .threshold import f1_curve, loro_thresholds

__all__ = ['PROTOCOL', 'Protocol', 'Recording', 'attribute_detections', 'bandpass', 'bootstrap_mean_ci',
           'bootstrap_median_ci', 'causal_bandpass', 'causal_statistic', 'derivation', 'detect', 'ear_derivations',
           'event_hits', 'f1_curve', 'fit_clock', 'holm', 'leave_one_out_sign', 'line_classes', 'load_recording',
           'loro_thresholds', 'match', 'match_asymmetric', 'paired_summary', 'ratio_of_means_ci', 'return_sweep_labels',
           'robust_sd', 'run_causal', 'run_primary', 'sample_times', 'save_recording', 'scale', 'sign_flip_exact', 'step_statistic',
           'upward_crossings', '__version__']
