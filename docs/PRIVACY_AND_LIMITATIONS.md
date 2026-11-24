# Privacy, rights and limitations

## What is and is not in this repository

Included: original code written for this package, the pre-specified analysis parameters, synthetic test fixtures
and tests, documentation.

Not included, and must never be added: recordings or excerpts of them, eye-tracker event files, per-recording,
per-event or cohort-level outputs, figures or figure values, participant or recording identifiers, timestamps of
sessions, experimental stimuli, protocol documents of the original study, internal project records, credentials,
local file paths. `scripts/release_scan.py` checks a checkout for such material before every push.

## Release status

Version 0.1.0 is the first public release. It is published under the MIT licence ([../LICENSE](../LICENSE)), with the
copyright holder and author confirmed by the owner of the repository. The public Git history starts from a reviewed
code-only tree; the development history of the private pre-releases (0.1.0rc1–rc3) is not published. The licence
covers this code only. It grants no rights to the recordings, whose access is governed by their data controller, or
to the accompanying article.

## Scientific limitations that users of the code should know

* **Recordings, not persons.** Whether the analysed recordings come from distinct individuals is unverified.
  Statistics treat recordings as units.
* **Undocumented geometry.** Electrode sites within each ear, the scalp-cap placement over the earpieces, EOG
  electrode sites, eye-tracker mounting, screen size and viewing distance are undocumented. Screen distances are
  therefore in pixels, not degrees of visual angle, and electrode drawings are conceptual.
* **Amplitude scale.** The stored ear-electrode values are not in the declared microvolt scale; all amplitudes are in
  units of each signal's robust SD. Gains are compared only between derivations of the same recorder.
* **Emulated earpieces.** Two-earpiece montages are re-combinations of one common-reference recording, not
  measurements with separate earpieces; they do not establish how real separate earpieces would perform.
* **What is detected.** The detector responds to large leftward saccades; it does not track lines or reading
  progress (see the article).
* **Multiplicity.** The article's inferential tests are organised in pre-specified families with Holm adjustment;
  conclusions must be drawn at the family level reported there.
* **Exploratory study.** The analysis was amended during the study; the article lists every amendment and every
  withdrawn conclusion.

## Reporting a problem

See [../SECURITY.md](../SECURITY.md).
