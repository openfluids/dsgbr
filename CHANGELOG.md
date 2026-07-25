# Changelog

## [Unreleased]

### Changed

- Relicense from BSD-3-Clause to Apache-2.0, effective from 0.5.0 onward. 0.4.0
  and earlier releases remain under BSD-3-Clause.

## [0.4.0] - 2026-07-25

### Changed

- Raise the default `distance_high` from 1 to 5 bins. A separation of 1 is not a constraint at all — any two distinct bins satisfy it — so above `switch_frequency` nothing prevented the noisy top of a single broad peak from being reported as several peaks. The README scene reproduced this: one harmonic whose FWHM spans about 14 bins was returned as two detections 4 bins apart.

  The value was chosen on the benchmark suite rather than on any single figure. At 5 the mean false positives fall (0.800 → 0.600 on `dense_lowfreq`, 3.250 → 3.150 on `noisy_welch`) with no loss of true positives, lifting F1 to 0.680 ± 0.253 and 0.449 ± 0.249 respectively. At 8 the false positives fall further but true peaks start disappearing (`dense_lowfreq` mean tp 1.950 → 1.800), so 5 is the point where precision improves for free. `clean_tones`, `steep_slope`, and `no_peaks` are unaffected.

  Detection output changes for anyone relying on the previous default. Pass `distance_high=1` (or `{"DH": 1}`) to restore it.

- Update the golden fixtures accordingly. Three detections were removed across `dense_lowfreq-rt25` and `noisy_welch-rt25`, each one 2 bins from a peak that was retained; no detection was added and no retained value moved.

- Refresh the README validation table and hero figure, which now shows exactly one detection per injected harmonic.

## [0.3.0] - 2026-07-25

### Fixed

- Restore the highest-frequency peak to band down-selection. The top band was half-open on the right while its upper edge equalled the maximum frequency, so a peak sitting exactly on that edge matched no band and was silently discarded. The outer bands are now unbounded, since the upper edge is `10 ** log10(freq_max)` and does not always round-trip to `freq_max` — a closed comparison alone still dropped the largest peak for values such as 3.7 and 987.654321.
- Rank the over-budget truncation in `select_peaks_by_frequency_bands` by amplitude instead of slicing a frequency-sorted array, which had kept the lowest frequencies regardless of strength.
- Stop the one-slot-per-band floor from exceeding `max_peaks` when `n_bands > max_peaks`, which forced a truncation that discarded peaks the allotment had already promised.
- Re-apply the minimum-separation rule after peak refinement. Hill-climbing towards raw-PSD maxima could pull two legally spaced peaks closer than `distance_low`/`distance_high`.
- Correct the `DSGBR_PARAM_ALIASES` comment, which described a bidirectional mapping the dictionary never contained.

### Changed

- `ulf_max_points=0` now retains no ultra-low-frequency peaks, matching its documented meaning. It previously behaved as unlimited because the cap was skipped on a falsy value. Use `ulf_fmax <= 0` to disable ULF handling entirely.
- Peaks at non-positive frequencies are assigned to the first band rather than being dropped by every band.

Detection results are unchanged on the benchmark suite, the golden fixtures, and the
README figure; these paths are reached only by non-default configurations.

## [0.2.1] - 2026-07-24

### Changed

- Point project URLs, README badges, clone instructions, and the citation entry at `github.com/openfluids/dsgbr` after the repository moved to the openfluids organization.

## [0.2.0] - 2026-06-11

### Added

- Validation section in the README with measured benchmark F1 scores against tuned `scipy.signal.find_peaks` and a reproduction command.
- README hero figure showing detection on a noisy synthetic spectrum with default parameters (`examples/readme_figure.py`).

### Fixed

- Rebuild the detector baseline as a wide rolling median of the raw PSD, decoupled from search smoothing and more robust to narrow peaks.
- Recalibrate default detection thresholds for synthetic clean-tone detection with low false positives on no-peak controls.

## [0.1.2] - 2026-02-13

### Added

- Python 3.13 and 3.14 to CI test matrix and PyPI classifiers.

## [0.1.1] - 2026-02-13

### Fixed

- Pin all GitHub Actions to commit SHAs (zizmor compliance).
- Disable uv cache in CI to prevent cache-poisoning warnings.
- Add `--system` flag to `uv pip install` for CI runners.
- Resolve mypy strict errors in `_detector.py`.
- Exclude tests and examples from mypy strict checking.

### Changed

- Replace Hypothesis property-based tests with deterministic parametrized tests.
- Remove `hypothesis` from dependencies.
- Default branch renamed from `main` to `master`.

## [0.1.0] - 2026-02-13

### Added

- Initial release as standalone package.
- `dsgbr_detector()` — five-stage peak detection pipeline.
- `DetectionConfig` — frozen dataclass with 17 tunable parameters.
- `compute_support_series()` — visualization helper returning intermediate arrays.
- `select_peaks_by_frequency_bands()` — band-balanced peak down-selection.
- `find_nearest_frequency()` — closest-frequency lookup utility.
- Backward-compatible `detect_peaks_case_adaptive()` wrapper.
- `DSGBR.py` shim for `from dsgbr.DSGBR import ...` import paths.
- Input validation in `DetectionConfig.__post_init__`.
- NumPy-style docstrings throughout.
- Comprehensive test suite (113 tests, >90% coverage).
- BSD 3-Clause license.
