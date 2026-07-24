# Changelog

## [Unreleased]

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
