# Changelog

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
