# DSGBR Reference Guide

Dual Savitzky–Golay Baseline Ratio (DSGBR) is a spectral peak detector for
frequency-domain signals — power spectral densities from Welch estimates,
periodograms, or similar — where many peaks can coexist in dense frequency
regions and the background is neither flat nor noise-free. The detector grew
out of fluid-dynamics signal analysis, where spectra slope steeply over
several decades and a fixed prominence threshold (the usual
`scipy.signal.find_peaks` recipe) either drowns in low-frequency power or
misses everything at high frequency.

The name reflects the original design, which smoothed at two Savitzky–Golay
scales. The current baseline is a rolling median (see below); the name stays
for continuity.

## How it works

The core idea is to compare the spectrum against a local estimate of its own
background, so the acceptance criterion is a *ratio* rather than an absolute
height. A peak three times above its local background is detected the same
way at 0.01 Hz as at 10 Hz, regardless of the slope between them.

For frequencies `f` and PSD `P(f)`, the pipeline runs these stages in order:

1. **Validate inputs.** Both arrays must be one-dimensional, equal-length,
   finite, with increasing frequencies and non-negative PSD. Violations
   raise `ValueError`; empty inputs return empty results.
1. **Build SEARCH** — a light Savitzky–Golay smooth of the PSD
   (`smooth_window`, default 3; polynomial order `smooth_polyorder`,
   default 2), applied to `log10(P)` by default. SEARCH suppresses
   single-bin noise spikes while keeping peak shapes nearly intact.
1. **Build BASELINE** — a wide rolling median of the *raw* PSD
   (window = `baseline_window_frac` × N, default 0.05, so about N/20 bins),
   also in the log domain by default. The two series are deliberately
   decoupled: changing the search smoothing never changes what counts as
   background.
1. **Form the ratio** `SEARCH / BASELINE` and find local maxima of SEARCH.
   Candidates survive only where the ratio meets `ratio_threshold`
   (default 3.3).
1. **Greedy spacing selection.** Candidates are visited strongest-first;
   each must sit at least `distance_low` bins (default 2) from already
   accepted peaks below `switch_frequency` (default 0.02) and
   `distance_high` bins (default 1) above it. Linear-frequency bins pack
   low frequencies densely, so the stricter low-frequency spacing avoids
   reporting one physical peak twice.
1. **Refine positions.** Each accepted index hill-climbs to the nearest
   local maximum of the *raw* PSD, so reported frequencies and amplitudes
   come from the data, not the smoothed series.
1. **Ultra-low-frequency guardrail.** Peaks below `ulf_fmax` must have a
   Q-factor (centre frequency / FWHM, measured on SEARCH) of at least
   `ulf_min_q`, and at most `ulf_max_points` of them are kept. Near the
   left edge of a spectrum, leakage and detrending residue produce broad
   bumps that pass the ratio test; a sharp genuine line passes the Q test,
   a leakage bump does not.
1. **Band-balanced down-selection.** If more than `max_peaks` survive,
   peaks are reduced across logarithmic frequency bands (`band_strategy`,
   `n_bands`) so the result is not dominated by one crowded region.
   Results are returned sorted by frequency.

## Design notes on the BASELINE

The baseline estimator is the part of the algorithm that has changed most
since the original design, and the choices are worth recording:

- **Median, not mean or smooth.** Narrow peaks occupy a minority of a wide
  window, so a rolling median sits under them instead of being dragged up.
  A peak-masked Savitzky–Golay pass was considered and rejected: it needs
  an iteration scheme and extra public parameters to calibrate.
- **Raw PSD, not the SEARCH series.** An earlier version built the baseline
  from the already-smoothed series with a narrow window; on clean spectra
  the baseline then tracked the peaks themselves and the ratio saturated
  near 1, which is why the rework decoupled the two series.
- **Log domain with a positive floor.** Median filtering runs on
  `log10(P)` by default. Zero bins are clamped to a floor derived from the
  median positive value (scaled by machine epsilon) so that a sparse
  spectrum of mostly-zero bins does not inherit peak power as its floor.
- **Streaming median filter.** The implementation uses
  `scipy.ndimage.median_filter` rather than a sliding-window matrix, which
  would cost O(N·window) memory on long spectra.

If either smoothing step fails (for example a window longer than the data),
the detector warns and falls back to the raw PSD for that series rather than
aborting.

## Public API

- `dsgbr_detector(frequencies, psd, *, case_info=None, return_support=False)`
  - Returns `(peak_frequencies, peak_amplitudes)`.
  - If `return_support=True`, returns `(peak_frequencies, peak_amplitudes, support)`.
- `compute_support_series(frequencies, psd, case_info=None)`
  - Returns internal arrays used for reproducibility and plotting (`SEARCH`, `BASELINE`, ratio, indices).
- `select_peaks_by_frequency_bands(peak_frequencies, peak_heights, *, max_peaks, strategy, n_bands)`
  - Down-selects peaks by band while preserving representation across frequency ranges.
- `DetectionConfig`
  - Frozen dataclass for configuration; validates values on construction.
- `detect_peaks_case_adaptive(...)`
  - Deprecated backward-compatible wrapper around `dsgbr_detector(...)`.

`DSGBR_PARAM_ALIASES` maps compact keys (`RT`, `SW`, `BWF`, `DH`, `DL`, `SF`, `MP`) to canonical names.

## Defaults

- `smooth_window = 3`
- `smooth_polyorder = 2`
- `smooth_on_log = True`
- `baseline_window_frac = 0.05`
- `baseline_on_log = True`
- `ratio_threshold = 3.3`
- `distance_low = 2`
- `distance_high = 1`
- `switch_frequency = 0.02`
- `max_peaks = 25`
- `band_strategy = "proportional"`
- `ulf_fmax = 0.001`, `ulf_min_q = 9.0`, `ulf_max_points = 5`

These defaults were recalibrated together with the rolling-median baseline
against the synthetic benchmark suite in `benchmarks/` (see the validation
table in the README). The sensitivity study below documents how far each can
move before performance degrades.

## Parameter sensitivity

Reproduction command for the recorded study below:

```bash
uv run --extra tests python -m benchmarks.sensitivity
```

The study varied one detector parameter at a time around the current defaults
(`RT=3.3`, `SW=3`, `BWF=0.05`) over deterministic synthetic realizations
(`seed=0`, `n=20` per point). A robust range is the contiguous interval of
tested values containing the best mean F1 where every point remains within 5%
of that scenario's best mean F1 for the same 1-D sweep.

| Scenario      | Parameter | Robust range | Best tested value | Max mean F1 |
| ------------- | --------- | -----------: | ----------------: | ----------: |
| clean_tones   | RT        |      1.5-3.3 |               1.5 |       1.000 |
| clean_tones   | SW        |          3-3 |                 3 |       1.000 |
| clean_tones   | BWF       |    0.05-0.25 |              0.05 |       1.000 |
| dense_lowfreq | RT        |     2.79-3.3 |              2.79 |       0.687 |
| dense_lowfreq | SW        |          3-3 |                 3 |       0.658 |
| dense_lowfreq | BWF       |    0.05-0.25 |             0.112 |       0.662 |
| steep_slope   | RT        |    2.79-3.43 |              2.79 |       1.000 |
| steep_slope   | SW        |          3-7 |                 3 |       0.967 |
| steep_slope   | BWF       |    0.05-0.25 |              0.05 |       0.967 |
| noisy_welch   | RT        |    4.07-4.07 |              4.07 |       0.582 |
| noisy_welch   | SW        |          5-5 |                 5 |       0.558 |
| noisy_welch   | BWF       |   0.05-0.112 |              0.05 |       0.447 |
| no_peaks      | RT        |        1.5-6 |               1.5 |       0.000 |
| no_peaks      | SW        |         3-11 |                 3 |       0.000 |
| no_peaks      | BWF       |    0.01-0.25 |              0.01 |       0.000 |

`RT` controls the acceptance ratio above the local baseline. Below the robust
range the detector admits noise and shoulder structure as false positives,
especially in noisy Welch-like spectra; above it, true low-contrast peaks are
missed as the ratio cutoff becomes too strict. The recorded sweeps show the
best `RT` is scenario-dependent, with clean tones tolerating low thresholds and
noisy spectra favoring stricter thresholds. If your spectra resemble the
noisy_welch scenario (heavy estimator scatter, low peak contrast), `RT` near 4
with `SW=5` measurably outperforms the defaults there; the shipped 3.3 is the
compromise that stays inside the robust range of every other scenario.

`SW` controls the narrow search smoothing window. Values above the robust range
can smear nearby or narrow peaks, reducing localization and recall in dense
low-frequency cases; overly small values preserve noise fluctuations, though the
current default of `3` remained best or near-best in most recorded scenarios.
Noisy Welch spectra were the exception in this run, where `SW=5` reduced false
structure enough to improve mean F1.

`BWF` controls the wide rolling baseline window as a fraction of the spectrum.
Below the robust range the baseline can track local peak shoulders too closely,
shrinking the ratio contrast and creating unstable detections. Above the robust
range the baseline becomes less local and can over- or under-estimate curved
backgrounds; in this run the upper tested values remained robust for clean,
dense, and steep scenarios but noisier spectra narrowed to `0.05-0.112`.

## Support series

`compute_support_series` (or `return_support=True`) exposes the intermediate
arrays, which is the recommended way to debug a detection or to build plots
like the README figure:

- `search_series`, `baseline_series` — the two compared series
- `ratio_series` — their ratio
- `rthreshold` — the acceptance threshold as a series, for overlay plots
- `local_baseline` — baseline scaled by the threshold
- `candidate_indices` — local maxima that passed the ratio gate
- `accepted_indices` — indices surviving spacing, refinement, and guardrails
- `peak_frequencies`, `peak_heights` — the final result
- `detector_config` — the resolved configuration, for provenance

For most users, only `peak_frequencies` and `peak_heights` are required.

## Edge behavior

- Empty inputs return empty arrays (and an empty support dictionary).
- Invalid inputs (length mismatch, non-finite values, non-increasing
  frequencies, negative PSD) raise `ValueError` with a specific message.
- Smoothing or baseline failures degrade to the raw PSD with a
  `RuntimeWarning` instead of raising.
- All-zero or sparse-positive spectra are handled by the positive floor
  described above; an isolated spike over zeros is still detected.

## Package scope

This repository is independent of project-specific pipelines and was extracted
for direct reuse in external projects. The synthetic benchmark and sensitivity
machinery lives in `benchmarks/`; golden-output regression tests in
`tests/test_golden.py` pin the detector's numerical behavior and double as the
tripwire for regenerating the README validation table.
