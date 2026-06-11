![DSGBR banner](assets/readme-banner-v1.png)

[![PyPI](https://img.shields.io/pypi/v/dsgbr.svg)](https://pypi.org/project/dsgbr/)
[![Documentation](https://img.shields.io/badge/docs-README-blue.svg)](https://github.com/ricardofrantz/dsgbr#readme)
[![CI](https://github.com/ricardofrantz/dsgbr/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ricardofrantz/dsgbr/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/dsgbr.svg)](https://pypi.org/project/dsgbr/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

**Dual Savitzky-Golay Baseline Ratio (DSGBR)** detects spectral peaks in
frequency-domain signals. It is built for dense, noisy power spectra from fluid
dynamics, vibration analysis, and related experimental work.

## Algorithm

```
PSD ──► SEARCH (short-scale SG smooth)
            │
            ▼
        BASELINE (long-scale SG smooth)
            │
            ▼
        RATIO = SEARCH / BASELINE
            │
            ▼
        peaks where RATIO ≥ threshold
            │
            ├──► spacing rules
            ├──► ULF guardrail
            └──► band selection (if > max_peaks)
            │
            ▼
        (peak_frequencies, peak_heights)
```

The detector builds a short-scale **SEARCH** signal and a longer-scale
**BASELINE** signal using Savitzky-Golay filtering. A peak is accepted
when `SEARCH / BASELINE` exceeds a configurable ratio threshold, subject
to spacing constraints and an ultra-low-frequency guardrail.

## Install

```bash
pip install dsgbr
```

For development:

```bash
git clone https://github.com/ricardofrantz/dsgbr.git
cd dsgbr
uv pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np
from dsgbr import dsgbr_detector

# Synthetic PSD with known peaks
frequencies = np.linspace(0.001, 1.0, 2048)
psd = np.ones_like(frequencies)
psd[400] = 12.0  # inject a peak
psd[1200] = 8.0  # inject another

peak_f, peak_h = dsgbr_detector(
    frequencies, psd,
    case_info={"ratio_threshold": 1.5, "baseline_window": 61},
)
print(f"Detected {peak_f.size} peaks at f = {peak_f}")
```

## Configuration

All parameters are set through `DetectionConfig` or passed as a dictionary
via the `case_info` argument. Short aliases (RT, SW, BWF, etc.) are
supported for concise configuration.

| Parameter              | Alias | Default      | Description                                  |
| ---------------------- | ----- | ------------ | -------------------------------------------- |
| `ratio_threshold`      | RT    | 1.8          | Min SEARCH/BASELINE ratio for acceptance     |
| `smooth_window`        | SW    | 3            | Savitzky-Golay window for SEARCH (odd, >= 3) |
| `baseline_window_frac` | BWF   | 0.001        | Baseline window as fraction of data length   |
| `distance_low`         | DL    | 2            | Min bin separation below `switch_frequency`  |
| `distance_high`        | DH    | 1            | Min bin separation above `switch_frequency`  |
| `switch_frequency`     | SF    | 0.02         | Frequency threshold for spacing rules        |
| `max_peaks`            | MP    | 25           | Maximum peaks returned                       |
| `smooth_polyorder`     | —     | 2            | Polynomial order for SG filter               |
| `smooth_on_log`        | —     | True         | Smooth log10(PSD) instead of linear          |
| `baseline_window`      | —     | None         | Fixed baseline window (overrides BWF)        |
| `baseline_on_log`      | —     | True         | Baseline smoothing in log domain             |
| `band_strategy`        | —     | proportional | Band allocation: proportional or equal       |
| `n_bands`              | —     | 10           | Number of logarithmic frequency bands        |
| `ulf_fmax`             | —     | 0.001        | ULF band upper frequency limit               |
| `ulf_min_q`            | —     | 9.0          | Minimum Q-factor for ULF peaks               |
| `ulf_max_points`       | —     | 5            | Maximum ULF peaks to retain                  |

## Advanced usage

### Support series for visualization

```python
from dsgbr import compute_support_series

support = compute_support_series(frequencies, psd, case_info={"RT": 2.0})

# Plot SEARCH vs BASELINE overlay
import matplotlib.pyplot as plt
plt.semilogy(frequencies, support["search_series"], label="SEARCH")
plt.semilogy(frequencies, support["baseline_series"], label="BASELINE")
plt.semilogy(frequencies, support["rthreshold"], "--", label="Threshold")
plt.legend()
plt.show()
```

### Band-balanced peak selection

```python
from dsgbr import select_peaks_by_frequency_bands

# Reduce 100 peaks to 15, spread across frequency bands
sel_f, sel_h = select_peaks_by_frequency_bands(
    peak_frequencies, peak_heights,
    max_peaks=15, strategy="proportional", n_bands=8,
)
```

### Configuration via dataclass

```python
from dsgbr import DetectionConfig

cfg = DetectionConfig(ratio_threshold=2.5, smooth_window=7, max_peaks=10)
print(cfg.to_metadata())
```

## API reference

| Function / Class                                                         | Description                                  |
| ------------------------------------------------------------------------ | -------------------------------------------- |
| `dsgbr_detector(f, psd, *, case_info, return_support)`                   | Main detection pipeline                      |
| `compute_support_series(f, psd, case_info)`                              | Return intermediate arrays for visualization |
| `select_peaks_by_frequency_bands(f, h, *, max_peaks, strategy, n_bands)` | Band-balanced down-selection                 |
| `find_nearest_frequency(target, frequencies, heights)`                   | Closest detected frequency lookup            |
| `DetectionConfig`                                                        | Frozen dataclass with 17 parameters          |
| `detect_peaks_case_adaptive(...)`                                        | Deprecated alias for `dsgbr_detector`        |
| `DSGBR_PARAM_ALIASES`                                                    | Short-to-long parameter name mapping         |

## Examples

See [`examples/`](examples/) for runnable scripts:

- **`basic_usage.py`** — minimal detection example
- **`parameter_tuning.py`** — sweep ratio_threshold, compare peak counts
- **`visualization.py`** — SEARCH/BASELINE overlay plot

## How it works

DSGBR applies two Savitzky-Golay passes at different scales to separate
sharp spectral peaks from the slowly varying baseline. The ratio between
these two series naturally highlights peaks above the local background,
making the detector robust to spectral slope and broadband noise. For a
detailed description, see [`docs/algorithm.md`](docs/algorithm.md).

## Citation

If you use DSGBR in your research, please cite:

```bibtex
@software{dsgbr2026,
  author = {Frantz, Ricardo},
  title = {{DSGBR}: Dual Savitzky--Golay Baseline Ratio spectral peak detector},
  year = {2026},
  url = {https://github.com/ricardofrantz/dsgbr},
}
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).

## Contributing

Contributions are welcome. Please open an issue to discuss changes before
submitting a pull request. Run the full QA suite before submitting:

```bash
uv pip install -e ".[dev]"
pre-commit run --all-files
pytest --cov=dsgbr
```
