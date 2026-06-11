"""Core DSGBR detection pipeline.

The five-stage pipeline:

1. Build **SEARCH** series via Savitzky-Golay smoothing of the PSD.
2. Build **BASELINE** series from a wide rolling median of the raw PSD.
3. Accept candidate peaks where SEARCH / BASELINE >= ratio_threshold.
4. Apply spacing rules and ULF guardrail.
5. Optionally down-select across frequency bands.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, peak_widths, savgol_filter

from dsgbr._config import DetectionConfig
from dsgbr._selection import select_peaks_by_frequency_bands

try:  # not re-exported publicly; private module may move between SciPy versions
    from scipy.signal._peak_finding_utils import PeakPropertyWarning
except ImportError:  # pragma: no cover - fallback if the private module moves

    class PeakPropertyWarning(RuntimeWarning):  # type: ignore[no-redef]
        """Fallback warning class when SciPy does not expose PeakPropertyWarning."""


def dsgbr_detector(
    frequencies: np.ndarray,
    psd: np.ndarray,
    *,
    case_info: Any | None = None,
    return_support: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run the full DSGBR peak detection pipeline.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Frequency axis (Hz), monotonically increasing.
    psd : numpy.ndarray
        Power spectral density values corresponding to *frequencies*.
    case_info : dict or None, optional
        Parameter dictionary passed to :meth:`DetectionConfig.from_case_info`.
    return_support : bool, optional
        If ``True``, return intermediate arrays for visualization.

    Returns
    -------
    peak_f : numpy.ndarray
        Detected peak frequencies, sorted ascending.
    peak_h : numpy.ndarray
        Corresponding PSD amplitudes.
    support : dict
        Only returned when *return_support* is ``True``.  Contains
        ``search_series``, ``baseline_series``, ``ratio_series``,
        ``rthreshold``, ``detector_config``, ``candidate_indices``,
        ``accepted_indices``, ``peak_frequencies``, and ``peak_heights``.

    Raises
    ------
    ValueError
        If non-empty inputs have mismatched lengths, are not one-dimensional,
        contain non-finite values, have non-increasing frequencies, or contain
        negative PSD values.

    Examples
    --------
    >>> import numpy as np
    >>> f = np.linspace(0.001, 1.0, 2048)
    >>> psd = np.ones_like(f)
    >>> psd[500] = 10.0
    >>> peak_f, peak_h = dsgbr_detector(f, psd, case_info={"RT": "1.5"})
    """
    frequencies = np.asarray(frequencies)
    psd = np.asarray(psd)

    if frequencies.size == 0 or psd.size == 0:
        support = _initial_support()
        return (
            (np.array([]), np.array([]), support)
            if return_support
            else (np.array([]), np.array([]))
        )

    _validate_detector_inputs(frequencies, psd)

    cfg = DetectionConfig.from_case_info(case_info)
    search_series = _build_search_series(psd, cfg)
    baseline_series = _build_baseline_series(psd, cfg)
    ratio_series = search_series / np.maximum(baseline_series, 1e-300)

    support = _build_support(search_series, baseline_series, ratio_series, cfg)

    candidate_indices, _ = find_peaks(search_series, distance=1)
    support["candidate_indices"] = candidate_indices.copy()
    if candidate_indices.size == 0:
        return (
            (np.array([]), np.array([]), support)
            if return_support
            else (np.array([]), np.array([]))
        )

    candidate_indices = candidate_indices[ratio_series[candidate_indices] >= cfg.ratio_threshold]
    support["candidate_indices"] = candidate_indices.copy()
    if candidate_indices.size == 0:
        return (
            (np.array([]), np.array([]), support)
            if return_support
            else (np.array([]), np.array([]))
        )

    # Greedy spacing-aware selection (strongest first)
    order = np.argsort(search_series[candidate_indices])[::-1]
    accepted: list[int] = []
    low_distance = int(cfg.distance_low)
    high_distance = int(cfg.distance_high)
    switch_frequency = float(cfg.switch_frequency)
    for idx in order:
        peak_idx = int(candidate_indices[idx])
        freq = float(frequencies[peak_idx])
        min_dist = high_distance if freq >= switch_frequency else low_distance
        if all(abs(peak_idx - existing) >= min_dist for existing in accepted):
            accepted.append(peak_idx)

    if not accepted:
        support["accepted_indices"] = np.array([], dtype=int)
        return (
            (np.array([]), np.array([]), support)
            if return_support
            else (np.array([]), np.array([]))
        )

    accepted_idx = np.array(sorted(set(accepted)), dtype=int)
    accepted_idx = _refine_peak_indices(accepted_idx, psd)
    accepted_idx = _apply_ulf_guardrail(accepted_idx, frequencies, search_series, cfg)
    support["accepted_indices"] = accepted_idx.copy()
    if accepted_idx.size == 0:
        return (
            (np.array([]), np.array([]), support)
            if return_support
            else (np.array([]), np.array([]))
        )

    peak_f = frequencies[accepted_idx]
    peak_h = psd[accepted_idx]

    max_peaks = int(cfg.max_peaks)
    if peak_f.size > max_peaks:
        peak_f, peak_h = select_peaks_by_frequency_bands(
            peak_f,
            peak_h,
            max_peaks=max_peaks,
            strategy=cfg.band_strategy,
            n_bands=cfg.n_bands,
        )
        if peak_f.size == 0:
            support["accepted_indices"] = np.array([], dtype=int)
            return (
                (np.array([]), np.array([]), support)
                if return_support
                else (np.array([]), np.array([]))
            )

    order = np.argsort(peak_f)
    peak_f = peak_f[order]
    peak_h = peak_h[order]

    support["peak_frequencies"] = peak_f
    support["peak_heights"] = peak_h
    final_indices = np.array([int(np.argmin(np.abs(frequencies - f))) for f in peak_f], dtype=int)
    support["accepted_indices"] = final_indices

    if return_support:
        return peak_f, peak_h, support
    return peak_f, peak_h


def compute_support_series(
    frequencies: np.ndarray,
    psd: np.ndarray,
    case_info: Any | None = None,
) -> dict[str, Any]:
    """Return intermediate SEARCH, BASELINE, and ratio arrays for visualization.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Frequency axis (Hz).
    psd : numpy.ndarray
        Power spectral density values.
    case_info : dict or None, optional
        Parameter dictionary for :meth:`DetectionConfig.from_case_info`.

    Returns
    -------
    dict
        Keys: ``search_series``, ``baseline_series``, ``local_baseline``,
        ``ratio_series``, ``rthreshold``, ``detector_config``,
        ``candidate_indices``, ``accepted_indices``, ``peak_frequencies``,
        ``peak_heights``.
    """
    result = dsgbr_detector(frequencies, psd, case_info=case_info, return_support=True)
    return result[2]  # type: ignore[misc]  # return_support=True guarantees 3-tuple


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_detector_inputs(frequencies: np.ndarray, psd: np.ndarray) -> None:
    """Validate non-empty detector inputs before numeric processing."""
    if frequencies.size != psd.size:
        raise ValueError(
            "frequencies and psd must have the same length "
            f"(frequencies length={frequencies.size}, psd length={psd.size})"
        )

    if frequencies.ndim != 1 or psd.ndim != 1:
        raise ValueError(
            "frequencies and psd must be one-dimensional "
            f"(frequencies ndim={frequencies.ndim}, psd ndim={psd.ndim})"
        )

    bad_frequencies = int(np.count_nonzero(~np.isfinite(frequencies)))
    if bad_frequencies:
        raise ValueError(f"frequencies contains {bad_frequencies} non-finite values")

    bad_psd = int(np.count_nonzero(~np.isfinite(psd)))
    if bad_psd:
        raise ValueError(f"psd contains {bad_psd} non-finite values")

    if np.any(np.diff(frequencies) <= 0):
        raise ValueError("frequencies must be strictly increasing")

    if np.any(psd < 0):
        raise ValueError("psd must be nonnegative")


def _build_search_series(psd: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
    """Construct the SEARCH series by Savitzky-Golay smoothing the PSD.

    Parameters
    ----------
    psd : numpy.ndarray
        Raw power spectral density.
    cfg : DetectionConfig
        Configuration controlling smoothing behaviour.

    Returns
    -------
    numpy.ndarray
        Smoothed SEARCH series (same length as *psd*).
    """
    if cfg.smooth and cfg.smooth.lower() != "none":
        try:
            win = int(cfg.smooth_window)
            if win >= 3 and win < len(psd):
                arr = np.log10(psd + 1e-300) if cfg.smooth_on_log else psd
                arr = savgol_filter(arr, window_length=win, polyorder=int(cfg.smooth_polyorder))
                return np.power(10.0, arr) if cfg.smooth_on_log else arr
        except ValueError as exc:
            warnings.warn(
                f"SEARCH smoothing failed ({exc}); returning raw PSD",
                RuntimeWarning,
                stacklevel=2,
            )
    return cast(np.ndarray, psd.copy())


def _build_baseline_series(psd: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
    """Construct a peak-robust BASELINE series from the raw PSD.

    The baseline is deliberately decoupled from SEARCH smoothing: it uses a
    wide rolling median (default ``baseline_window_frac=0.05``, about N/20)
    directly on the raw PSD rather than on the Savitzky-Golay SEARCH series.
    The median is the primary robust estimator because narrow peaks occupy a
    minority of a wide window; a peak-masked Savitzky-Golay pass was rejected as
    more stateful and harder to calibrate without adding public parameters.

    Parameters
    ----------
    psd : numpy.ndarray
        Raw power spectral density.
    cfg : DetectionConfig
        Configuration controlling baseline window and domain.

    Returns
    -------
    numpy.ndarray
        BASELINE series (same length as *psd*).
    """
    base = psd
    if cfg.baseline_on_log:
        positive_values = base[base > 0]
        if positive_values.size:
            min_positive = float(np.min(positive_values))
            median_positive = float(np.median(positive_values))
            dtype = np.asarray(base).dtype
            eps = np.finfo(dtype).eps if np.issubdtype(dtype, np.floating) else np.finfo(float).eps
            positive_floor = median_positive * float(eps)
            if not 0.0 < positive_floor < min_positive:
                positive_floor = float(np.nextafter(min_positive, 0.0))
        else:
            positive_floor = 1e-300
        base = np.log10(np.maximum(base, positive_floor))
    try:
        win = _baseline_window_length(len(psd), cfg)
        if win <= 3 or win >= len(psd):
            base_sm = base.copy()
        else:
            # streaming median: avoids the N x win window matrix of a
            # sliding_window_view approach (O(N*win) memory on long spectra);
            # mode="nearest" replicates edge values, matching edge padding
            base_sm = median_filter(base, size=win, mode="nearest")
        baseline = np.power(10.0, base_sm) if cfg.baseline_on_log else base_sm
        return cast(np.ndarray, baseline)
    except ValueError as exc:
        warnings.warn(
            f"BASELINE estimation failed ({exc}); using raw PSD",
            RuntimeWarning,
            stacklevel=2,
        )
        fallback = psd.copy() if not cfg.baseline_on_log else np.power(10.0, base)
        return cast(np.ndarray, fallback)


def _baseline_window_length(n_points: int, cfg: DetectionConfig) -> int:
    """Return an odd baseline window length independent of SEARCH smoothing."""
    if cfg.baseline_window and int(cfg.baseline_window) > 0:
        win = int(cfg.baseline_window)
    elif cfg.baseline_window_frac and cfg.baseline_window_frac > 0:
        win = int(max(7, round(n_points * float(cfg.baseline_window_frac))))
    else:
        win = max(15, (n_points // 50) * 2 + 1)
    if win % 2 == 0:
        win += 1
    return win


def _apply_ulf_guardrail(
    indices: np.ndarray,
    frequencies: np.ndarray,
    search_series: np.ndarray,
    cfg: DetectionConfig,
) -> np.ndarray:
    """Filter ultra-low-frequency peaks by Q-factor and amplitude cap.

    Parameters
    ----------
    indices : numpy.ndarray
        Accepted peak indices.
    frequencies : numpy.ndarray
        Frequency axis.
    search_series : numpy.ndarray
        SEARCH series for width estimation.
    cfg : DetectionConfig
        ULF guardrail parameters.

    Returns
    -------
    numpy.ndarray
        Filtered peak indices.
    """
    if indices.size == 0 or cfg.ulf_fmax <= 0:
        return indices

    freq = np.asarray(frequencies)
    ul_mask = freq[indices] < cfg.ulf_fmax
    if not np.any(ul_mask):
        return indices

    ul_indices = indices[ul_mask]
    df = float(np.median(np.diff(freq))) if freq.size > 1 else 1.0

    valid_mask = search_series[ul_indices] > 0
    if not np.any(valid_mask):
        return cast(np.ndarray, indices[~ul_mask])

    target_indices = ul_indices[valid_mask]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PeakPropertyWarning)
        widths, _, _, _ = peak_widths(search_series, target_indices, rel_height=0.5)

    if widths.size == 0:
        return cast(np.ndarray, indices[~ul_mask])

    positive_widths = widths > 0
    if not np.any(positive_widths):
        return cast(np.ndarray, indices[~ul_mask])

    target_indices = target_indices[positive_widths]
    widths = widths[positive_widths]

    fwhm = widths * df
    q_vals = np.maximum(1e-12, freq[target_indices]) / np.maximum(1e-12, fwhm)
    keep = q_vals >= cfg.ulf_min_q
    ul_indices = target_indices[keep]
    if ul_indices.size == 0:
        return cast(np.ndarray, indices[~ul_mask])

    cap = int(cfg.ulf_max_points)
    if cap and ul_indices.size > cap:
        order = np.argsort(search_series[ul_indices])[::-1][:cap]
        ul_indices = ul_indices[order]

    combined = np.concatenate([indices[~ul_mask], ul_indices])
    combined = np.unique(combined.astype(int))
    return combined


def _initial_support() -> dict[str, Any]:
    """Return an empty support dictionary for zero-length inputs."""
    cfg = DetectionConfig()
    baseline = np.array([], dtype=float)
    return {
        "search_series": np.array([], dtype=float),
        "baseline_series": baseline,
        "local_baseline": baseline,
        "ratio_series": np.array([], dtype=float),
        "rthreshold": np.array([], dtype=float),
        "detector_config": cfg.to_metadata(),
        "candidate_indices": np.array([], dtype=int),
        "accepted_indices": np.array([], dtype=int),
        "peak_frequencies": np.array([], dtype=float),
        "peak_heights": np.array([], dtype=float),
    }


def _build_support(
    search_series: np.ndarray,
    baseline_series: np.ndarray,
    ratio_series: np.ndarray,
    cfg: DetectionConfig,
) -> dict[str, Any]:
    """Build the support dictionary for a non-empty detection run.

    Parameters
    ----------
    search_series : numpy.ndarray
        SEARCH series.
    baseline_series : numpy.ndarray
        BASELINE series.
    ratio_series : numpy.ndarray
        SEARCH / BASELINE ratio series.
    cfg : DetectionConfig
        Active configuration.

    Returns
    -------
    dict
        Support arrays and metadata.
    """
    return {
        "search_series": search_series,
        "baseline_series": baseline_series,
        "local_baseline": baseline_series,
        "ratio_series": ratio_series,
        "rthreshold": baseline_series * cfg.ratio_threshold,
        "detector_config": cfg.to_metadata(),
        "candidate_indices": np.array([], dtype=int),
        "accepted_indices": np.array([], dtype=int),
        "peak_frequencies": np.array([], dtype=float),
        "peak_heights": np.array([], dtype=float),
    }


def _refine_peak_indices(indices: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Hill-climb each candidate index towards the nearest PSD local maximum.

    Parameters
    ----------
    indices : numpy.ndarray
        Raw candidate peak indices from the smoothed SEARCH series.
    psd : numpy.ndarray
        Original (unsmoothed) PSD for precise peak positioning.

    Returns
    -------
    numpy.ndarray
        Refined peak indices, sorted and deduplicated.
    """
    if indices.size == 0:
        return indices

    arr = np.asarray(psd)
    if arr.size == 0:
        return indices

    refined: list[int] = []
    upper = arr.size - 1

    for raw_idx in np.asarray(indices, dtype=int):
        idx = int(np.clip(raw_idx, 0, upper))
        left = max(0, idx - 3)
        right = min(upper, idx + 3)
        window = arr[left : right + 1]
        if window.size == 0:
            refined.append(idx)
            continue

        local = int(np.argmax(window)) + left
        best = local

        for _ in range(6):  # bounded hill-climb towards the nearest PSD maximum
            current = arr[best]
            left_idx = best - 1 if best > 0 else best
            right_idx = best + 1 if best < upper else best

            left_val = arr[left_idx] if left_idx < best else float("-inf")
            right_val = arr[right_idx] if right_idx > best else float("-inf")

            if left_val <= current and right_val <= current:
                break

            if right_val > left_val:
                if right_idx == best:
                    break
                best = right_idx
            else:
                if left_idx == best:
                    break
                best = left_idx

        refined.append(best)

    return np.array(sorted(set(refined)), dtype=int)
