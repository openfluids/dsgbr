"""Peak selection and frequency-matching utilities.

This module provides post-detection helpers for down-selecting peaks
across frequency bands and finding nearest detected frequencies.
"""

from __future__ import annotations

import numpy as np


def find_nearest_frequency(
    target_freq: float, frequencies: np.ndarray, heights: np.ndarray
) -> float:
    """Return the closest detected frequency to *target_freq*.

    Parameters
    ----------
    target_freq : float
        The target frequency to match against.
    frequencies : numpy.ndarray
        Array of detected peak frequencies.
    heights : numpy.ndarray
        Corresponding peak amplitudes (unused but kept for API consistency).

    Returns
    -------
    float
        The element of *frequencies* closest to *target_freq*, or ``0.0``
        if *target_freq* <= 0 or *frequencies* is empty.

    Examples
    --------
    >>> find_nearest_frequency(0.15, np.array([0.1, 0.2, 0.3]), np.array([1, 2, 3]))
    0.1
    """
    if target_freq <= 0 or len(frequencies) == 0:
        return 0.0
    idx = int(np.argmin(np.abs(np.asarray(frequencies) - float(target_freq))))
    return float(frequencies[idx])


def select_peaks_by_frequency_bands(
    peak_frequencies: np.ndarray,
    peak_heights: np.ndarray,
    *,
    max_peaks: int = 200,
    strategy: str = "proportional",
    n_bands: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Down-select peaks by distributing allowance across logarithmic bands.

    This function divides the detected frequency range into logarithmically
    spaced bands and allocates a peak quota to each band, ensuring broad
    spectral coverage even when peaks cluster in one region.

    Parameters
    ----------
    peak_frequencies : numpy.ndarray
        Detected peak frequencies (Hz).
    peak_heights : numpy.ndarray
        Corresponding peak amplitudes.
    max_peaks : int, optional
        Maximum total peaks to retain (default 200).
    strategy : {'proportional', 'equal'}, optional
        Band allocation strategy.  ``'proportional'`` gives more slots
        to bands with more candidates; ``'equal'`` splits evenly.
    n_bands : int, optional
        Number of logarithmic frequency bands (default 6).

    Returns
    -------
    selected_frequencies : numpy.ndarray
        Down-selected peak frequencies, sorted ascending.
    selected_heights : numpy.ndarray
        Corresponding amplitudes.

    Examples
    --------
    >>> f = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    >>> h = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> sf, sh = select_peaks_by_frequency_bands(f, h, max_peaks=3, n_bands=3)
    >>> sf.size <= 3
    True
    """
    peak_frequencies = np.asarray(peak_frequencies)
    peak_heights = np.asarray(peak_heights)

    if peak_frequencies.size <= max_peaks:
        return peak_frequencies, peak_heights

    freq_pos = peak_frequencies[peak_frequencies > 0]
    if freq_pos.size == 0:
        return np.array([]), np.array([])

    freq_min = float(freq_pos.min())
    freq_max = float(peak_frequencies.max())
    if freq_min > 1e-4:
        freq_min = 1e-4

    bands = max(1, int(n_bands))
    log_min, log_max = np.log10(freq_min), np.log10(freq_max)
    band_edges = np.logspace(log_min, log_max, bands + 1)

    candidates: list[np.ndarray] = []
    for i in range(bands):
        # The first band absorbs everything at or below its lower edge, including
        # non-positive frequencies, and the last band is closed on the right.  With
        # half-open bands on both ends a peak sitting exactly on freq_max, or below
        # the clamped lower edge, matches no band and is silently discarded.
        above = (
            np.ones(peak_frequencies.shape, dtype=bool)
            if i == 0
            else peak_frequencies >= band_edges[i]
        )
        below = (
            peak_frequencies <= band_edges[i + 1]
            if i == bands - 1
            else peak_frequencies < band_edges[i + 1]
        )
        candidates.append(np.where(above & below)[0])

    allotments = _compute_allotments(candidates, max_peaks, bands, strategy)

    selected_freqs: list[float] = []
    selected_heights: list[float] = []
    for i, cand in enumerate(candidates):
        n_select = min(allotments[i], cand.size)
        if n_select <= 0:
            continue
        idx = np.argsort(peak_heights[cand])[::-1][:n_select]
        selected_freqs.extend(peak_frequencies[cand][idx])
        selected_heights.extend(peak_heights[cand][idx])
    sel_f = np.asarray(selected_freqs)
    sel_h = np.asarray(selected_heights)

    if sel_f.size == 0:
        return sel_f, sel_h

    freq_order = np.argsort(sel_f)
    sel_f = sel_f[freq_order]
    sel_h = sel_h[freq_order]

    if sel_f.size > max_peaks:
        # Rank by amplitude, then restore ascending frequency order.  Slicing the
        # frequency-sorted array directly would keep the lowest frequencies rather
        # than the strongest peaks.
        keep = np.sort(np.argsort(sel_h)[::-1][:max_peaks])
        sel_f = sel_f[keep]
        sel_h = sel_h[keep]

    return sel_f, sel_h


def _compute_allotments(
    candidates: list[np.ndarray],
    max_peaks: int,
    bands: int,
    strategy: str,
) -> list[int]:
    """Compute per-band peak allotments.

    Parameters
    ----------
    candidates : list of numpy.ndarray
        Indices of candidate peaks in each band.
    max_peaks : int
        Total peak budget.
    bands : int
        Number of frequency bands.
    strategy : str
        ``'equal'`` or ``'proportional'``.

    Returns
    -------
    list of int
        Number of peaks allocated to each band.
    """
    allotments = [0] * bands
    if strategy == "equal":
        per_band = max_peaks // bands
        remainder = max_peaks % bands
        for i in range(bands):
            allotments[i] = per_band + (1 if i < remainder else 0)
    else:
        non_empty = [i for i, cand in enumerate(candidates) if cand.size > 0]
        # The one-slot-per-band floor must not exceed the budget: with more
        # non-empty bands than max_peaks it would over-allocate and force a
        # truncation that discards peaks the allotment had already promised.
        if len(non_empty) > max_peaks:
            non_empty = sorted(non_empty, key=lambda i: candidates[i].size, reverse=True)[
                :max_peaks
            ]
        remaining = max_peaks
        for i in non_empty:
            allotments[i] = 1
            remaining -= 1
        if remaining > 0:
            weights = np.array([cand.size for cand in candidates], dtype=float)
            total = weights.sum() if weights.sum() > 0 else 1.0
            fractional = remaining * (weights / total)
            integer = np.floor(fractional).astype(int)
            allotments = [a + b for a, b in zip(allotments, integer.tolist(), strict=False)]
            leftover = remaining - int(integer.sum())
            if leftover > 0:
                order = np.argsort(fractional - integer)[::-1]
                for idx in order[:leftover]:
                    allotments[idx] += 1
    return allotments
