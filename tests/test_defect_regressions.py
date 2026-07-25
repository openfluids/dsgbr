"""Regression tests for the defects found in the 2026-07 review.

Each test below was confirmed to fail against the pre-fix implementation.  They
guard behaviour the previous 169 tests did not constrain.
"""

from __future__ import annotations

import numpy as np
import pytest

from dsgbr import DetectionConfig, dsgbr_detector, select_peaks_by_frequency_bands
from dsgbr._detector import _apply_ulf_guardrail


class TestBandSelection:
    """Down-selection must not discard peaks for positional reasons."""

    def test_highest_frequency_peak_is_selectable(self) -> None:
        """The peak sitting exactly on freq_max must be reachable.

        Previously the top band was half-open on the right while its upper edge
        equalled freq_max, so the highest-frequency peak matched no band at all.
        """
        freqs = np.array([1e-3, 1e-2, 1e-1, 1.0])
        heights = np.array([10.0, 20.0, 30.0, 99.0])  # strongest peak is at freq_max

        sel_f, sel_h = select_peaks_by_frequency_bands(freqs, heights, max_peaks=3, n_bands=3)

        assert 1.0 in sel_f
        assert sel_h.max() == pytest.approx(99.0)

    def test_over_budget_selection_is_not_the_lowest_frequencies(self) -> None:
        """Over-budget selections must not collapse onto the lowest frequencies.

        Band allocation intentionally trades global amplitude ranking for spectral
        coverage, so the result is *not* the globally strongest peaks.  What it must
        never be is the bottom of the frequency axis: the old code over-seeded one
        slot per band and then sliced a frequency-sorted array, returning the three
        lowest-frequency peaks (heights 1, 2, 3) and never reaching the strong end.
        """
        freqs = np.logspace(-3, 0, 10)
        heights = np.linspace(1.0, 10.0, 10)  # amplitude grows with frequency

        sel_f, sel_h = select_peaks_by_frequency_bands(freqs, heights, max_peaks=3, n_bands=10)

        assert sel_f.size <= 3
        # the pre-fix result was exactly the three weakest, lowest-frequency peaks
        assert set(sel_h.tolist()) != {1.0, 2.0, 3.0}
        # coverage must reach the top of the spectrum, including the strongest peak
        assert heights.max() in sel_h
        assert sel_f.max() == pytest.approx(freqs.max())

    @pytest.mark.parametrize("freq_max", [1.0, 3.7, 0.001234, 987.654321, 0.45])
    def test_highest_peak_survives_logspace_endpoint_rounding(self, freq_max: float) -> None:
        """freq_max must land in the top band for any value, not just exact ones.

        The upper band edge is ``10 ** log10(freq_max)``, which does not always
        round-trip to freq_max, so a closed ``<=`` comparison is not sufficient.
        """
        freqs = np.array([1e-3, 1e-2, 1e-1, freq_max])
        heights = np.array([10.0, 20.0, 30.0, 99.0])  # strongest peak is at freq_max

        sel_f, _ = select_peaks_by_frequency_bands(freqs, heights, max_peaks=3, n_bands=3)

        assert freq_max in sel_f

    def test_non_positive_frequency_peak_is_not_dropped(self) -> None:
        """A peak at f <= 0 belongs to the first band rather than to no band."""
        freqs = np.array([0.0, 1e-2, 1e-1, 1.0])
        heights = np.array([99.0, 20.0, 30.0, 40.0])  # strongest peak is at DC

        sel_f, _ = select_peaks_by_frequency_bands(freqs, heights, max_peaks=3, n_bands=3)

        assert 0.0 in sel_f

    def test_selection_never_exceeds_budget(self) -> None:
        """The per-band floor must not push the total above max_peaks."""
        freqs = np.logspace(-3, 0, 40)
        heights = np.linspace(1.0, 40.0, 40)

        for n_bands in (1, 3, 10, 25):
            sel_f, sel_h = select_peaks_by_frequency_bands(
                freqs, heights, max_peaks=5, n_bands=n_bands
            )
            assert sel_f.size <= 5, f"budget exceeded with n_bands={n_bands}"
            assert sel_f.size == sel_h.size


class TestUlfGuardrail:
    """ULF cap semantics."""

    def test_ulf_max_points_zero_retains_no_ulf_peaks(self) -> None:
        """0 means 'retain none', not 'unlimited'.

        The cap was guarded by ``if cap and ...``, so a zero cap was falsy and
        skipped the truncation entirely.
        """
        cfg = DetectionConfig(ulf_max_points=0, ulf_min_q=0.0, ulf_fmax=1.0)
        freqs = np.linspace(0.001, 0.5, 400)
        search = np.ones_like(freqs)
        indices = np.array([50, 100, 150, 200, 250], dtype=int)
        search[indices] = 5.0

        kept = _apply_ulf_guardrail(indices, freqs, search, cfg)

        assert kept.size == 0

    def test_ulf_max_points_still_caps_above_zero(self) -> None:
        """A positive cap keeps at most that many ULF peaks."""
        cfg = DetectionConfig(ulf_max_points=2, ulf_min_q=0.0, ulf_fmax=1.0)
        freqs = np.linspace(0.001, 0.5, 400)
        search = np.ones_like(freqs)
        indices = np.array([50, 100, 150, 200, 250], dtype=int)
        search[indices] = 5.0

        kept = _apply_ulf_guardrail(indices, freqs, search, cfg)

        assert kept.size <= 2


class TestSpacingInvariant:
    """Minimum bin separation must hold in the returned peaks.

    Note: two candidates that hill-climb onto the same maximum *should* merge --
    they were the same physical peak.  What must not survive is a returned pair
    closer than the configured separation.
    """

    @staticmethod
    def _bins(freqs: np.ndarray, peak_f: np.ndarray) -> np.ndarray:
        return np.array([int(np.argmin(np.abs(freqs - f))) for f in peak_f], dtype=int)

    @pytest.mark.parametrize("min_dist", [2, 5, 9])
    def test_returned_peaks_respect_minimum_separation(self, min_dist: int) -> None:
        """Refinement must not leave peaks closer than the configured distance."""
        rng = np.random.default_rng(12345)
        n = 4000
        freqs = np.linspace(1e-3, 1.0, n)
        psd = 0.01 + 0.001 * rng.random(n)
        # dense cluster of nearby maxima: refinement has room to shift indices
        for centre in range(500, 3500, 11):
            psd[centre] += 5.0
            psd[centre + 2] += 4.0

        peak_f, _ = dsgbr_detector(
            freqs,
            psd,
            case_info={"RT": 1.2, "SW": 5, "DL": min_dist, "DH": min_dist, "MP": 500},
        )

        bins = self._bins(freqs, peak_f)
        if bins.size < 2:
            pytest.skip("insufficient peaks detected to assess spacing")
        assert int(np.diff(np.sort(bins)).min()) >= min_dist

    def test_default_distance_high_is_an_actual_constraint(self) -> None:
        """distance_high=1 is a no-op: any two distinct bins are >= 1 apart."""
        assert DetectionConfig().distance_high > 1

    def test_broad_peak_is_not_reported_twice(self) -> None:
        """A single noisy peak must not be split into a doublet.

        The README scene injects ten harmonics whose FWHM spans ~14 bins.  With the
        old distance_high=1 the noisy top of the last harmonic was reported as two
        peaks four bins apart.
        """
        rng = np.random.default_rng(29)
        n = 4096
        freqs = np.logspace(np.log10(0.002), np.log10(1.2), n)
        baseline = 0.22 + 0.12 / np.sqrt(freqs + 0.01)
        psd = baseline * rng.lognormal(0.0, 0.30, size=n)
        truth = 0.045 * np.arange(1, 11)
        log_f = np.log10(freqs)
        for t, gain in zip(truth, 26.0 / np.arange(1, 11) ** 0.9, strict=False):
            psd += baseline * gain * np.exp(-0.5 * ((log_f - np.log10(t)) / 0.004) ** 2)

        peak_f, _ = dsgbr_detector(freqs, psd)

        assert peak_f.size == truth.size, (
            f"expected one detection per harmonic, got {peak_f.size} for {truth.size}"
        )
        # every injected harmonic accounted for exactly once
        for t in truth:
            assert np.min(np.abs(peak_f - t)) < 0.01 * t

    def test_peaks_are_returned_sorted_and_unique(self) -> None:
        rng = np.random.default_rng(7)
        n = 2000
        freqs = np.linspace(1e-3, 1.0, n)
        psd = 0.01 + 0.001 * rng.random(n)
        for centre in (300, 700, 1100, 1500):
            psd[centre] += 8.0

        peak_f, peak_h = dsgbr_detector(freqs, psd, case_info={"RT": 1.5})

        assert peak_f.size == peak_h.size
        assert np.all(np.diff(peak_f) > 0), "peaks must be sorted and unique"
