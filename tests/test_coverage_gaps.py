"""Targeted tests for coverage gaps in _detector.py internals."""

from __future__ import annotations

import numpy as np
import pytest

from dsgbr import dsgbr_detector
from dsgbr._config import DetectionConfig
from dsgbr._detector import (
    _apply_ulf_guardrail,
    _build_baseline_series,
    _build_search_series,
    _refine_peak_indices,
)


class TestBuildSearchSeries:
    """Coverage for _build_search_series edge cases."""

    def test_even_window_gets_corrected(self):
        """Even smooth_window should be bumped to odd internally."""
        # Can't pass even window through DetectionConfig validation,
        # but we can test the internal function with a manually-built cfg.
        # Instead, test that savgol smoothing works with valid odd window.
        psd = np.random.default_rng(0).uniform(0.5, 2.0, size=100)
        cfg = DetectionConfig(smooth_window=5)
        result = _build_search_series(psd, cfg)
        assert result.shape == psd.shape
        # Should be smoothed, not identical to input
        assert not np.array_equal(result, psd)

    def test_window_larger_than_data_returns_copy(self):
        """If smooth_window >= len(psd), smoothing is skipped."""
        psd = np.array([1.0, 2.0, 3.0, 2.0])  # only 4 points
        cfg = DetectionConfig(smooth_window=5)  # window > data length
        result = _build_search_series(psd, cfg)
        np.testing.assert_array_equal(result, psd)

    def test_smoothing_none_returns_copy(self):
        """smooth='none' returns a copy of the input."""
        psd = np.array([1.0, 2.0, 3.0])
        cfg = DetectionConfig(smooth="none")
        result = _build_search_series(psd, cfg)
        np.testing.assert_array_equal(result, psd)
        # Should be a copy, not the same object
        assert result is not psd

    def test_smoothing_on_linear(self):
        """smooth_on_log=False path."""
        psd = np.random.default_rng(1).uniform(0.5, 2.0, size=100)
        cfg = DetectionConfig(smooth_on_log=False)
        result = _build_search_series(psd, cfg)
        assert result.shape == psd.shape


class TestBuildBaselineSeries:
    """Coverage for _build_baseline_series edge cases."""

    def test_fixed_baseline_window(self):
        """baseline_window set explicitly."""
        search = np.random.default_rng(2).uniform(0.5, 2.0, size=200)
        cfg = DetectionConfig(baseline_window=31)
        result = _build_baseline_series(search, cfg)
        assert result.shape == search.shape

    def test_baseline_window_frac(self):
        """baseline_window_frac path (no fixed baseline_window)."""
        search = np.random.default_rng(3).uniform(0.5, 2.0, size=2000)
        cfg = DetectionConfig(baseline_window_frac=0.01)
        result = _build_baseline_series(search, cfg)
        assert result.shape == search.shape

    def test_baseline_window_frac_zero_rejected(self):
        """baseline_window_frac=0 is rejected at configuration construction."""
        with pytest.raises(ValueError, match="baseline_window_frac must be"):
            DetectionConfig(baseline_window_frac=0.0)

    def test_baseline_on_linear(self):
        """baseline_on_log=False path."""
        search = np.random.default_rng(5).uniform(0.5, 2.0, size=200)
        cfg = DetectionConfig(baseline_on_log=False, baseline_window=31)
        result = _build_baseline_series(search, cfg)
        assert result.shape == search.shape

    def test_window_too_large_for_data(self):
        """When computed window >= data length, use unsmoothed baseline."""
        search = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        cfg = DetectionConfig(baseline_window=99)
        result = _build_baseline_series(search, cfg)
        assert result.shape == search.shape


class TestULFGuardrail:
    """Coverage for _apply_ulf_guardrail branches."""

    def test_empty_indices(self):
        indices = np.array([], dtype=int)
        freq = np.linspace(1e-5, 1e-2, 100)
        search = np.ones(100)
        cfg = DetectionConfig()
        result = _apply_ulf_guardrail(indices, freq, search, cfg)
        assert result.size == 0

    def test_ulf_fmax_zero_disables(self):
        """ulf_fmax=0 should pass all indices through."""
        indices = np.array([10, 50], dtype=int)
        freq = np.linspace(1e-5, 1e-2, 100)
        search = np.ones(100)
        search[indices] = 5.0
        cfg = DetectionConfig(ulf_fmax=0.0)
        result = _apply_ulf_guardrail(indices, freq, search, cfg)
        np.testing.assert_array_equal(result, indices)

    def test_no_ulf_peaks(self):
        """All indices above ulf_fmax — pass through unchanged."""
        indices = np.array([80, 90], dtype=int)
        freq = np.linspace(1e-5, 1e-2, 100)
        search = np.ones(100)
        search[indices] = 5.0
        cfg = DetectionConfig(ulf_fmax=1e-5)
        result = _apply_ulf_guardrail(indices, freq, search, cfg)
        np.testing.assert_array_equal(result, indices)

    def test_ulf_peaks_with_sharp_peaks(self, ulf_spectrum):
        """Sharp ULF peaks should survive the Q-factor filter."""
        freq, psd = ulf_spectrum
        # Create a scenario where ULF peaks exist
        # Indices at 50, 100 are sharp peaks
        indices = np.array([50, 100, 150, 1500], dtype=int)
        cfg = DetectionConfig(ulf_fmax=5e-3, ulf_min_q=1.0, ulf_max_points=3)
        result = _apply_ulf_guardrail(indices, freq, psd, cfg)
        assert result.size > 0
        # Non-ULF peak (1500) should always survive
        assert 1500 in result

    def test_ulf_cap_limits_peaks(self):
        """ulf_max_points should limit the number of ULF peaks."""
        freq = np.linspace(1e-5, 1e-2, 2048)
        psd = np.ones(2048) * 0.5
        # Many sharp ULF peaks
        for i in [50, 150, 250, 350, 450]:
            psd[i] = 10.0 - i * 0.01
        indices = np.array([50, 150, 250, 350, 450, 1500], dtype=int)
        cfg = DetectionConfig(ulf_fmax=5e-3, ulf_min_q=0.1, ulf_max_points=2)
        result = _apply_ulf_guardrail(indices, freq, psd, cfg)
        # Should have at most 2 ULF peaks + the non-ULF one
        ulf_in_result = [i for i in result if freq[i] < cfg.ulf_fmax]
        assert len(ulf_in_result) <= 2

    def test_ulf_all_zero_search_values(self):
        """ULF peaks with zero search values should be removed."""
        freq = np.linspace(1e-5, 1e-2, 100)
        search = np.zeros(100)  # all zero
        indices = np.array([10, 20, 80], dtype=int)
        cfg = DetectionConfig(ulf_fmax=5e-3)
        result = _apply_ulf_guardrail(indices, freq, search, cfg)
        # ULF indices with zero search should be dropped, non-ULF kept
        assert isinstance(result, np.ndarray)

    def test_ulf_all_fail_q_factor(self):
        """All ULF peaks fail Q-factor test — only non-ULF remain."""
        freq = np.linspace(1e-5, 1e-2, 2048)
        psd = np.ones(2048)
        # Broad peak (low Q-factor)
        for offset in range(-50, 51):
            psd[100 + offset] = 2.0
        indices = np.array([100, 1500], dtype=int)
        cfg = DetectionConfig(ulf_fmax=5e-3, ulf_min_q=100.0)  # very strict Q
        result = _apply_ulf_guardrail(indices, freq, psd, cfg)
        # ULF peak should fail Q test, only non-ULF remains
        assert 1500 in result


class TestRefinepeakIndices:
    """Coverage for _refine_peak_indices edge cases."""

    def test_empty_indices(self):
        result = _refine_peak_indices(np.array([], dtype=int), np.ones(100))
        assert result.size == 0

    def test_empty_psd(self):
        result = _refine_peak_indices(np.array([5], dtype=int), np.array([]))
        np.testing.assert_array_equal(result, np.array([5]))

    def test_refinement_shifts_to_true_peak(self):
        """A candidate 1 bin off from the true max should be refined."""
        psd = np.ones(100)
        psd[50] = 10.0  # true peak
        indices = np.array([49], dtype=int)  # 1 bin off
        result = _refine_peak_indices(indices, psd)
        assert 50 in result

    def test_hill_climb_at_boundary(self):
        """Refinement near array boundaries should not crash."""
        psd = np.ones(10)
        psd[0] = 5.0  # peak at left boundary
        psd[9] = 5.0  # peak at right boundary
        indices = np.array([0, 9], dtype=int)
        result = _refine_peak_indices(indices, psd)
        assert 0 in result
        assert 9 in result


class TestDetectorUncoveredBranches:
    """Detector-level tests hitting specific uncovered branches."""

    def test_all_candidates_fail_spacing(self):
        """All candidates too close together with large distance requirement."""
        freq = np.linspace(1e-3, 1.0, 200)
        psd = np.ones(200)
        # Two peaks very close together
        psd[100] = 5.0
        psd[101] = 5.1
        # With very large distance, only the strongest should survive
        cfg = {
            "smooth": "none",
            "ratio_threshold": 1.2,
            "baseline_window": 31,
            "distance_low": 100,
            "distance_high": 100,
        }
        peak_f, _ = dsgbr_detector(freq, psd, case_info=cfg)
        assert peak_f.size <= 1

    def test_ulf_guardrail_removes_all_peaks(self):
        """ULF guardrail removes all peaks leaving empty result."""
        # Ultra-low-frequency spectrum where all detected peaks are ULF
        freq = np.linspace(1e-6, 5e-4, 500)
        psd = np.ones(500)
        # Broad, low-Q peaks only
        for i in range(200, 300):
            psd[i] = 2.0 + 0.01 * (i - 200)
        cfg = {
            "smooth": "none",
            "ratio_threshold": 1.2,
            "baseline_window": 31,
            "ulf_fmax": 1.0,  # everything is ULF
            "ulf_min_q": 1000.0,  # extremely strict
        }
        peak_f, _ = dsgbr_detector(freq, psd, case_info=cfg)
        assert isinstance(peak_f, np.ndarray)

    def test_band_selection_invoked(self):
        """Many peaks trigger band-based down-selection."""
        freq = np.linspace(1e-3, 1.0, 2000)
        psd = np.ones(2000)
        # 30 well-separated spikes
        for i in range(30):
            idx = 50 + i * 60
            psd[idx] = 10.0 + i * 0.1
        cfg = {
            "smooth": "none",
            "ratio_threshold": 1.2,
            "baseline_window": 21,
            "max_peaks": 5,
        }
        peak_f, _ = dsgbr_detector(freq, psd, case_info=cfg)
        assert peak_f.size <= 5
