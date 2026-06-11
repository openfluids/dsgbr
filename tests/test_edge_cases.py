"""Edge case tests for extreme and degenerate inputs."""

from __future__ import annotations

import numpy as np
import pytest

from dsgbr import compute_support_series, detect_peaks_case_adaptive, dsgbr_detector


class TestTinyInputs:
    """Very short arrays."""

    def test_single_point(self):
        f = np.array([0.5])
        p = np.array([1.0])
        peak_f, peak_h = dsgbr_detector(f, p)
        assert peak_f.size == 0
        assert peak_h.size == 0

    def test_two_points(self):
        f = np.array([0.1, 0.2])
        p = np.array([1.0, 5.0])
        peak_f, peak_h = dsgbr_detector(f, p)
        assert isinstance(peak_f, np.ndarray)
        assert isinstance(peak_h, np.ndarray)

    def test_three_points_with_peak(self):
        f = np.array([0.1, 0.2, 0.3])
        p = np.array([1.0, 10.0, 1.0])
        # Smoothing window = 3 equals data length, so smoothing may not apply
        peak_f, _ = dsgbr_detector(f, p, case_info={"smooth": "none", "RT": 1.2})
        assert isinstance(peak_f, np.ndarray)


class TestExtremeValues:
    """NaN, Inf, zeros, and large values."""

    def test_all_zeros(self):
        f = np.linspace(1e-3, 1.0, 100)
        p = np.zeros(100)
        peak_f, _ = dsgbr_detector(f, p)
        assert isinstance(peak_f, np.ndarray)

    def test_very_large_values(self):
        f = np.linspace(1e-3, 1.0, 200)
        p = np.ones(200) * 1e15
        p[100] = 1e20
        peak_f, _ = dsgbr_detector(
            f, p, case_info={"smooth": "none", "RT": 1.2, "baseline_window": 31}
        )
        assert isinstance(peak_f, np.ndarray)

    def test_near_epsilon_psd(self):
        f = np.linspace(1e-3, 1.0, 200)
        p = np.ones(200) * 1e-300
        p[100] = 1e-290
        peak_f, _ = dsgbr_detector(f, p)
        assert isinstance(peak_f, np.ndarray)

    def test_nan_in_psd(self):
        f = np.linspace(1e-3, 1.0, 200)
        p = np.ones(200)
        p[50] = np.nan
        with pytest.raises(ValueError, match="psd contains 1 non-finite values"):
            dsgbr_detector(f, p)

    def test_inf_in_psd(self):
        f = np.linspace(1e-3, 1.0, 200)
        p = np.ones(200)
        p[50] = np.inf
        with pytest.raises(ValueError, match="psd contains 1 non-finite values"):
            dsgbr_detector(f, p)


class TestInvalidInputs:
    """Invalid detector inputs raise actionable errors."""

    def test_mismatched_lengths_reports_both_lengths(self):
        f = np.linspace(1e-3, 1.0, 5)
        p = np.ones(4)
        with pytest.raises(ValueError, match=r"frequencies length=5, psd length=4"):
            dsgbr_detector(f, p)

    def test_non_1d_frequencies(self):
        f = np.array([[0.1, 0.2], [0.3, 0.4]])
        p = np.ones(4)
        with pytest.raises(ValueError, match=r"frequencies ndim=2, psd ndim=1"):
            dsgbr_detector(f, p)

    def test_non_1d_psd(self):
        f = np.linspace(0.1, 0.4, 4)
        p = np.ones((2, 2))
        with pytest.raises(ValueError, match=r"frequencies ndim=1, psd ndim=2"):
            dsgbr_detector(f, p)

    def test_non_monotonic_frequencies(self):
        f = np.array([0.1, 0.3, 0.2, 0.4])
        p = np.ones(4)
        with pytest.raises(ValueError, match="frequencies must be strictly increasing"):
            dsgbr_detector(f, p)

    def test_non_finite_frequencies_reports_count(self):
        f = np.array([0.1, np.nan, np.inf, 0.4])
        p = np.ones(4)
        with pytest.raises(ValueError, match="frequencies contains 2 non-finite values"):
            dsgbr_detector(f, p)

    def test_non_finite_psd_reports_count(self):
        f = np.linspace(0.1, 0.4, 4)
        p = np.array([1.0, np.nan, np.inf, 1.0])
        with pytest.raises(ValueError, match="psd contains 2 non-finite values"):
            dsgbr_detector(f, p)

    def test_negative_psd(self):
        f = np.linspace(0.1, 0.4, 4)
        p = np.array([1.0, 0.0, -0.1, 1.0])
        with pytest.raises(ValueError, match="psd must be nonnegative"):
            dsgbr_detector(f, p)

    def test_compute_support_series_inherits_validation(self):
        f = np.array([0.1, 0.2, 0.2])
        p = np.ones(3)
        with pytest.raises(ValueError, match="frequencies must be strictly increasing"):
            compute_support_series(f, p)

    def test_detect_peaks_case_adaptive_inherits_validation(self):
        f = np.linspace(0.1, 0.4, 4)
        p = np.array([1.0, -0.1, 1.0, 1.0])
        with pytest.raises(ValueError, match="psd must be nonnegative"):
            detect_peaks_case_adaptive(f, p)


class TestNoSmoothing:
    """Smoothing disabled edge cases."""

    def test_smooth_none_string(self):
        f = np.linspace(1e-3, 1.0, 100)
        p = np.ones(100)
        p[50] = 5.0
        peak_f, _ = dsgbr_detector(f, p, case_info={"smooth": "none", "RT": 1.2})
        assert isinstance(peak_f, np.ndarray)

    def test_smooth_empty_string(self):
        f = np.linspace(1e-3, 1.0, 100)
        p = np.ones(100)
        p[50] = 5.0
        peak_f, _ = dsgbr_detector(f, p, case_info={"smooth": ""})
        assert isinstance(peak_f, np.ndarray)
