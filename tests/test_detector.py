"""Tests for the core DSGBR detection pipeline."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from dsgbr import compute_support_series, dsgbr_detector
from dsgbr._config import DetectionConfig
from dsgbr._detector import _build_baseline_series


class TestEmptyAndFlat:
    """Behavior on degenerate inputs."""

    def test_empty_input(self, empty_spectrum):
        f, p = empty_spectrum
        peak_f, peak_h = dsgbr_detector(f, p)
        assert peak_f.size == 0
        assert peak_h.size == 0

    def test_empty_with_support(self, empty_spectrum):
        f, p = empty_spectrum
        peak_f, _peak_h, support = dsgbr_detector(f, p, return_support=True)
        assert peak_f.size == 0
        assert support["search_series"].size == 0
        assert support["detector_config"]["ratio_threshold"] == 3.3

    def test_flat_spectrum_no_peaks(self, flat_spectrum):
        f, p = flat_spectrum
        peak_f, _peak_h = dsgbr_detector(f, p)
        assert peak_f.size == 0

    def test_flat_spectrum_with_low_threshold(self, flat_spectrum):
        f, p = flat_spectrum
        peak_f, _ = dsgbr_detector(f, p, case_info={"RT": 1.0})
        # Even with RT=1.0, a perfectly flat spectrum has no local maxima
        # in the search series, so no peaks should be detected
        assert peak_f.size == 0


class TestKnownPeaks:
    """Detection of injected peaks."""

    def test_detects_all_known_peaks(self, known_peaks_spectrum):
        f, p, spike_indices = known_peaks_spectrum
        cfg = {"smooth": "none", "ratio_threshold": 1.2, "baseline_window": 61, "max_peaks": 10}
        peak_f, _ = dsgbr_detector(f, p, case_info=cfg)
        known_freqs = set(np.round(f[spike_indices], 12))
        detected_freqs = set(np.round(peak_f, 12))
        assert known_freqs.issubset(detected_freqs)

    def test_output_sorted_ascending(self, known_peaks_spectrum):
        f, p, _ = known_peaks_spectrum
        cfg = {"smooth": "none", "ratio_threshold": 1.2, "baseline_window": 61}
        peak_f, _ = dsgbr_detector(f, p, case_info=cfg)
        assert np.all(np.diff(peak_f) > 0)

    def test_support_dict_structure(self, known_peaks_spectrum):
        f, p, _ = known_peaks_spectrum
        cfg = {"smooth": "none", "ratio_threshold": 1.2, "baseline_window": 61}
        _, _, support = dsgbr_detector(f, p, case_info=cfg, return_support=True)
        required_keys = {
            "search_series",
            "baseline_series",
            "local_baseline",
            "ratio_series",
            "rthreshold",
            "detector_config",
            "candidate_indices",
            "accepted_indices",
            "peak_frequencies",
            "peak_heights",
        }
        assert required_keys.issubset(support.keys())
        assert support["accepted_indices"].dtype == int


class TestRatioThreshold:
    """Ratio threshold controls peak acceptance."""

    def test_higher_threshold_fewer_peaks(self, known_peaks_spectrum):
        f, p, _ = known_peaks_spectrum
        base_cfg = {"smooth": "none", "baseline_window": 61}

        _, _ = dsgbr_detector(f, p, case_info={**base_cfg, "RT": 1.2})
        low_f, _ = dsgbr_detector(f, p, case_info={**base_cfg, "RT": 1.2})
        high_f, _ = dsgbr_detector(f, p, case_info={**base_cfg, "RT": 5.0})
        assert high_f.size <= low_f.size

    def test_very_high_threshold_no_peaks(self, known_peaks_spectrum):
        f, p, _ = known_peaks_spectrum
        peak_f, _ = dsgbr_detector(f, p, case_info={"RT": 100.0})
        assert peak_f.size == 0


class TestSpacing:
    """Frequency-dependent spacing rules."""

    def test_switch_frequency_affects_spacing(self):
        """With large distance_low, low-freq peaks get filtered but high-freq survive."""
        frequencies = np.linspace(0.01, 1.0, 1000)
        psd = np.ones_like(frequencies)
        # Two peaks well-separated in high-freq region
        psd[700] = 10.0
        psd[900] = 10.0
        # With distance_high=1, both high-freq peaks should survive
        cfg_high = {
            "smooth": "none",
            "ratio_threshold": 1.2,
            "baseline_window": 61,
            "distance_high": 1,
            "distance_low": 300,
            "switch_frequency": 0.01,
        }
        peak_f_high, _ = dsgbr_detector(frequencies, psd, case_info=cfg_high)
        assert peak_f_high.size >= 2

        # With switch_frequency very high (all peaks treated as low-freq),
        # the large distance_low should reduce detections
        cfg_low = {
            "smooth": "none",
            "ratio_threshold": 1.2,
            "baseline_window": 61,
            "distance_high": 1,
            "distance_low": 300,
            "switch_frequency": 10.0,
        }
        peak_f_low, _ = dsgbr_detector(frequencies, psd, case_info=cfg_low)
        assert peak_f_low.size <= peak_f_high.size


class TestMaxPeaks:
    """max_peaks limit is respected."""

    def test_respects_max_peaks_limit(self):
        frequencies = np.linspace(1e-3, 1.0, 500)
        psd = np.ones_like(frequencies)
        spike_indices = np.array([20, 60, 110, 180, 250, 330, 420])
        psd[spike_indices] = np.array([12, 11, 13, 10, 9.5, 9.7, 9.8])
        peak_f, _ = dsgbr_detector(
            frequencies,
            psd,
            case_info={"smooth": "none", "RT": 1.2, "baseline_window": 61, "max_peaks": 3},
        )
        assert peak_f.size <= 3
        assert np.all(np.diff(peak_f) > 0)

    def test_max_peaks_one(self):
        frequencies = np.linspace(1e-3, 1.0, 500)
        psd = np.ones_like(frequencies)
        psd[100] = 20.0
        psd[300] = 15.0
        peak_f, _ = dsgbr_detector(
            frequencies,
            psd,
            case_info={"smooth": "none", "RT": 1.2, "baseline_window": 61, "max_peaks": 1},
        )
        assert peak_f.size <= 1


class TestDenseComb:
    """Dense spectral comb handling."""

    def test_dense_comb_detection(self, dense_comb_spectrum):
        f, p = dense_comb_spectrum
        cfg = {"smooth": "none", "ratio_threshold": 1.5, "baseline_window": 101, "max_peaks": 15}
        peak_f, _ = dsgbr_detector(f, p, case_info=cfg)
        assert peak_f.size <= 15
        assert peak_f.size > 0


class TestSmoothingModes:
    """Different smoothing configurations."""

    def test_smoothing_none(self, known_peaks_spectrum):
        f, p, _spike_indices = known_peaks_spectrum
        peak_f, _ = dsgbr_detector(
            f, p, case_info={"smooth": "none", "RT": 1.2, "baseline_window": 61}
        )
        assert peak_f.size >= 4

    def test_smoothing_savgol(self, known_peaks_spectrum):
        f, p, _ = known_peaks_spectrum
        peak_f, _ = dsgbr_detector(
            f, p, case_info={"smooth": "savgol", "SW": 5, "RT": 1.2, "baseline_window": 61}
        )
        assert peak_f.size > 0

    def test_smoothing_on_log_vs_linear(self, known_peaks_spectrum):
        f, p, _ = known_peaks_spectrum
        base = {"smooth": "savgol", "SW": 5, "RT": 1.5, "baseline_window": 61}
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            log_f, _ = dsgbr_detector(f, p, case_info={**base, "smooth_on_log": True})
            lin_f, _ = dsgbr_detector(f, p, case_info={**base, "smooth_on_log": False})
        # Both should detect peaks but may differ
        assert log_f.size > 0 or lin_f.size > 0
        assert np.all(np.isfinite(log_f))
        assert np.all(np.isfinite(lin_f))

    def test_linear_search_log_baseline_has_no_runtime_warnings(self, known_peaks_spectrum):
        f, p, _ = known_peaks_spectrum
        cfg = {"smooth": "savgol", "SW": 5, "smooth_on_log": False, "baseline_on_log": True}
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            peak_f, peak_h, support = dsgbr_detector(f, p, case_info=cfg, return_support=True)
        assert np.all(np.isfinite(peak_f))
        assert np.all(np.isfinite(peak_h))
        assert np.all(np.isfinite(support["baseline_series"]))
        assert np.all(np.isfinite(support["ratio_series"]))

    def test_log_baseline_detects_single_spike_among_zero_bins(self):
        frequencies = np.linspace(0.1, 1.1, 101)
        psd = np.zeros(101)
        psd[50] = 10.0

        peak_f, _peak_h, support = dsgbr_detector(
            frequencies,
            psd,
            case_info={"RT": 1e6, "baseline_window": 21},
            return_support=True,
        )

        assert frequencies[50] in peak_f
        assert support["ratio_series"][50] > 1e6

    def test_log_baseline_matches_rolling_median_for_all_positive_psd(self):
        psd = np.linspace(0.25, 2.0, 101)
        cfg = DetectionConfig.from_case_info({"baseline_on_log": True, "baseline_window": 21})
        baseline = _build_baseline_series(psd, cfg)
        win, pad = 21, 10
        padded = np.pad(np.log10(psd), (pad, pad), mode="edge")
        expected = np.power(
            10.0,
            np.array([np.median(padded[i : i + win]) for i in range(psd.size)]),
        )
        np.testing.assert_allclose(baseline, expected, rtol=1e-12)


class TestComputeSupport:
    """compute_support_series helper."""

    def test_consistent_shapes(self):
        f = np.linspace(1e-3, 1e-2, 100)
        p = np.random.default_rng(0).uniform(0.5, 2.0, size=100)
        support = compute_support_series(f, p)
        assert support["search_series"].shape == p.shape
        assert support["baseline_series"].shape == p.shape
        assert support["ratio_series"].shape == p.shape
        assert support["rthreshold"].shape == p.shape

    def test_detector_config_in_support(self):
        f = np.linspace(1e-3, 1.0, 100)
        p = np.ones_like(f)
        support = compute_support_series(f, p, case_info={"RT": 2.5})
        assert support["detector_config"]["ratio_threshold"] == 2.5


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------

_RNG_SEEDS = [0, 42, 99, 12345, 271828]
_SIZES = [10, 50, 100, 300, 500]


class TestPropertyBased:
    """Property-based tests using deterministic random PSD arrays."""

    @pytest.mark.parametrize("seed,size", zip(_RNG_SEEDS, _SIZES, strict=True))
    def test_detector_never_crashes(self, seed, size):
        """Detector should never raise on valid positive inputs."""
        rng = np.random.default_rng(seed)
        psd = rng.uniform(1e-10, 1e6, size=size)
        f = np.linspace(1e-3, 1.0, size)
        peak_f, peak_h = dsgbr_detector(f, psd)
        assert isinstance(peak_f, np.ndarray)
        assert isinstance(peak_h, np.ndarray)
        assert peak_f.shape == peak_h.shape

    @pytest.mark.parametrize("seed,size", zip(_RNG_SEEDS, _SIZES, strict=True))
    def test_max_peaks_always_honored(self, seed, size):
        """Output should never exceed max_peaks."""
        rng = np.random.default_rng(seed)
        psd = rng.uniform(1e-10, 1e6, size=size)
        f = np.linspace(1e-3, 1.0, size)
        max_peaks = 5
        peak_f, _ = dsgbr_detector(f, psd, case_info={"max_peaks": max_peaks})
        assert peak_f.size <= max_peaks

    @pytest.mark.parametrize("seed,size", zip(_RNG_SEEDS, _SIZES, strict=True))
    def test_output_always_sorted(self, seed, size):
        """peak_f should always be sorted ascending."""
        rng = np.random.default_rng(seed)
        psd = rng.uniform(1e-10, 1e6, size=size)
        f = np.linspace(1e-3, 1.0, size)
        peak_f, _ = dsgbr_detector(f, psd)
        if peak_f.size > 1:
            assert np.all(np.diff(peak_f) > 0)
