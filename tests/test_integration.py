"""Integration tests: end-to-end synthetic signal detection."""

from __future__ import annotations

import numpy as np

from dsgbr import compute_support_series, dsgbr_detector


class TestSyntheticSignal:
    """End-to-end detection on synthetic quasi-periodic signals."""

    def test_sine_superposition(self):
        """Three sines at known frequencies should produce detectable peaks."""
        # Use a moderate-length signal so the FFT output is manageable
        # for the baseline estimation (not thousands of points).
        dt = 0.005
        t = np.arange(0, 20, dt)
        signal = (
            1.0 * np.sin(2 * np.pi * 2.0 * t)
            + 0.5 * np.sin(2 * np.pi * 5.0 * t)
            + 0.3 * np.sin(2 * np.pi * 10.0 * t)
        )
        # Compute PSD via FFT
        n = len(signal)
        freqs = np.fft.rfftfreq(n, d=dt)
        psd = np.abs(np.fft.rfft(signal)) ** 2 / n

        # Use no smoothing to preserve the sharp FFT peaks
        cfg = {"smooth": "none", "RT": 1.5, "baseline_window": 51, "max_peaks": 50}
        peak_f, _ = dsgbr_detector(freqs, psd, case_info=cfg)

        assert peak_f.size >= 3
        # Check that 2, 5, 10 Hz are approximately detected
        # (FFT bin resolution = 1/20 = 0.05 Hz)
        for target in [2.0, 5.0, 10.0]:
            diffs = np.abs(peak_f - target)
            assert diffs.min() < 0.1, f"Peak near {target} Hz not found in {peak_f}"

    def test_support_consistency(self):
        """Support dict should be consistent between direct and helper calls."""
        f = np.linspace(1e-3, 1.0, 1000)
        psd = np.ones_like(f)
        psd[200] = 10.0
        psd[600] = 8.0

        cfg = {"smooth": "none", "RT": 1.3, "baseline_window": 61}

        _, _, direct_support = dsgbr_detector(f, psd, case_info=cfg, return_support=True)
        helper_support = compute_support_series(f, psd, case_info=cfg)

        np.testing.assert_array_equal(
            direct_support["search_series"], helper_support["search_series"]
        )
        np.testing.assert_array_equal(
            direct_support["baseline_series"], helper_support["baseline_series"]
        )
        np.testing.assert_array_equal(
            direct_support["peak_frequencies"], helper_support["peak_frequencies"]
        )

    def test_noisy_signal_detection(self, noisy_spectrum):
        """Should detect at least some peaks even with colored noise."""
        f, p, _spike_indices = noisy_spectrum
        cfg = {"smooth": "savgol", "SW": 7, "RT": 2.0, "baseline_window": 101}
        peak_f, _ = dsgbr_detector(f, p, case_info=cfg)
        # Should find at least one of the 4 injected peaks
        assert peak_f.size > 0
