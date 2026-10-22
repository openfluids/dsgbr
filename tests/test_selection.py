"""Tests for peak selection and frequency-matching utilities."""

from __future__ import annotations

import numpy as np

from dsgbr import find_nearest_frequency, select_peaks_by_frequency_bands


class TestSelectPeaksByFrequencyBands:
    """Band-based peak down-selection."""

    def test_passthrough_when_under_limit(self):
        f = np.array([0.01, 0.05, 0.1])
        h = np.array([1.0, 2.0, 3.0])
        sf, sh = select_peaks_by_frequency_bands(f, h, max_peaks=10)
        np.testing.assert_array_equal(sf, f)
        np.testing.assert_array_equal(sh, h)

    def test_reduces_to_max_peaks(self):
        f = np.linspace(0.01, 1.0, 50)
        h = np.random.default_rng(42).uniform(1, 10, size=50)
        sf, _sh = select_peaks_by_frequency_bands(f, h, max_peaks=10, n_bands=5)
        assert sf.size <= 10

    def test_output_sorted_ascending(self):
        f = np.array([1e-4, 3e-4, 1e-3, 2.3e-3, 8e-3, 1.8e-2, 3.4e-2, 6e-2, 0.11, 0.28, 0.7])
        h = np.array([1.0, 3.0, 0.8, 2.5, 0.7, 4.1, 2.7, 1.2, 3.3, 5.1, 2.2])
        sf, _ = select_peaks_by_frequency_bands(f, h, max_peaks=4, n_bands=3)
        assert np.all(np.diff(sf) > 0)

    def test_equal_strategy(self):
        f = np.array([1e-4, 3e-4, 1e-3, 2.3e-3, 8e-3, 1.8e-2, 3.4e-2, 6e-2, 0.11, 0.28, 0.7])
        h = np.array([1.0, 3.0, 0.8, 2.5, 0.7, 4.1, 2.7, 1.2, 3.3, 5.1, 2.2])
        sf, sh = select_peaks_by_frequency_bands(f, h, max_peaks=4, strategy="equal", n_bands=3)
        assert sf.size <= 4
        assert sf.size == sh.size

    def test_proportional_strategy(self):
        f = np.array([1e-4, 3e-4, 1e-3, 2.3e-3, 8e-3, 1.8e-2, 3.4e-2, 6e-2, 0.11, 0.28, 0.7])
        h = np.array([1.0, 3.0, 0.8, 2.5, 0.7, 4.1, 2.7, 1.2, 3.3, 5.1, 2.2])
        sf, _ = select_peaks_by_frequency_bands(
            f,
            h,
            max_peaks=4,
            strategy="proportional",
            n_bands=3,
        )
        assert sf.size <= 4

    def test_deterministic(self):
        f = np.linspace(0.01, 1.0, 100)
        h = np.random.default_rng(99).uniform(1, 10, size=100)
        sf1, sh1 = select_peaks_by_frequency_bands(f, h, max_peaks=10, n_bands=5)
        sf2, sh2 = select_peaks_by_frequency_bands(f, h, max_peaks=10, n_bands=5)
        np.testing.assert_array_equal(sf1, sf2)
        np.testing.assert_array_equal(sh1, sh2)

    def test_single_band(self):
        f = np.linspace(0.01, 1.0, 50)
        h = np.random.default_rng(42).uniform(1, 10, size=50)
        sf, _ = select_peaks_by_frequency_bands(f, h, max_peaks=5, n_bands=1)
        assert sf.size <= 5

    def test_empty_input(self):
        f = np.array([])
        h = np.array([])
        sf, sh = select_peaks_by_frequency_bands(f, h, max_peaks=5)
        assert sf.size == 0
        assert sh.size == 0

    def test_all_zero_frequencies(self):
        f = np.zeros(5)
        h = np.ones(5)
        sf, _sh = select_peaks_by_frequency_bands(f, h, max_peaks=2)
        # With all-zero frequencies, positive filter finds nothing
        assert sf.size == 0


class TestFindNearestFrequency:
    """Closest-frequency lookup utility."""

    def test_exact_match(self):
        f = np.array([0.1, 0.2, 0.3])
        h = np.array([1, 2, 3])
        assert find_nearest_frequency(0.2, f, h) == 0.2

    def test_between_values(self):
        f = np.array([0.1, 0.2, 0.3])
        h = np.array([1, 2, 3])
        result = find_nearest_frequency(0.15, f, h)
        assert result == 0.1  # closer to 0.1

    def test_negative_target(self):
        f = np.array([0.1, 0.2])
        h = np.array([1, 2])
        assert find_nearest_frequency(-1.0, f, h) == 0.0

    def test_zero_target(self):
        f = np.array([0.1, 0.2])
        h = np.array([1, 2])
        assert find_nearest_frequency(0.0, f, h) == 0.0

    def test_empty_frequencies(self):
        f = np.array([])
        h = np.array([])
        assert find_nearest_frequency(0.5, f, h) == 0.0

    def test_single_frequency(self):
        f = np.array([0.42])
        h = np.array([1.0])
        assert find_nearest_frequency(999.0, f, h) == 0.42
