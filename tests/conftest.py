"""Shared fixtures for DSGBR test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def flat_spectrum() -> tuple[np.ndarray, np.ndarray]:
    """Flat (uniform) PSD with 2048 points — no peaks expected."""
    frequencies = np.linspace(1e-3, 1.0, 2048)
    psd = np.ones_like(frequencies)
    return frequencies, psd


@pytest.fixture
def known_peaks_spectrum() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat PSD with 4 injected spikes at known indices.

    Returns
    -------
    frequencies : numpy.ndarray
    psd : numpy.ndarray
    spike_indices : numpy.ndarray
        Indices of the injected peaks.
    """
    frequencies = np.linspace(1e-3, 1.0, 2048)
    psd = np.ones_like(frequencies)
    spike_indices = np.array([110, 412, 1120, 1710], dtype=int)
    psd[spike_indices] = np.array([12.0, 16.0, 9.5, 11.5])
    return frequencies, psd, spike_indices


@pytest.fixture
def dense_comb_spectrum() -> tuple[np.ndarray, np.ndarray]:
    """~20 closely spaced peaks over a flat background."""
    frequencies = np.linspace(1e-3, 1.0, 4096)
    psd = np.ones_like(frequencies)
    # Place 20 peaks every ~200 bins
    for i in range(20):
        idx = 100 + i * 200
        if idx < len(psd):
            psd[idx] = 8.0 + np.random.default_rng(42).uniform(-2, 2)
    return frequencies, psd


@pytest.fixture
def noisy_spectrum() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Known peaks + colored noise background.

    Returns
    -------
    frequencies : numpy.ndarray
    psd : numpy.ndarray
    spike_indices : numpy.ndarray
    """
    rng = np.random.default_rng(123)
    frequencies = np.linspace(1e-3, 1.0, 2048)
    # 1/f-like background noise
    psd = 1.0 / (frequencies + 0.01) * 0.1 + rng.exponential(0.05, size=2048)
    spike_indices = np.array([200, 600, 1200, 1800], dtype=int)
    psd[spike_indices] += np.array([5.0, 8.0, 4.0, 6.0])
    return frequencies, psd, spike_indices


@pytest.fixture
def ulf_spectrum() -> tuple[np.ndarray, np.ndarray]:
    """Spectrum with ultra-low-frequency peaks (< 1e-3 Hz)."""
    frequencies = np.linspace(1e-5, 1e-2, 2048)
    psd = np.ones_like(frequencies) * 0.5
    # Inject sharp peaks in ULF region
    psd[50] = 10.0
    psd[100] = 8.0
    psd[150] = 6.0
    # Broader peak that should fail Q-factor test
    for offset in range(-10, 11):
        psd[300 + offset] = 4.0
    return frequencies, psd


@pytest.fixture
def empty_spectrum() -> tuple[np.ndarray, np.ndarray]:
    """Zero-length arrays."""
    return np.array([], dtype=float), np.array([], dtype=float)
