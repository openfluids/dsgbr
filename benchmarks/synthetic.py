"""Synthetic power spectral density generator with explicit ground truth.

The generator models spectra as a smooth power-law baseline plus Lorentzian
peaks whose amplitudes are specified relative to the local baseline. Optional
multiplicative chi-squared scatter approximates Welch-averaged periodograms.
All randomness is supplied by an explicit :class:`numpy.random.Generator` so
callers control reproducibility without global seeding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
GridKind = Literal["log", "linear"]


@dataclass(frozen=True)
class PeakSpec:
    """Parameters for one injected Lorentzian peak.

    Parameters
    ----------
    frequency:
        Center frequency of the peak in hertz.
    amplitude_ratio:
        Peak height divided by the smooth baseline at ``frequency``.
    q_factor:
        Quality factor controlling width as ``frequency / q_factor``.
    """

    frequency: float
    amplitude_ratio: float
    q_factor: float


@dataclass(frozen=True)
class BaselineSpec:
    """Power-law baseline parameters."""

    reference_frequency: float = 1.0
    reference_power: float = 1.0
    alpha: float = 5.0 / 3.0
    plateau_frequency: float | None = None


@dataclass(frozen=True)
class NoiseSpec:
    """Welch-like multiplicative noise parameters."""

    ndof: float = 8.0
    enabled: bool = True


class SpectrumTruth(TypedDict):
    """Structured record of injected spectrum components."""

    peak_frequencies: FloatArray
    amplitude_ratios: FloatArray
    q_factors: FloatArray
    baseline: FloatArray


class Scenario(TypedDict):
    """Named arguments for :func:`make_spectrum`."""

    n_bins: int
    f_min: float
    f_max: float
    peaks: tuple[PeakSpec, ...]
    baseline: BaselineSpec
    noise: NoiseSpec
    grid: GridKind


def _coerce_peak(peak: PeakSpec | Sequence[float]) -> PeakSpec:
    if isinstance(peak, PeakSpec):
        return peak
    frequency, amplitude_ratio, q_factor = peak
    return PeakSpec(float(frequency), float(amplitude_ratio), float(q_factor))


def _baseline(frequencies: FloatArray, spec: BaselineSpec) -> FloatArray:
    if spec.reference_frequency <= 0.0:
        msg = "baseline reference_frequency must be positive"
        raise ValueError(msg)
    if spec.reference_power <= 0.0:
        msg = "baseline reference_power must be positive"
        raise ValueError(msg)

    effective_frequency = frequencies
    if spec.plateau_frequency is not None:
        if spec.plateau_frequency <= 0.0:
            msg = "baseline plateau_frequency must be positive when provided"
            raise ValueError(msg)
        effective_frequency = np.maximum(frequencies, spec.plateau_frequency)
    return spec.reference_power * (effective_frequency / spec.reference_frequency) ** (-spec.alpha)


def make_spectrum(
    n_bins: int,
    f_min: float,
    f_max: float,
    *,
    peaks: Sequence[PeakSpec | Sequence[float]],
    baseline: BaselineSpec | None = None,
    noise: NoiseSpec | None = None,
    rng: np.random.Generator,
    grid: GridKind = "log",
) -> tuple[FloatArray, FloatArray, SpectrumTruth]:
    """Generate a synthetic PSD and structured truth record.

    Parameters
    ----------
    n_bins:
        Number of frequency bins.
    f_min, f_max:
        Positive frequency range, inclusive.
    peaks:
        Sequence of ``(f0, amplitude_ratio, q_factor)`` triples or
        :class:`PeakSpec` instances. Peak amplitudes are relative to the smooth
        baseline at ``f0``.
    baseline:
        Power-law baseline parameters. Defaults to a Kolmogorov-like ``f^-5/3``
        slope.
    noise:
        Welch-like multiplicative chi-squared noise. Pass ``NoiseSpec(enabled=False)``
        for a noise-free PSD.
    rng:
        Explicit NumPy random generator used for all stochastic draws.
    grid:
        ``"log"`` (default) or ``"linear"`` frequency spacing.

    Returns
    -------
    frequencies, psd, truth:
        Frequency grid, synthetic PSD, and injected peak/baseline truth.
    """
    if n_bins < 2:
        msg = "n_bins must be at least 2"
        raise ValueError(msg)
    if f_min <= 0.0 or f_max <= f_min:
        msg = "frequency range must satisfy 0 < f_min < f_max"
        raise ValueError(msg)

    if grid == "log":
        frequencies = np.geomspace(f_min, f_max, n_bins, dtype=np.float64)
    elif grid == "linear":
        frequencies = np.linspace(f_min, f_max, n_bins, dtype=np.float64)
    else:
        msg = "grid must be 'log' or 'linear'"
        raise ValueError(msg)

    baseline_spec = baseline if baseline is not None else BaselineSpec()
    noise_spec = noise if noise is not None else NoiseSpec()
    smooth_baseline = _baseline(frequencies, baseline_spec)
    psd = smooth_baseline.copy()

    peak_specs = tuple(_coerce_peak(peak) for peak in peaks)
    for peak in peak_specs:
        if peak.frequency <= 0.0 or peak.amplitude_ratio < 0.0 or peak.q_factor <= 0.0:
            msg = "peaks must have positive frequency/q_factor and non-negative amplitude_ratio"
            raise ValueError(msg)
        baseline_at_peak = _baseline(np.array([peak.frequency], dtype=np.float64), baseline_spec)[0]
        gamma = peak.frequency / peak.q_factor
        lorentzian = 1.0 / (1.0 + (2.0 * (frequencies - peak.frequency) / gamma) ** 2)
        psd += peak.amplitude_ratio * baseline_at_peak * lorentzian

    if noise_spec.enabled:
        if noise_spec.ndof <= 0.0:
            msg = "noise ndof must be positive"
            raise ValueError(msg)
        psd *= rng.chisquare(2.0 * noise_spec.ndof, size=n_bins) / (2.0 * noise_spec.ndof)

    truth: SpectrumTruth = {
        "peak_frequencies": np.array([peak.frequency for peak in peak_specs], dtype=np.float64),
        "amplitude_ratios": np.array(
            [peak.amplitude_ratio for peak in peak_specs], dtype=np.float64
        ),
        "q_factors": np.array([peak.q_factor for peak in peak_specs], dtype=np.float64),
        "baseline": smooth_baseline,
    }
    return frequencies, psd, truth


def scenarios() -> Mapping[str, Scenario]:
    """Return canonical synthetic benchmark scenarios."""
    return {
        "clean_tones": {
            "n_bins": 1024,
            "f_min": 1.0,
            "f_max": 1_000.0,
            "peaks": (PeakSpec(18.0, 4.0, 35.0), PeakSpec(140.0, 3.0, 50.0)),
            "baseline": BaselineSpec(reference_frequency=1.0, reference_power=1.0),
            "noise": NoiseSpec(enabled=False),
            "grid": "log",
        },
        "dense_lowfreq": {
            "n_bins": 2048,
            "f_min": 0.5,
            "f_max": 500.0,
            "peaks": (
                PeakSpec(3.0, 2.5, 20.0),
                PeakSpec(4.2, 2.0, 25.0),
                PeakSpec(7.5, 3.0, 30.0),
            ),
            "baseline": BaselineSpec(
                reference_frequency=1.0, reference_power=2.0, plateau_frequency=1.0
            ),
            "noise": NoiseSpec(ndof=12.0),
            "grid": "log",
        },
        "steep_slope": {
            "n_bins": 1536,
            "f_min": 1.0,
            "f_max": 2_000.0,
            "peaks": (PeakSpec(60.0, 3.5, 40.0), PeakSpec(700.0, 5.0, 70.0)),
            "baseline": BaselineSpec(reference_frequency=1.0, reference_power=1.5, alpha=2.2),
            "noise": NoiseSpec(ndof=16.0),
            "grid": "log",
        },
        "noisy_welch": {
            "n_bins": 1024,
            "f_min": 2.0,
            "f_max": 800.0,
            "peaks": (PeakSpec(45.0, 2.5, 25.0), PeakSpec(300.0, 4.0, 45.0)),
            "baseline": BaselineSpec(reference_frequency=2.0, reference_power=1.0),
            "noise": NoiseSpec(ndof=4.0),
            "grid": "log",
        },
        "no_peaks": {
            "n_bins": 512,
            "f_min": 1.0,
            "f_max": 300.0,
            "peaks": (),
            "baseline": BaselineSpec(reference_frequency=1.0, reference_power=1.0),
            "noise": NoiseSpec(ndof=8.0),
            "grid": "log",
        },
    }
