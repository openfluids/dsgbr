"""Tests for the real-data benchmark helpers.

The pure helpers are tested unconditionally.  Anything needing the CWRU recordings
is skipped when the cache is empty, so CI stays hermetic and works offline; the
data is fetched on demand and never bundled.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.real.cwru import (
    CASES,
    SKF_6205_DE,
    envelope_spectrum,
    harmonic_recall,
    is_cached,
    load_drive_end,
)
from dsgbr import dsgbr_detector

requires_data = pytest.mark.skipif(
    not all(is_cached(case) for case in CASES),
    reason="CWRU recordings not cached; run `python -m benchmarks.real.compare` to fetch",
)


class TestHarmonicRecall:
    """Scoring helper, independent of any downloaded data."""

    def test_counts_matching_harmonics(self) -> None:
        peaks = np.array([10.0, 20.0, 30.0, 40.0])
        assert harmonic_recall(peaks, 10.0, n_harmonics=4) == 4

    def test_missing_harmonics_are_not_counted(self) -> None:
        peaks = np.array([10.0, 30.0])
        assert harmonic_recall(peaks, 10.0, n_harmonics=4) == 2

    def test_tolerance_is_relative(self) -> None:
        # 1% off is inside the default 1.5% tolerance, 5% off is not
        assert harmonic_recall(np.array([101.0]), 100.0, n_harmonics=1) == 1
        assert harmonic_recall(np.array([105.0]), 100.0, n_harmonics=1) == 0

    def test_degenerate_inputs(self) -> None:
        assert harmonic_recall(np.array([]), 10.0) == 0
        assert harmonic_recall(np.array([10.0]), 0.0) == 0

    def test_bearing_multipliers_are_ordered_as_expected(self) -> None:
        """Geometry fixes the ordering: cage < ball spin < outer race < inner race."""
        assert SKF_6205_DE["FTF"] < SKF_6205_DE["BSF"] < SKF_6205_DE["BPFO"] < SKF_6205_DE["BPFI"]


class TestEnvelopeSpectrum:
    """Preprocessing on a synthetic signal with a known modulation."""

    def test_recovers_a_known_modulation_frequency(self) -> None:
        """A 3 kHz carrier amplitude-modulated at 120 Hz must peak at 120 Hz."""
        fs, duration, modulation = 12_000.0, 4.0, 120.0
        t = np.arange(int(fs * duration)) / fs
        carrier = np.sin(2 * np.pi * 3000.0 * t)
        signal = (1.0 + 0.8 * np.sin(2 * np.pi * modulation * t)) * carrier

        freqs, power = envelope_spectrum(signal, fs=fs, nperseg=8192)

        assert freqs.size > 0
        peak_hz = freqs[int(np.argmax(power))]
        assert abs(peak_hz - modulation) / modulation < 0.02


@requires_data
class TestAgainstRealBearings:
    """End-to-end checks against geometry-derived truth."""

    def test_outer_race_fault_series_is_recovered(self) -> None:
        case = next(c for c in CASES if c.tag == "130")
        signal, shaft_hz = load_drive_end(case)
        freqs, power = envelope_spectrum(signal)

        peaks, _ = dsgbr_detector(freqs, power, case_info={"MP": 500})

        recall = harmonic_recall(peaks, SKF_6205_DE["BPFO"] * shaft_hz)
        assert recall >= 6, f"expected most BPFO harmonics, recovered {recall}/8"

    def test_shaft_speed_is_plausible(self) -> None:
        """Guards the loader: these recordings run near 1750-1800 rpm."""
        for case in CASES:
            _, shaft_hz = load_drive_end(case)
            assert 25.0 < shaft_hz < 32.0, f"{case.tag}: implausible {shaft_hz * 60:.0f} rpm"

    def test_the_faulted_bearing_scores_above_the_healthy_one(self) -> None:
        """The fault series must be more present in the faulted recording.

        This is the comparison that matters: an absolute recall number on one
        recording says little, because with many peaks a tolerance window can be
        satisfied by chance.
        """
        faulted = next(c for c in CASES if c.tag == "130")
        healthy = next(c for c in CASES if c.fault is None)

        scores = {}
        for case in (faulted, healthy):
            signal, shaft_hz = load_drive_end(case)
            freqs, power = envelope_spectrum(signal)
            peaks, heights = dsgbr_detector(freqs, power, case_info={"MP": 500})
            target = SKF_6205_DE["BPFO"] * shaft_hz
            # amplitude carried by the fault series, relative to all detected power
            mask = np.array(
                [
                    min(abs(p - k * target) / (k * target) for k in range(1, 9)) < 0.015
                    for p in peaks
                ]
            )
            scores[case.tag] = float(heights[mask].sum() / heights.sum()) if peaks.size else 0.0

        assert scores[faulted.tag] > scores[healthy.tag], (
            f"fault series should dominate in the faulted recording: {scores}"
        )
