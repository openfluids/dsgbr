"""DSGBR against tuned ``scipy.signal.find_peaks`` on real bearing envelopes.

Run with::

    uv run --extra tests python -m benchmarks.real.compare

Both detectors see the identical envelope spectrum, so the comparison isolates peak
selection.  ``find_peaks`` is given the best prominence found by a sweep on the same
spectrum, which flatters it -- it is tuned per case, DSGBR is not.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
from scipy.signal import find_peaks

from benchmarks.real.cwru import (
    CASES,
    SKF_6205_DE,
    Case,
    envelope_spectrum,
    harmonic_recall,
    load_drive_end,
)
from dsgbr import dsgbr_detector


def _best_find_peaks(
    freqs: np.ndarray, power: np.ndarray, fundamental: float, n_harmonics: int
) -> tuple[np.ndarray, float]:
    """Sweep prominence and keep whichever value recovers the most harmonics."""
    best_peaks: np.ndarray = np.array([])
    best_score, best_prominence = -1, float("nan")
    reference = float(np.median(power))
    for scale in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        prominence = reference * scale
        idx, _ = find_peaks(power, prominence=prominence)
        score = harmonic_recall(freqs[idx], fundamental, n_harmonics=n_harmonics)
        if score > best_score:
            best_peaks, best_score, best_prominence = freqs[idx], score, prominence
    return best_peaks, best_prominence


def evaluate(case: Case, *, case_info: dict[str, object] | None = None) -> dict[str, object]:
    """Detect on one recording and score both detectors against geometry."""
    signal, shaft_hz = load_drive_end(case)
    freqs, power = envelope_spectrum(signal)

    peaks, _ = dsgbr_detector(freqs, power, case_info=case_info)

    fault_hz = SKF_6205_DE[case.fault] * shaft_hz if case.fault else 0.0
    row: dict[str, object] = {
        "tag": case.tag,
        "description": case.description,
        "shaft_hz": shaft_hz,
        "n_peaks": int(peaks.size),
        "shaft_recall": harmonic_recall(peaks, shaft_hz),
        "fault": case.fault or "-",
        "fault_recall": harmonic_recall(peaks, fault_hz) if case.fault else 0,
    }

    if case.fault:
        scipy_peaks, prominence = _best_find_peaks(freqs, power, fault_hz, 8)
        row["scipy_fault_recall"] = harmonic_recall(scipy_peaks, fault_hz)
        row["scipy_n_peaks"] = int(scipy_peaks.size)
        row["scipy_prominence"] = prominence
    else:
        # No fault seeded: any recovered "fault" series would be a false alarm, so
        # score every bearing series and report the worst offender.
        row["false_alarm"] = max(
            harmonic_recall(peaks, mult * shaft_hz, n_harmonics=5)
            for mult in SKF_6205_DE.values()
        )
    return row


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-info",
        default=None,
        help='JSON detector overrides, e.g. \'{"DH": 1, "MP": 500}\'',
    )
    args = parser.parse_args(argv)

    case_info = None
    if args.case_info:
        import json

        case_info = json.loads(args.case_info)

    print(f"{'case':>5}  {'description':<28} {'peaks':>6} {'fault':>5} {'recall':>7} {'shaft':>7}")
    print("-" * 74)
    for case in CASES:
        row = evaluate(case, case_info=case_info)
        fault_col = (
            f"{row['fault_recall']}/8" if case.fault else f"FA {row['false_alarm']}/5"
        )
        print(
            f"{row['tag']:>5}  {row['description']:<28} {row['n_peaks']:>6} "
            f"{row['fault']:>5} {fault_col:>7} {row['shaft_recall']}/8"
        )
        if case.fault:
            print(
                f"{'':>5}  {'  tuned find_peaks':<28} {row['scipy_n_peaks']:>6} "
                f"{row['fault']:>5} {row['scipy_fault_recall']}/8"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
