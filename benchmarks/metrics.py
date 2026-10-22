"""Peak matching and detection scores for synthetic spectra.

Matching uses a greedy nearest-neighbour policy within relative frequency
tolerance. Each truth peak can be matched at most once; duplicate detections
near the same truth peak become false positives. This simple policy is
adequate for the benchmark densities and keeps the score easy to audit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MatchResult:
    """Greedy peak-matching result."""

    matches: tuple[tuple[int, int], ...]
    tp: int
    fp: int
    fn: int


@dataclass(frozen=True)
class DetectionScores:
    """Precision, recall, F1, and confusion counts."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def _as_float_array(values: Sequence[float] | FloatArray) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def match_peaks(
    detected_f: Sequence[float] | FloatArray,
    true_f: Sequence[float] | FloatArray,
    *,
    rtol: float = 0.01,
) -> MatchResult:
    """Greedily match detected peaks to truth by nearest relative frequency.

    Parameters
    ----------
    detected_f, true_f:
        Detected and true peak frequencies.
    rtol:
        Maximum relative error ``abs(detected - true) / true`` for a match.

    Returns
    -------
    MatchResult
        Matched ``(detected_index, true_index)`` pairs plus ``tp``, ``fp``, and
        ``fn`` counts.
    """
    if rtol < 0.0:
        msg = "rtol must be non-negative"
        raise ValueError(msg)

    detected = _as_float_array(detected_f)
    truth = _as_float_array(true_f)
    if detected.size == 0:
        return MatchResult(matches=(), tp=0, fp=0, fn=int(truth.size))
    if truth.size == 0:
        return MatchResult(matches=(), tp=0, fp=int(detected.size), fn=0)
    if np.any(truth <= 0.0) or np.any(detected <= 0.0):
        msg = "frequencies must be positive"
        raise ValueError(msg)

    candidates: list[tuple[float, int, int]] = []
    for det_idx, det_freq in enumerate(detected):
        relative_errors = np.abs(det_freq - truth) / truth
        for true_idx_raw in np.flatnonzero(relative_errors <= rtol):
            true_idx = int(true_idx_raw)
            candidates.append((float(relative_errors[true_idx]), det_idx, true_idx))

    used_detected: set[int] = set()
    used_truth: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, det_idx, true_idx in sorted(candidates):
        if det_idx in used_detected or true_idx in used_truth:
            continue
        used_detected.add(det_idx)
        used_truth.add(true_idx)
        matches.append((det_idx, true_idx))

    tp = len(matches)
    fp = int(detected.size) - tp
    fn = int(truth.size) - tp
    return MatchResult(matches=tuple(matches), tp=tp, fp=fp, fn=fn)


def precision_recall_f1(
    detected_f: Sequence[float] | FloatArray,
    true_f: Sequence[float] | FloatArray,
    *,
    rtol: float = 0.01,
) -> DetectionScores:
    """Compute precision, recall, F1, and counts from peak frequencies."""
    result = match_peaks(detected_f, true_f, rtol=rtol)
    precision = result.tp / (result.tp + result.fp) if result.tp + result.fp else 0.0
    recall = result.tp / (result.tp + result.fn) if result.tp + result.fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return DetectionScores(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=result.tp,
        fp=result.fp,
        fn=result.fn,
    )
