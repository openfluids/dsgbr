"""Compare DSGBR defaults with a tuned ``scipy.signal.find_peaks`` baseline.

The benchmark uses a strict train/eval seed split. For each scenario family
(currently one family per named synthetic scenario), SciPy's prominence is chosen
by grid search on tuning realizations generated from ``seed + 1_000_000`` and is
then held fixed for evaluation realizations generated from ``seed``. Evaluation
realizations are never used to tune the SciPy baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

if __package__ in {None, ""}:  # Support ``python benchmarks/compare.py`` from the repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from benchmarks.metrics import DetectionScores, precision_recall_f1
from benchmarks.synthetic import Scenario, SpectrumTruth, make_spectrum, scenarios
from dsgbr import dsgbr_detector

FloatArray = NDArray[np.float64]
RESULTS_PATH = Path(__file__).with_name("results.json")
DEFAULT_REALIZATIONS = 20
QUICK_REALIZATIONS = 3
TUNING_SEED_OFFSET = 1_000_000
PROMINENCE_GRID = (0.03, 0.05, 0.08, 0.12, 0.18, 0.27, 0.40, 0.60, 0.90, 1.30)


class SummaryStats(TypedDict):
    """Mean and sample standard deviation for a metric."""

    mean: float
    std: float


class DetectorSummary(TypedDict):
    """Summary metrics and counts for one detector."""

    precision: SummaryStats
    recall: SummaryStats
    f1: SummaryStats
    tp: SummaryStats
    fp: SummaryStats
    fn: SummaryStats


class ScenarioResult(TypedDict):
    """Machine-readable result for one scenario."""

    tuned_prominence: float
    summaries: dict[str, DetectorSummary]
    realizations: list[dict[str, Any]]


class CompareResults(TypedDict):
    """Machine-readable benchmark payload."""

    seed: int
    realizations: int
    tuning_realizations: int
    tuning_seed_offset: int
    scenarios: dict[str, ScenarioResult]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realizations", type=int, default=DEFAULT_REALIZATIONS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"run {QUICK_REALIZATIONS} evaluation realizations",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(scenarios().keys()),
        help="scenario to run; may be repeated (default: all scenarios)",
    )
    return parser.parse_args(argv)


def _rng(seed: int, scenario_index: int, realization: int, *, tuning: bool) -> np.random.Generator:
    base_seed = seed + (TUNING_SEED_OFFSET if tuning else 0)
    return np.random.default_rng(base_seed + scenario_index * 10_000 + realization)


def _truth_frequencies(truth: SpectrumTruth) -> FloatArray:
    return np.asarray(truth["peak_frequencies"], dtype=np.float64)


def _dsgbr_detect(frequencies: FloatArray, psd: FloatArray) -> FloatArray:
    result = cast(tuple[FloatArray, FloatArray], dsgbr_detector(frequencies, psd))
    return np.asarray(result[0], dtype=np.float64)


def _scipy_detect(frequencies: FloatArray, psd: FloatArray, *, prominence: float) -> FloatArray:
    log_psd = np.log10(np.maximum(psd, np.finfo(np.float64).tiny))
    peak_indices, _ = find_peaks(log_psd, prominence=prominence)
    return frequencies[np.asarray(peak_indices, dtype=np.int64)]


def _score_detected(detected: FloatArray, truth_frequencies: FloatArray) -> DetectionScores:
    return precision_recall_f1(detected, truth_frequencies)


def _tune_prominence(
    scenario_name: str,
    scenario_index: int,
    scenario: Scenario,
    *,
    seed: int,
    tuning_realizations: int,
) -> float:
    best_prominence = PROMINENCE_GRID[0]
    best_f1 = -1.0
    best_fp = sys.maxsize
    for prominence in PROMINENCE_GRID:
        f1_values: list[float] = []
        total_fp = 0
        for realization in range(tuning_realizations):
            frequencies, psd, truth = make_spectrum(
                **scenario,
                rng=_rng(seed, scenario_index, realization, tuning=True),
            )
            detected = _scipy_detect(frequencies, psd, prominence=prominence)
            scores = _score_detected(detected, _truth_frequencies(truth))
            f1_values.append(scores.f1)
            total_fp += scores.fp
        mean_f1 = float(np.mean(f1_values)) if f1_values else 0.0
        if mean_f1 > best_f1 or (mean_f1 == best_f1 and total_fp < best_fp):
            best_f1 = mean_f1
            best_fp = total_fp
            best_prominence = prominence
    mean_fp = best_fp / tuning_realizations if tuning_realizations else 0.0
    print(
        f"tuned {scenario_name}: scipy prominence={best_prominence:.3g} "
        f"(train mean F1={best_f1:.3f}, mean FP={mean_fp:.3f}, n={tuning_realizations})",
        flush=True,
    )
    return best_prominence


def _stats(values: Sequence[float]) -> SummaryStats:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)) if array.size else 0.0,
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
    }


def _summary(scores: Sequence[DetectionScores]) -> DetectorSummary:
    return {
        "precision": _stats([score.precision for score in scores]),
        "recall": _stats([score.recall for score in scores]),
        "f1": _stats([score.f1 for score in scores]),
        "tp": _stats([score.tp for score in scores]),
        "fp": _stats([score.fp for score in scores]),
        "fn": _stats([score.fn for score in scores]),
    }


def _score_dict(scores: DetectionScores) -> dict[str, int | float]:
    return {
        "precision": scores.precision,
        "recall": scores.recall,
        "f1": scores.f1,
        "tp": scores.tp,
        "fp": scores.fp,
        "fn": scores.fn,
    }


def _run_scenario(
    scenario_name: str,
    scenario_index: int,
    scenario: Scenario,
    *,
    seed: int,
    realizations: int,
    tuning_realizations: int,
) -> ScenarioResult:
    prominence = _tune_prominence(
        scenario_name,
        scenario_index,
        scenario,
        seed=seed,
        tuning_realizations=tuning_realizations,
    )
    detector_scores: dict[str, list[DetectionScores]] = {"dsgbr": [], "scipy_find_peaks": []}
    realization_details: list[dict[str, Any]] = []

    for realization in range(realizations):
        frequencies, psd, truth = make_spectrum(
            **scenario,
            rng=_rng(seed, scenario_index, realization, tuning=False),
        )
        truth_f = _truth_frequencies(truth)
        detected_by_detector = {
            "dsgbr": _dsgbr_detect(frequencies, psd),
            "scipy_find_peaks": _scipy_detect(frequencies, psd, prominence=prominence),
        }
        detail: dict[str, Any] = {
            "realization": realization,
            "truth_frequencies": truth_f.tolist(),
            "detectors": {},
        }
        for detector_name, detected in detected_by_detector.items():
            scores = _score_detected(detected, truth_f)
            detector_scores[detector_name].append(scores)
            detail_detectors = cast(dict[str, Any], detail["detectors"])
            detail_detectors[detector_name] = {
                "detected_frequencies": detected.tolist(),
                "scores": _score_dict(scores),
            }
        realization_details.append(detail)
        dsgbr_running = _summary(detector_scores["dsgbr"])["f1"]["mean"]
        scipy_running = _summary(detector_scores["scipy_find_peaks"])["f1"]["mean"]
        print(
            f"{scenario_name}: {realization + 1}/{realizations} "
            f"running F1 dsgbr={dsgbr_running:.3f} scipy={scipy_running:.3f}",
            flush=True,
        )

    return {
        "tuned_prominence": prominence,
        "summaries": {
            detector_name: _summary(scores) for detector_name, scores in detector_scores.items()
        },
        "realizations": realization_details,
    }


def _format_metric(
    summary: DetectorSummary,
    metric: Literal["precision", "recall", "f1", "tp", "fp", "fn"],
) -> str:
    stats = summary[metric]
    return f"{stats['mean']:.3f} +/- {stats['std']:.3f}"


def _table(results: CompareResults) -> str:
    rows = [
        (
            "scenario",
            "detector",
            "precision",
            "recall",
            "f1",
            "mean tp",
            "mean fp",
            "mean fn",
            "n",
        )
    ]
    for scenario_name, scenario_result in results["scenarios"].items():
        for detector_name, summary in scenario_result["summaries"].items():
            rows.append(
                (
                    scenario_name,
                    detector_name,
                    _format_metric(summary, "precision"),
                    _format_metric(summary, "recall"),
                    _format_metric(summary, "f1"),
                    f"{summary['tp']['mean']:.3f}",
                    f"{summary['fp']['mean']:.3f}",
                    f"{summary['fn']['mean']:.3f}",
                    str(results["realizations"]),
                )
            )
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    lines = []
    for row_index, row in enumerate(rows):
        line = "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        lines.append(line)
        if row_index == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def _selected_scenarios(names: Iterable[str] | None) -> dict[str, Scenario]:
    all_scenarios = dict(scenarios())
    if names is None:
        return all_scenarios
    return {name: all_scenarios[name] for name in names}


def main(argv: Sequence[str] | None = None) -> CompareResults:
    """Run the benchmark, print a table, write JSON, and return the payload."""
    args = _parse_args(argv)
    realizations = QUICK_REALIZATIONS if args.quick else int(args.realizations)
    if realizations <= 0:
        msg = "--realizations must be positive"
        raise ValueError(msg)
    tuning_realizations = QUICK_REALIZATIONS if args.quick else min(8, realizations)
    selected = _selected_scenarios(cast(Sequence[str] | None, args.scenario))

    payload: CompareResults = {
        "seed": int(args.seed),
        "realizations": realizations,
        "tuning_realizations": tuning_realizations,
        "tuning_seed_offset": TUNING_SEED_OFFSET,
        "scenarios": {},
    }
    scenario_order = list(scenarios().keys())
    for scenario_name, scenario in selected.items():
        scenario_index = scenario_order.index(scenario_name)
        print(f"running {scenario_name} ({realizations} eval realizations)", flush=True)
        payload["scenarios"][scenario_name] = _run_scenario(
            scenario_name,
            scenario_index,
            scenario,
            seed=int(args.seed),
            realizations=realizations,
            tuning_realizations=tuning_realizations,
        )

    table = _table(payload)
    print(table)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {RESULTS_PATH}")
    return payload


if __name__ == "__main__":
    main()
