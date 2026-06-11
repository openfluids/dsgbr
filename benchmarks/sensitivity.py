"""Parameter sensitivity sweeps for DSGBR defaults.

The study varies one detector parameter at a time around the current defaults
(``RT=3.3``, ``SW=3``, ``BWF=0.05``) and also runs a coarse ``RT`` x ``BWF``
grid. For each scenario and sweep point it reports mean F1 plus sample standard
deviation over the same deterministic seed/realization convention used by
``benchmarks.compare``.

A parameter's robust range for a scenario is the contiguous interval of tested
values containing the best mean F1 where every point has mean F1 at least 95% of
that scenario/sweep maximum. If all tested F1 values are zero, the whole tested
range is reported as robust.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict, cast

if __package__ in {None, ""}:  # Support ``python benchmarks/sensitivity.py`` from the repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks.compare import (
    DEFAULT_REALIZATIONS,
    QUICK_REALIZATIONS,
    TUNING_SEED_OFFSET,
    _rng,
    _score_detected,
    _selected_scenarios,
    _stats,
    _truth_frequencies,
)
from benchmarks.synthetic import Scenario, make_spectrum, scenarios
from dsgbr import dsgbr_detector

RESULTS_PATH = Path(__file__).with_name("sensitivity.json")

RT_GRID = (1.5, 2.14, 2.79, 3.3, 3.43, 4.07, 4.71, 5.36, 6.0)
SW_GRID = (3, 5, 7, 9, 11)
BWF_GRID = (0.01, 0.022, 0.05, 0.112, 0.25)
GRID_RT = RT_GRID
GRID_BWF = BWF_GRID

PARAMETERS: Mapping[str, tuple[str, Sequence[float | int]]] = {
    "RT": ("ratio_threshold", RT_GRID),
    "SW": ("smooth_window", SW_GRID),
    "BWF": ("baseline_window_frac", BWF_GRID),
}


class PointStats(TypedDict):
    """Mean and sample standard deviation for one sweep point."""

    value: float | int
    mean_f1: float
    std_f1: float


class SweepResult(TypedDict):
    """Machine-readable results for one 1-D parameter sweep."""

    points: list[PointStats]
    robust_range: list[float | int]
    robust_threshold: float


class ScenarioSensitivity(TypedDict):
    """Sensitivity payload for one synthetic scenario."""

    sweeps: dict[str, SweepResult]
    grid_rt_bwf: dict[str, Any]


class SensitivityResults(TypedDict):
    """Top-level sensitivity benchmark payload."""

    seed: int
    realizations: int
    tuning_seed_offset: int
    defaults: dict[str, float | int]
    scenarios: dict[str, ScenarioSensitivity]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realizations", type=int, default=DEFAULT_REALIZATIONS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"run {QUICK_REALIZATIONS} realizations per point",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(scenarios().keys()),
        help="scenario to run; may be repeated (default: all scenarios)",
    )
    return parser.parse_args(argv)


def _case_info(param_values: Mapping[str, float | int]) -> dict[str, str]:
    return {key: str(value) for key, value in param_values.items()}


def _detect_with_params(
    frequencies: np.ndarray,
    psd: np.ndarray,
    param_values: Mapping[str, float | int],
) -> np.ndarray:
    detected, _ = cast(
        tuple[np.ndarray, np.ndarray],
        dsgbr_detector(frequencies, psd, case_info=_case_info(param_values)),
    )
    return np.asarray(detected, dtype=np.float64)


def _score_values(
    scenario: Scenario,
    scenario_index: int,
    *,
    seed: int,
    realizations: int,
    param_values: Mapping[str, float | int],
) -> list[float]:
    f1_values: list[float] = []
    for realization in range(realizations):
        frequencies, psd, truth = make_spectrum(
            **scenario,
            rng=_rng(seed, scenario_index, realization, tuning=False),
        )
        detected = _detect_with_params(frequencies, psd, param_values)
        scores = _score_detected(detected, _truth_frequencies(truth))
        f1_values.append(scores.f1)
    return f1_values


def _point_stats(value: float | int, f1_values: Sequence[float]) -> PointStats:
    stats = _stats(f1_values)
    return {"value": value, "mean_f1": stats["mean"], "std_f1": stats["std"]}


def _robust_range(points: Sequence[PointStats]) -> tuple[list[float | int], float]:
    if not points:
        return [], 0.0
    means = np.asarray([point["mean_f1"] for point in points], dtype=np.float64)
    max_index = int(np.argmax(means))
    max_f1 = float(means[max_index])
    threshold = 0.95 * max_f1
    if max_f1 == 0.0:
        return [points[0]["value"], points[-1]["value"]], threshold

    left = max_index
    while left > 0 and means[left - 1] >= threshold:
        left -= 1
    right = max_index
    while right + 1 < len(points) and means[right + 1] >= threshold:
        right += 1
    return [points[left]["value"], points[right]["value"]], threshold


def _run_sweep(
    scenario_name: str,
    scenario_index: int,
    scenario: Scenario,
    *,
    seed: int,
    realizations: int,
    parameter: str,
    values: Sequence[float | int],
) -> SweepResult:
    points: list[PointStats] = []
    for value in values:
        f1_values = _score_values(
            scenario,
            scenario_index,
            seed=seed,
            realizations=realizations,
            param_values={parameter: value},
        )
        point = _point_stats(value, f1_values)
        points.append(point)
        print(
            f"sweep {scenario_name} {parameter}={value}: "
            f"F1={point['mean_f1']:.3f} +/- {point['std_f1']:.3f} (n={realizations})",
            flush=True,
        )
    robust, threshold = _robust_range(points)
    return {"points": points, "robust_range": robust, "robust_threshold": threshold}


def _run_rt_bwf_grid(
    scenario_name: str,
    scenario_index: int,
    scenario: Scenario,
    *,
    seed: int,
    realizations: int,
    rt_values: Sequence[float] = GRID_RT,
    bwf_values: Sequence[float] = GRID_BWF,
) -> dict[str, Any]:
    matrix: list[list[float]] = []
    for rt in rt_values:
        row: list[float] = []
        for bwf in bwf_values:
            values = _score_values(
                scenario,
                scenario_index,
                seed=seed,
                realizations=realizations,
                param_values={"RT": rt, "BWF": bwf},
            )
            mean_f1 = float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0
            row.append(mean_f1)
            print(
                f"grid {scenario_name} RT={rt} BWF={bwf}: F1={mean_f1:.3f} (n={realizations})",
                flush=True,
            )
        matrix.append(row)
    return {"rt_values": list(rt_values), "bwf_values": list(bwf_values), "mean_f1": matrix}


def _format_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.3g}"


def _format_range(values: Sequence[float | int]) -> str:
    if not values:
        return "n/a"
    return f"{_format_value(values[0])}-{_format_value(values[-1])}"


def _sweep_table(results: SensitivityResults) -> str:
    rows = [("scenario", "parameter", "robust range", "best", "max mean F1", "n")]
    for scenario_name, scenario_result in results["scenarios"].items():
        for parameter, sweep in scenario_result["sweeps"].items():
            best_point = max(sweep["points"], key=lambda point: point["mean_f1"])
            rows.append(
                (
                    scenario_name,
                    parameter,
                    _format_range(sweep["robust_range"]),
                    _format_value(best_point["value"]),
                    f"{best_point['mean_f1']:.3f}",
                    str(results["realizations"]),
                )
            )
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    lines: list[str] = []
    for row_index, row in enumerate(rows):
        lines.append("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
        if row_index == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def _grid_tables(results: SensitivityResults) -> str:
    sections: list[str] = []
    for scenario_name, scenario_result in results["scenarios"].items():
        grid = scenario_result["grid_rt_bwf"]
        bwf_values = cast(list[float], grid["bwf_values"])
        rt_values = cast(list[float], grid["rt_values"])
        matrix = cast(list[list[float]], grid["mean_f1"])
        rows: list[tuple[str, ...]] = [
            ("RT \\ BWF", *(_format_value(value) for value in bwf_values))
        ]
        for rt, row_values in zip(rt_values, matrix, strict=True):
            rows.append((_format_value(rt), *(f"{value:.3f}" for value in row_values)))
        widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
        lines = [f"RT x BWF mean F1: {scenario_name}"]
        for row_index, row in enumerate(rows):
            lines.append("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
            if row_index == 0:
                lines.append("  ".join("-" * width for width in widths))
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _selected(names: Iterable[str] | None) -> dict[str, Scenario]:
    return _selected_scenarios(names)


def run(
    *,
    seed: int = 0,
    realizations: int = DEFAULT_REALIZATIONS,
    scenario_names: Sequence[str] | None = None,
    sweep_values: Mapping[str, Sequence[float | int]] | None = None,
    grid_rt_values: Sequence[float] = GRID_RT,
    grid_bwf_values: Sequence[float] = GRID_BWF,
) -> SensitivityResults:
    """Run sensitivity sweeps and return the machine-readable payload."""
    if realizations <= 0:
        msg = "realizations must be positive"
        raise ValueError(msg)

    selected = _selected(scenario_names)
    scenario_order = list(scenarios().keys())
    payload: SensitivityResults = {
        "seed": seed,
        "realizations": realizations,
        "tuning_seed_offset": TUNING_SEED_OFFSET,
        "defaults": {"RT": 3.3, "SW": 3, "BWF": 0.05},
        "scenarios": {},
    }
    for scenario_name, scenario in selected.items():
        scenario_index = scenario_order.index(scenario_name)
        print(f"running sensitivity for {scenario_name} ({realizations} realizations)", flush=True)
        sweeps: dict[str, SweepResult] = {}
        for alias, (parameter, default_values) in PARAMETERS.items():
            values = (
                sweep_values[alias] if sweep_values and alias in sweep_values else default_values
            )
            sweeps[alias] = _run_sweep(
                scenario_name,
                scenario_index,
                scenario,
                seed=seed,
                realizations=realizations,
                parameter=parameter,
                values=values,
            )
        grid = _run_rt_bwf_grid(
            scenario_name,
            scenario_index,
            scenario,
            seed=seed,
            realizations=realizations,
            rt_values=grid_rt_values,
            bwf_values=grid_bwf_values,
        )
        payload["scenarios"][scenario_name] = {"sweeps": sweeps, "grid_rt_bwf": grid}
    return payload


def main(argv: Sequence[str] | None = None) -> SensitivityResults:
    """Run sweeps, print tables, write JSON, and return the payload."""
    args = _parse_args(argv)
    realizations = QUICK_REALIZATIONS if args.quick else int(args.realizations)
    payload = run(
        seed=int(args.seed),
        realizations=realizations,
        scenario_names=cast(Sequence[str] | None, args.scenario),
    )
    summary = _sweep_table(payload)
    grids = _grid_tables(payload)
    print("\nRobust ranges (within 5% of max mean F1):")
    print(summary)
    print()
    print(grids)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {RESULTS_PATH}")
    return payload


if __name__ == "__main__":
    main()
