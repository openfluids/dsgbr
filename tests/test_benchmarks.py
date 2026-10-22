import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import compare, sensitivity
from benchmarks.metrics import match_peaks, precision_recall_f1
from benchmarks.synthetic import NoiseSpec, make_spectrum, scenarios


def test_make_spectrum_is_deterministic_for_fixed_seed() -> None:
    scenario = scenarios()["noisy_welch"]

    first = make_spectrum(**scenario, rng=np.random.default_rng(12345))
    second = make_spectrum(**scenario, rng=np.random.default_rng(12345))

    np.testing.assert_array_equal(first[0], second[0], err_msg="frequency grids differ")
    np.testing.assert_array_equal(first[1], second[1], err_msg="PSD values differ")
    np.testing.assert_array_equal(
        first[2]["peak_frequencies"],
        second[2]["peak_frequencies"],
        err_msg="truth peak frequencies differ",
    )


def test_peak_injection_lands_near_requested_frequency() -> None:
    requested_frequency = 50.0
    frequencies, psd, truth = make_spectrum(
        4096,
        1.0,
        200.0,
        peaks=[(requested_frequency, 20.0, 80.0)],
        noise=NoiseSpec(enabled=False),
        rng=np.random.default_rng(7),
        grid="linear",
    )

    excess = psd - truth["baseline"]
    observed_frequency = frequencies[int(np.argmax(excess))]

    assert np.isclose(observed_frequency, requested_frequency, rtol=0.0, atol=0.05), (
        f"injected peak maximum at {observed_frequency} Hz, expected "
        f"{requested_frequency} Hz within 0.05 Hz"
    )


def test_metrics_no_detections_counts_all_truth_as_missed() -> None:
    scores = precision_recall_f1([], [10.0, 20.0])

    assert scores.tp == 0, f"expected 0 true positives, got {scores.tp}"
    assert scores.fp == 0, f"expected 0 false positives, got {scores.fp}"
    assert scores.fn == 2, f"expected 2 false negatives, got {scores.fn}"
    assert scores.precision == 0.0, f"expected zero precision, got {scores.precision}"
    assert scores.recall == 0.0, f"expected zero recall, got {scores.recall}"
    assert scores.f1 == 0.0, f"expected zero F1, got {scores.f1}"


def test_metrics_no_truth_counts_all_detections_as_false_positive() -> None:
    scores = precision_recall_f1([10.0, 20.0], [])

    assert scores.tp == 0, f"expected 0 true positives, got {scores.tp}"
    assert scores.fp == 2, f"expected 2 false positives, got {scores.fp}"
    assert scores.fn == 0, f"expected 0 false negatives, got {scores.fn}"
    assert scores.precision == 0.0, f"expected zero precision, got {scores.precision}"
    assert scores.recall == 0.0, f"expected zero recall, got {scores.recall}"
    assert scores.f1 == 0.0, f"expected zero F1, got {scores.f1}"


def test_metrics_perfect_detection_scores_one() -> None:
    scores = precision_recall_f1([10.0, 20.0], [10.0, 20.0])

    assert scores.tp == 2, f"expected 2 true positives, got {scores.tp}"
    assert scores.fp == 0, f"expected 0 false positives, got {scores.fp}"
    assert scores.fn == 0, f"expected 0 false negatives, got {scores.fn}"
    assert scores.precision == 1.0, f"expected precision 1.0, got {scores.precision}"
    assert scores.recall == 1.0, f"expected recall 1.0, got {scores.recall}"
    assert scores.f1 == 1.0, f"expected F1 1.0, got {scores.f1}"


def test_duplicate_detections_do_not_match_same_truth_twice() -> None:
    result = match_peaks([99.9, 100.1, 200.0], [100.0], rtol=0.01)
    scores = precision_recall_f1([99.9, 100.1, 200.0], [100.0], rtol=0.01)

    assert result.tp == 1, f"expected one duplicate detection to match, got {result.tp}"
    assert result.fp == 2, f"expected remaining detections as false positives, got {result.fp}"
    assert result.fn == 0, f"expected no missed truth peaks, got {result.fn}"
    assert scores.tp == 1 and scores.fp == 2 and scores.fn == 0, (
        f"unexpected duplicate-match scores: tp={scores.tp}, fp={scores.fp}, fn={scores.fn}"
    )


def test_compare_quick_one_scenario_writes_sane_results() -> None:
    result = compare.main(["--quick", "--scenario", "noisy_welch", "--seed", "99"])

    scenario = result["scenarios"]["noisy_welch"]
    assert result["realizations"] == 3
    assert len(scenario["realizations"]) == 3
    assert set(scenario["summaries"]) == {"dsgbr", "scipy_find_peaks"}
    assert {"tp", "fp", "fn"}.issubset(scenario["summaries"]["dsgbr"])
    assert scenario["summaries"]["dsgbr"]["f1"]["mean"] > 0.0
    assert compare.RESULTS_PATH.exists()


def test_sensitivity_smoke_one_scenario_two_point_sweep_is_finite() -> None:
    two_point_sweeps = {"RT": (2.0, 3.3), "SW": (3, 5), "BWF": (0.02, 0.05)}
    result = sensitivity.run(
        seed=99,
        realizations=2,
        scenario_names=["noisy_welch"],
        sweep_values=two_point_sweeps,
        grid_rt_values=(2.0, 3.3),
        grid_bwf_values=(0.02, 0.05),
    )

    scenario = result["scenarios"]["noisy_welch"]
    assert result["realizations"] == 2
    assert set(scenario["sweeps"]) == {"RT", "SW", "BWF"}
    for sweep in scenario["sweeps"].values():
        for point in sweep["points"]:
            assert np.isfinite(point["mean_f1"])
            assert np.isfinite(point["std_f1"])
    assert all(len(sweep["points"]) == 2 for sweep in scenario["sweeps"].values())
    for sweep in scenario["sweeps"].values():
        means = [point["mean_f1"] for point in sweep["points"]]
        values = [point["value"] for point in sweep["points"]]
        best = max(means)
        lo, hi = sweep["robust_range"]
        assert np.isclose(sweep["robust_threshold"], 0.95 * best)
        assert lo in values and hi in values and lo <= hi
        # the best-scoring tested value must lie inside the robust range
        assert lo <= values[int(np.argmax(means))] <= hi
    matrix = np.asarray(scenario["grid_rt_bwf"]["mean_f1"], dtype=np.float64)
    assert matrix.shape == (2, 2)
    assert np.all(np.isfinite(matrix))
