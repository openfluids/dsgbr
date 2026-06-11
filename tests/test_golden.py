"""Golden regression coverage for canonical synthetic detector outputs.

To regenerate the literals after an intentional algorithm change, run this
copy-pasteable snippet from the repository root and paste the printed arrays
back into ``GOLDEN_CASES``.
If goldens change, regenerate the README validation table too::

    uv run --extra tests python - <<'PY'
    import numpy as np
    from benchmarks.synthetic import make_spectrum, scenarios
    from dsgbr import dsgbr_detector

    configs = (("default", None), ("rt25", {"RT": 2.5}))
    for scenario_name in ("clean_tones", "dense_lowfreq", "noisy_welch", "no_peaks"):
        scenario = scenarios()[scenario_name]
        for config_name, case_info in configs:
            frequencies, psd, _ = make_spectrum(
                **scenario,
                rng=np.random.default_rng(12345),
            )
            peak_frequencies, peak_heights = dsgbr_detector(frequencies, psd, case_info=case_info)
            print(f"{scenario_name} {config_name}")
            print(np.array2string(peak_frequencies, separator=", ", precision=17))
            print(np.array2string(peak_heights, separator=", ", precision=17))
    PY
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from benchmarks.synthetic import make_spectrum, scenarios
from dsgbr import dsgbr_detector

SEED = 12345
DEFAULT_CONFIG = None
NONDEFAULT_CONFIG = {"RT": 2.5}

GOLDEN_CASES = (
    pytest.param(
        "clean_tones",
        "default",
        DEFAULT_CONFIG,
        np.array([17.994176034642358, 140.16112218895825]),
        np.array([0.04043142001520672, 0.00104893947381343]),
        id="clean_tones-default",
    ),
    pytest.param(
        "clean_tones",
        "rt25",
        NONDEFAULT_CONFIG,
        np.array([17.994176034642358, 140.16112218895825]),
        np.array([0.04043142001520672, 0.00104893947381343]),
        id="clean_tones-rt25",
    ),
    pytest.param(
        "dense_lowfreq",
        "default",
        DEFAULT_CONFIG,
        np.array([3.010562039595504, 3.07213953619564, 4.190563231143876, 7.437337476250562]),
        np.array([1.657649113925265, 1.6911761487695085, 1.1322521842070379, 0.3238366928264127]),
        id="dense_lowfreq-default",
    ),
    pytest.param(
        "dense_lowfreq",
        "rt25",
        NONDEFAULT_CONFIG,
        np.array(
            [
                0.5085080132878241,
                3.010562039595504,
                3.07213953619564,
                4.190563231143876,
                4.27627626071263,
                7.437337476250562,
                7.4877029552333285,
                7.6925987856191345,
            ]
        ),
        np.array(
            [
                3.612043838812659,
                1.657649113925265,
                1.6911761487695085,
                1.1322521842070379,
                0.6497966331313445,
                0.3238366928264127,
                0.2997276000072826,
                0.21878221344720702,
            ]
        ),
        id="dense_lowfreq-rt25",
    ),
    pytest.param(
        "noisy_welch",
        "default",
        DEFAULT_CONFIG,
        np.array(
            [
                10.369787023848575,
                15.995327635780118,
                44.57751113623613,
                58.36056560688896,
                78.67576203691237,
                85.39860416771144,
                135.64127593372106,
                299.0695637072566,
            ]
        ),
        np.array(
            [
                0.2236717637910994,
                0.09263410310381584,
                0.03259583213065864,
                0.01249834460967479,
                0.00695527829540262,
                0.00615386017360129,
                0.00278110329946289,
                0.00291533153692148,
            ]
        ),
        id="noisy_welch-default",
    ),
    pytest.param(
        "noisy_welch",
        "rt25",
        NONDEFAULT_CONFIG,
        np.array(
            [
                2.083698613470022,
                2.108249563339453,
                2.158222677961419,
                3.945324609846487,
                4.257444522444987,
                4.621243062986664,
                5.705936119036193,
                10.369787023848575,
                15.995327635780118,
                24.963389898567367,
                25.856212529776307,
                44.57751113623613,
                45.1027407611612,
                58.36056560688896,
                73.7669484875634,
                78.67576203691237,
                81.48962270085192,
                85.39860416771144,
                135.64127593372106,
                299.0695637072566,
            ]
        ),
        np.array(
            [
                1.2594697110841342,
                1.5630386451043379,
                1.7748978031638114,
                0.7342792094159478,
                0.7612858576526705,
                0.5624564223959047,
                0.3772463694499171,
                0.2236717637910994,
                0.09263410310381584,
                0.03839323891942015,
                0.03461934668795098,
                0.03259583213065864,
                0.02887654923782076,
                0.01249834460967479,
                0.00579640111735184,
                0.00695527829540262,
                0.00487737174300525,
                0.00615386017360129,
                0.00278110329946289,
                0.00291533153692148,
            ]
        ),
        id="noisy_welch-rt25",
    ),
    pytest.param(
        "no_peaks",
        "default",
        DEFAULT_CONFIG,
        np.array([]),
        np.array([]),
        id="no_peaks-default-empty",
    ),
    pytest.param(
        "no_peaks",
        "rt25",
        NONDEFAULT_CONFIG,
        np.array([1.0573967641760889, 23.543417338909958]),
        np.array([1.842419434051949, 0.01321081794557913]),
        id="no_peaks-rt25",
    ),
)


@pytest.mark.parametrize(
    ("scenario_name", "config_name", "case_info", "expected_frequencies", "expected_heights"),
    GOLDEN_CASES,
)
def test_synthetic_detector_outputs_match_goldens(
    scenario_name: str,
    config_name: str,
    case_info: dict[str, float | int] | None,
    expected_frequencies: np.ndarray,
    expected_heights: np.ndarray,
) -> None:
    frequencies, psd, _ = make_spectrum(
        **scenarios()[scenario_name],
        rng=np.random.default_rng(SEED),
    )

    peak_frequencies, peak_heights = dsgbr_detector(frequencies, psd, case_info=case_info)

    header = f"scenario={scenario_name} config={config_name}"
    np.testing.assert_allclose(
        peak_frequencies,
        expected_frequencies,
        rtol=1e-12,
        err_msg=f"{header} peak_frequencies",
    )
    np.testing.assert_allclose(
        peak_heights,
        expected_heights,
        rtol=1e-12,
        err_msg=f"{header} peak_heights",
    )
