"""Tests for DetectionConfig construction, validation, and serialization."""

from __future__ import annotations

import warnings

import pytest

from dsgbr import DetectionConfig, DSGBRDetectionConfig


class TestDefaults:
    """Default construction and basic properties."""

    def test_default_construction(self):
        cfg = DetectionConfig()
        assert cfg.ratio_threshold == 3.3
        assert cfg.smooth_window == 3
        assert cfg.max_peaks == 25
        assert cfg.n_bands == 10

    def test_frozen_immutability(self):
        cfg = DetectionConfig()
        with pytest.raises(AttributeError):
            cfg.ratio_threshold = 2.0  # type: ignore[misc]

    def test_dsgbr_detection_config_alias(self):
        assert DSGBRDetectionConfig is DetectionConfig


class TestValidation:
    """__post_init__ validation rules."""

    def test_ratio_threshold_below_one(self):
        with pytest.raises(ValueError, match=r"ratio_threshold must be >= 1\.0"):
            DetectionConfig(ratio_threshold=0.5)

    def test_ratio_threshold_exactly_one(self):
        cfg = DetectionConfig(ratio_threshold=1.0)
        assert cfg.ratio_threshold == 1.0

    def test_smooth_window_too_small(self):
        with pytest.raises(ValueError, match="smooth_window must be >= 3"):
            DetectionConfig(smooth_window=1)

    def test_smooth_window_even(self):
        with pytest.raises(ValueError, match="smooth_window must be odd"):
            DetectionConfig(smooth_window=4)

    def test_polyorder_exceeds_window(self):
        with pytest.raises(ValueError, match=r"smooth_polyorder.*must be.*< smooth_window"):
            DetectionConfig(smooth_window=3, smooth_polyorder=3)

    def test_max_peaks_zero(self):
        with pytest.raises(ValueError, match="max_peaks must be >= 1"):
            DetectionConfig(max_peaks=0)

    def test_n_bands_zero(self):
        with pytest.raises(ValueError, match="n_bands must be >= 1"):
            DetectionConfig(n_bands=0)

    @pytest.mark.parametrize(
        ("field", "bad_value", "match"),
        [
            ("band_strategy", "invalid", "band_strategy must be"),
            ("baseline_window_frac", 0.0, "baseline_window_frac must be"),
            ("baseline_window_frac", 1.1, "baseline_window_frac must be"),
            ("baseline_window", 3, "baseline_window must be > 3"),
            ("distance_low", 0, "distance_low must be >= 1"),
            ("distance_high", 0, "distance_high must be >= 1"),
            ("switch_frequency", -0.1, "switch_frequency must be >= 0"),
            ("ulf_min_q", -0.1, "ulf_min_q must be >= 0"),
            ("ulf_max_points", -1, "ulf_max_points must be >= 0"),
            ("smooth_polyorder", -1, "smooth_polyorder must be >= 0"),
        ],
    )
    def test_newly_validated_fields_reject_bad_values(self, field, bad_value, match):
        with pytest.raises(ValueError, match=match):
            DetectionConfig(**{field: bad_value})

    @pytest.mark.parametrize(
        ("field", "boundary_value"),
        [
            ("band_strategy", "equal"),
            ("baseline_window_frac", 1.0),
            ("baseline_window", 4),
            ("distance_low", 1),
            ("distance_high", 1),
            ("switch_frequency", 0.0),
            ("ulf_min_q", 0.0),
            ("ulf_max_points", 0),
            ("smooth_polyorder", 0),
            ("ulf_fmax", -1.0),
        ],
    )
    def test_newly_validated_fields_accept_boundary_values(self, field, boundary_value):
        cfg = DetectionConfig(**{field: boundary_value})
        assert getattr(cfg, field) == boundary_value


class TestFromCaseInfo:
    """from_case_info alias resolution and edge cases."""

    def test_short_aliases(self):
        cfg = DetectionConfig.from_case_info(
            {
                "RT": "2.2",
                "SW": "11",
                "BWF": "0.002",
                "DH": "5",
                "DL": "4",
                "SF": "0.01",
                "MP": "7",
            }
        )
        assert cfg.ratio_threshold == 2.2
        assert cfg.smooth_window == 11
        assert cfg.baseline_window_frac == 0.002
        assert cfg.distance_high == 5
        assert cfg.distance_low == 4
        assert cfg.switch_frequency == 0.01
        assert cfg.max_peaks == 7

    def test_long_names(self):
        cfg = DetectionConfig.from_case_info(
            {
                "ratio_threshold": 2.5,
                "smooth_window": 7,
            }
        )
        assert cfg.ratio_threshold == 2.5
        assert cfg.smooth_window == 7

    def test_none_returns_defaults(self):
        cfg = DetectionConfig.from_case_info(None)
        assert cfg == DetectionConfig()

    def test_empty_dict_returns_defaults(self):
        cfg = DetectionConfig.from_case_info({})
        assert cfg == DetectionConfig()

    def test_non_dict_returns_defaults(self):
        cfg = DetectionConfig.from_case_info("not a dict")
        assert cfg == DetectionConfig()

    def test_negative_baseline_window_ignored(self):
        cfg = DetectionConfig.from_case_info({"baseline_window": "-1"})
        assert cfg.baseline_window is None

    @pytest.mark.parametrize("bad_value", ["abc", None])
    def test_invalid_value_warns_and_skips(self, bad_value):
        with pytest.warns(
            UserWarning,
            match=r"dsgbr: ignoring invalid value for 'RT':",
        ) as warning_records:
            cfg = DetectionConfig.from_case_info({"RT": bad_value})

        assert repr(bad_value) in str(warning_records[0].message)
        assert cfg.ratio_threshold == 3.3

    def test_invalid_value_continues_to_next_alias(self):
        with pytest.warns(
            UserWarning,
            match=r"dsgbr: ignoring invalid value for 'ratio_threshold':",
        ):
            cfg = DetectionConfig.from_case_info({"ratio_threshold": "abc", "RT": "2.4"})

        assert cfg.ratio_threshold == 2.4

    def test_unknown_keys_warn_mode_lists_sorted_bogus_keys(self):
        with pytest.warns(UserWarning) as warning_records:
            cfg = DetectionConfig.from_case_info(
                {"RT": "2.1", "zzz": 1, "bogus": 2}, on_unknown="warn"
            )

        assert cfg.ratio_threshold == 2.1
        assert len(warning_records) == 1
        message = str(warning_records[0].message)
        assert "bogus" in message
        assert "zzz" in message
        assert message.index("'bogus'") < message.index("'zzz'")

    def test_unknown_keys_default_mode_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cfg = DetectionConfig.from_case_info({"RT": "2.1", "bogus": 2})

        assert cfg.ratio_threshold == 2.1

    def test_clean_config_produces_zero_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cfg = DetectionConfig.from_case_info({"RT": "2.1", "SW": "7"})

        assert cfg.ratio_threshold == 2.1
        assert cfg.smooth_window == 7

    def test_invalid_on_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="on_unknown must be"):
            DetectionConfig.from_case_info({"RT": "2.1"}, on_unknown="loud")

    def test_legacy_prominence_alias(self):
        cfg = DetectionConfig.from_case_info({"prominence_window": 51})
        assert cfg.baseline_window == 51

    def test_legacy_two_pass_alias(self):
        cfg = DetectionConfig.from_case_info({"two_pass_fmin": 0.05})
        assert cfg.switch_frequency == 0.05


class TestToMetadata:
    """Serialization round-trip."""

    def test_round_trip(self):
        cfg = DetectionConfig(ratio_threshold=2.0, smooth_window=7)
        meta = cfg.to_metadata()
        assert meta["ratio_threshold"] == 2.0
        assert meta["smooth_window"] == 7
        assert isinstance(meta, dict)

    def test_all_fields_present(self):
        cfg = DetectionConfig()
        meta = cfg.to_metadata()
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(cfg)}
        assert set(meta.keys()) == field_names
