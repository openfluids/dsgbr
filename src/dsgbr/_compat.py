"""Backward-compatibility aliases and deprecated entry points.

This module preserves the old ``detect_peaks_case_adaptive`` name and the
``DSGBRDetectionConfig`` type alias so that existing call sites continue
to work without modification.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from dsgbr._config import DetectionConfig
from dsgbr._detector import dsgbr_detector

#: Bidirectional mapping for DSGBR parameters.
#: Allows use of either short keys (RT, SW, BWF) or full parameter names.
DSGBR_PARAM_ALIASES: dict[str, str] = {
    # Short -> Long mappings
    "RT": "ratio_threshold",
    "SW": "smooth_window",
    "BWF": "baseline_window_frac",
    "DL": "distance_low",
    "DH": "distance_high",
    "SF": "switch_frequency",
    "MP": "max_peaks",
}

#: Type alias preserved for backward compatibility.
DSGBRDetectionConfig = DetectionConfig


def detect_peaks_case_adaptive(
    frequencies: np.ndarray,
    psd: np.ndarray,
    case_info: Any | None = None,
    *,
    return_support: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Wrap :func:`dsgbr_detector` with a deprecation warning.

    .. deprecated:: 0.1.0
        Use :func:`dsgbr.dsgbr_detector` instead.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Frequency axis (Hz).
    psd : numpy.ndarray
        Power spectral density values.
    case_info : dict or None, optional
        Parameter dictionary for configuration.
    return_support : bool, optional
        If ``True``, return intermediate arrays for visualization.

    Returns
    -------
    tuple
        Same as :func:`dsgbr_detector`.
    """
    warnings.warn(
        "detect_peaks_case_adaptive is deprecated, use dsgbr_detector instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return dsgbr_detector(frequencies, psd, case_info=case_info, return_support=return_support)
