"""Dual Savitzky-Golay Baseline Ratio (DSGBR) spectral peak detector.

The DSGBR detector applies two Savitzky-Golay smoothing passes -- one to build
the SEARCH series, another to obtain a broader BASELINE -- and accepts peaks
where the SEARCH/BASELINE ratio exceeds a configurable threshold.

Public API
----------
.. autosummary::
    DetectionConfig
    dsgbr_detector
    compute_support_series
    select_peaks_by_frequency_bands
    find_nearest_frequency
    detect_peaks_case_adaptive
    DSGBRDetectionConfig
    DSGBR_PARAM_ALIASES
"""

from dsgbr._compat import DSGBR_PARAM_ALIASES, DSGBRDetectionConfig, detect_peaks_case_adaptive
from dsgbr._config import DetectionConfig
from dsgbr._detector import compute_support_series, dsgbr_detector
from dsgbr._selection import find_nearest_frequency, select_peaks_by_frequency_bands

__version__ = "0.2.0"
__all__ = [
    "DSGBR_PARAM_ALIASES",
    "DSGBRDetectionConfig",
    "DetectionConfig",
    "compute_support_series",
    "detect_peaks_case_adaptive",
    "dsgbr_detector",
    "find_nearest_frequency",
    "select_peaks_by_frequency_bands",
]
