"""Backward-compatibility shim for ``from dsgbr.DSGBR import ...``.

This module re-exports the full public API so that existing call sites
(e.g. ``from dsgbr.DSGBR import dsgbr_detector``) continue to work
unchanged after the package restructure.
"""

from dsgbr._compat import (
    DSGBR_PARAM_ALIASES,
    DSGBRDetectionConfig,
    detect_peaks_case_adaptive,
)
from dsgbr._config import DetectionConfig
from dsgbr._detector import compute_support_series, dsgbr_detector
from dsgbr._selection import (
    find_nearest_frequency,
    select_peaks_by_frequency_bands,
)

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
