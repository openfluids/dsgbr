#!/usr/bin/env python3
"""Basic DSGBR usage: detect peaks in a synthetic PSD.

This example creates a flat PSD with injected spikes, runs the detector,
and prints the results.
"""

import numpy as np

from dsgbr import dsgbr_detector

# Create a synthetic spectrum: flat background + 4 spikes
frequencies = np.linspace(0.001, 1.0, 2048)
psd = np.ones_like(frequencies)
psd[200] = 12.0
psd[600] = 16.0
psd[1100] = 9.5
psd[1700] = 11.5

# Detect peaks with moderate settings
peak_f, peak_h = dsgbr_detector(
    frequencies,
    psd,
    case_info={
        "smooth": "none",
        "ratio_threshold": 1.5,
        "baseline_window": 61,
        "max_peaks": 10,
    },
)

print(f"Detected {peak_f.size} peaks:")
for f, h in zip(peak_f, peak_h, strict=True):
    print(f"  f = {f:.4f} Hz,  PSD = {h:.2f}")
