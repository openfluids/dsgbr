#!/usr/bin/env python3
"""Parameter tuning: sweep ratio_threshold and compare peak counts.

This example shows how the ratio_threshold (RT) parameter controls
detection sensitivity. Lower RT detects more peaks (including noise);
higher RT is more selective.
"""

import numpy as np

from dsgbr import dsgbr_detector

# Create a noisy spectrum with peaks of varying strength
rng = np.random.default_rng(42)
frequencies = np.linspace(0.001, 1.0, 4096)
psd = 1.0 + rng.exponential(0.1, size=len(frequencies))

# Inject peaks of decreasing amplitude
for idx, amp in [(300, 20.0), (800, 10.0), (1500, 5.0), (2500, 3.0), (3500, 2.0)]:
    psd[idx] = amp

# Sweep ratio_threshold
print(f"{'RT':>6}  {'Peaks':>5}  Detected frequencies")
print("-" * 60)

for rt in [1.2, 1.5, 2.0, 3.0, 5.0, 10.0]:
    peak_f, _ = dsgbr_detector(
        frequencies,
        psd,
        case_info={
            "smooth": "none",
            "ratio_threshold": rt,
            "baseline_window": 101,
            "max_peaks": 20,
        },
    )
    freqs_str = ", ".join(f"{f:.3f}" for f in peak_f[:5])
    if peak_f.size > 5:
        freqs_str += f" (+{peak_f.size - 5} more)"
    print(f"{rt:6.1f}  {peak_f.size:5d}  {freqs_str}")
