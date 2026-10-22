#!/usr/bin/env python3
"""Visualization: SEARCH/BASELINE overlay with detected peaks.

Requires matplotlib: pip install matplotlib
"""

import numpy as np

from dsgbr import compute_support_series, dsgbr_detector

# Build a synthetic spectrum with 3 clear peaks
frequencies = np.linspace(0.001, 1.0, 2048)
psd = np.ones_like(frequencies) * 0.5
psd[300] = 15.0
psd[800] = 10.0
psd[1500] = 7.0

# Add some broadband noise
rng = np.random.default_rng(123)
psd += rng.exponential(0.05, size=len(psd))

config = {
    "smooth": "savgol",
    "smooth_window": 7,
    "ratio_threshold": 2.0,
    "baseline_window": 61,
    "max_peaks": 10,
}

# Detect peaks and get support series
peak_f, peak_h = dsgbr_detector(frequencies, psd, case_info=config)
support = compute_support_series(frequencies, psd, case_info=config)

# Plot
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed. Install with: pip install matplotlib")
    raise SystemExit(1) from None

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# Top: PSD + SEARCH + BASELINE + threshold
ax1.semilogy(frequencies, psd, alpha=0.4, label="PSD", color="grey")
ax1.semilogy(frequencies, support["search_series"], label="SEARCH", color="C0")
ax1.semilogy(frequencies, support["baseline_series"], label="BASELINE", color="C1")
ax1.semilogy(frequencies, support["rthreshold"], "--", label="Threshold", color="C3")
ax1.plot(peak_f, peak_h, "rv", markersize=8, label=f"Peaks ({peak_f.size})")
ax1.set_ylabel("PSD")
ax1.legend(loc="upper right", fontsize=8)
ax1.set_title("DSGBR Detection Overlay")

# Bottom: ratio series
ax2.plot(frequencies, support["ratio_series"], color="C2", label="SEARCH/BASELINE")
ax2.axhline(
    config["ratio_threshold"], color="C3", ls="--", label=f"RT = {config['ratio_threshold']}"
)
ax2.set_xlabel("Frequency")
ax2.set_ylabel("Ratio")
ax2.legend(loc="upper right", fontsize=8)

fig.tight_layout()
fig.savefig("dsgbr_visualization.png", dpi=150, bbox_inches="tight")
print(f"Saved dsgbr_visualization.png ({peak_f.size} peaks detected)")
plt.close(fig)
