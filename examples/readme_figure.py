"""Generate the README hero figure for DSGBR."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dsgbr import dsgbr_detector

SEED = 29
N_POINTS = 4096
F_MIN_HZ = 0.002
F_MAX_HZ = 1.2
# Frequency comb: fundamental plus harmonics, as shed by a periodic wake
F0_HZ = 0.045
N_HARMONICS = 10
TRUTH_HZ = F0_HZ * np.arange(1, N_HARMONICS + 1)
PEAK_GAINS = 26.0 / np.arange(1, N_HARMONICS + 1) ** 0.9
PEAK_WIDTH_DECADES = np.full(N_HARMONICS, 0.004)
NOISE_SIGMA = 0.30
OUTPUT = Path("docs/figures/readme_detection.png")


def make_scene() -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic, noisy PSD with injected log-Gaussian peaks."""
    rng = np.random.default_rng(SEED)
    frequencies = np.logspace(np.log10(F_MIN_HZ), np.log10(F_MAX_HZ), N_POINTS)
    baseline = 0.22 + 0.12 / np.sqrt(frequencies + 0.01) + 0.06 * frequencies**1.2
    psd = baseline * rng.lognormal(mean=0.0, sigma=NOISE_SIGMA, size=N_POINTS)
    log_f = np.log10(frequencies)
    for truth_hz, gain, width in zip(TRUTH_HZ, PEAK_GAINS, PEAK_WIDTH_DECADES, strict=False):
        peak = np.exp(-0.5 * ((log_f - np.log10(truth_hz)) / width) ** 2)
        psd += baseline * gain * peak
    return frequencies, psd


def nearest_truth_report(detected_hz: np.ndarray) -> str:
    """Format detected peaks against the nearest injected truth."""
    rows = []
    for truth_hz in TRUTH_HZ:
        if detected_hz.size:
            nearest = detected_hz[np.argmin(np.abs(detected_hz - truth_hz))]
            rows.append(f"truth {truth_hz:.5f} Hz -> detected {nearest:.5f} Hz")
        else:
            rows.append(f"truth {truth_hz:.5f} Hz -> detected none")
    return "\n".join(rows)


def main() -> None:
    frequencies, psd = make_scene()
    peak_f, peak_h, support = dsgbr_detector(frequencies, psd, return_support=True)

    fig, ax = plt.subplots(figsize=(6.7, 4.2), dpi=150)
    ax.loglog(frequencies, psd, color="0.58", linewidth=0.8, alpha=0.9, label="noisy PSD")
    ax.loglog(frequencies, support["search_series"], color="#1f77b4", linewidth=1.2, label="SEARCH")
    ax.loglog(
        frequencies,
        support["baseline_series"],
        color="#ff7f0e",
        linewidth=1.4,
        label="rolling-median BASELINE",
    )
    ax.scatter(peak_f, peak_h, s=42, marker="o", color="#d62728", zorder=5, label="detected peaks")
    ymin, ymax = ax.get_ylim()
    ax.vlines(
        TRUTH_HZ,
        ymin,
        ymin * (ymax / ymin) ** 0.08,
        color="black",
        linewidth=1.2,
        label="injected truth",
    )
    ax.set_ylim(ymin, ymax)
    ax.set_title("DSGBR detection of a frequency comb in a noisy spectrum")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (arbitrary units / Hz)")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
    fig.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, format="png", metadata={"Software": "matplotlib"})
    plt.close(fig)

    print("Detected frequencies (Hz):", ", ".join(f"{f:.5f}" for f in peak_f))
    print("Injected truth frequencies (Hz):", ", ".join(f"{f:.5f}" for f in TRUTH_HZ))
    print(nearest_truth_report(peak_f))
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
