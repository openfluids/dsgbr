"""CWRU bearing vibration: real signals with geometry-derived ground truth.

The Case Western Reserve University Bearing Data Center publishes accelerometer
recordings of a drive-end bearing under seeded faults.  It suits a peak detector
unusually well because the truth is not hand-labelled: for a given bearing the fault
frequencies are fixed multiples of shaft speed, and every recording carries its own
shaft speed, so the expected peaks are computed rather than asserted.

The recordings are **not redistributed with this package**.  CWRU makes them
available for research but under no explicit open licence, while dsgbr is
Apache-2.0, so :func:`fetch` downloads on demand into a local cache and verifies a
recorded digest.  Set ``DSGBR_CWRU_DATA_DIR`` to relocate the cache, or to point at
an existing copy so nothing is downloaded at all.

Source: https://engineering.case.edu/bearingdatacenter
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch

BASE_URL = "https://engineering.case.edu/sites/default/files"
SAMPLE_RATE_HZ = 12_000.0

#: Fault-frequency multipliers for the SKF 6205-2RS JEM drive-end bearing, as
#: multiples of shaft rotation frequency.  These follow from ball count, ball
#: diameter, pitch diameter and contact angle -- they are geometry, not fitted.
SKF_6205_DE: dict[str, float] = {
    "BPFI": 5.4152,  # ball pass frequency, inner race
    "BPFO": 3.5848,  # ball pass frequency, outer race
    "BSF": 2.3568,  # ball spin frequency
    "FTF": 0.3983,  # fundamental train (cage) frequency
}


@dataclass(frozen=True)
class Case:
    """One recording and the fault it was seeded with."""

    tag: str
    description: str
    fault: str | None  # key into SKF_6205_DE, or None for the healthy baseline
    sha256: str


#: Baseline plus one fault of each race.  The healthy recording is the false-alarm
#: control: a detector that reports fault harmonics there is finding nothing real.
CASES: tuple[Case, ...] = (
    Case(
        "97",
        "healthy baseline",
        None,
        "16bf48babcf1c7ac224bc1a81cd9eafdb27e42d5cf559761907e067e8eeadf3c",
    ),
    Case(
        "105",
        "inner race fault, 0.007 in",
        "BPFI",
        "f80b0ea04fd06b372a0eaec7c056543ea37e4bb4727a5b173d2a5bacd2aa9cab",
    ),
    Case(
        "130",
        "outer race fault, 0.007 in",
        "BPFO",
        "35a095307d0971477049b343a1b5981dde465a58fb7f233ad89b035068c1717d",
    ),
)


def cache_dir() -> Path:
    """Directory holding downloaded recordings."""
    override = os.environ.get("DSGBR_CWRU_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "dsgbr" / "cwru"


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def is_cached(case: Case) -> bool:
    """Whether *case* is already present and intact."""
    path = cache_dir() / f"{case.tag}.mat"
    return path.is_file() and _digest(path) == case.sha256


def fetch(case: Case, *, allow_download: bool = True) -> Path:
    """Return a local path to *case*, downloading it once if needed.

    Raises
    ------
    FileNotFoundError
        If the file is absent and *allow_download* is false.
    OSError
        If the downloaded file does not match the recorded digest.
    """
    path = cache_dir() / f"{case.tag}.mat"
    if path.is_file():
        if _digest(path) == case.sha256:
            return path
        path.unlink()  # corrupt or truncated; fetch again

    if not allow_download:
        raise FileNotFoundError(
            f"{path} is missing. Run benchmarks.real.compare, or set "
            "DSGBR_CWRU_DATA_DIR to a directory holding the CWRU .mat files."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{case.tag}.mat"
    urllib.request.urlretrieve(url, path)

    actual = _digest(path)
    if actual != case.sha256:
        path.unlink(missing_ok=True)
        raise OSError(f"digest mismatch for {url}: expected {case.sha256}, got {actual}")
    return path


def load_drive_end(case: Case) -> tuple[np.ndarray, float]:
    """Return the drive-end acceleration signal and its shaft frequency in Hz.

    Variable names are not uniform across the archive -- some files zero-pad the tag
    to three digits (``X097_DE_time`` against ``X105_DE_time``) -- so the arrays are
    located by suffix rather than by reconstructing the expected name.
    """
    from scipy.io import loadmat

    payload = loadmat(fetch(case))

    def _lookup(suffix: str) -> np.ndarray:
        matches = [key for key in payload if key.endswith(suffix)]
        if not matches:
            raise KeyError(f"no {suffix!r} array in CWRU file {case.tag}")
        return np.asarray(payload[sorted(matches)[0]])

    signal = _lookup("_DE_time").ravel().astype(float)
    rpm = float(_lookup("RPM").ravel()[0])
    return signal, rpm / 60.0


def envelope_spectrum(
    signal: np.ndarray,
    *,
    fs: float = SAMPLE_RATE_HZ,
    band: tuple[float, float] = (2000.0, 5000.0),
    nperseg: int = 32768,
    f_range: tuple[float, float] = (3.0, 1000.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the standard bearing envelope spectrum.

    Bearing faults modulate a structural resonance rather than radiating at the fault
    frequency directly, so the fault series appears in the envelope of a band around
    that resonance, not in the raw spectrum.

    Every parameter is fixed and documented so the benchmark measures the detector
    rather than preprocessing choices.  ``nperseg`` is deliberately large: at 8192 the
    resolution is 1.5 Hz and higher harmonics merge, which is precisely the regime
    where a bin-valued separation rule misbehaves.
    """
    nyquist = fs / 2.0
    coeffs_b, coeffs_a = butter(4, [band[0] / nyquist, band[1] / nyquist], btype="band")
    envelope = np.abs(hilbert(filtfilt(coeffs_b, coeffs_a, signal)))

    freqs, power = welch(envelope - envelope.mean(), fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    keep = (freqs > f_range[0]) & (freqs < f_range[1])
    return freqs[keep], power[keep]


def harmonic_recall(
    peaks: np.ndarray, fundamental: float, *, n_harmonics: int = 8, tol: float = 0.015
) -> int:
    """Count harmonics ``1..n`` of *fundamental* matched by some detected peak.

    A harmonic counts as recovered when a peak lies within *tol* relative frequency
    of it.  Harmonics falling outside the analysed band are not counted, so the
    caller must keep the band wide enough for the series it asks about.
    """
    peaks = np.asarray(peaks, dtype=float)
    if peaks.size == 0 or fundamental <= 0:
        return 0
    found = 0
    for order in range(1, n_harmonics + 1):
        target = order * fundamental
        if abs(peaks[np.argmin(np.abs(peaks - target))] - target) / target < tol:
            found += 1
    return found
