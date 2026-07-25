"""Keep the documented defaults honest.

The README configuration table and the docs/algorithm.md default list are the
first thing users read.  0.4.0 shipped with the table still claiming
``distance_high = 1`` after the default had become 5, so these are checked
mechanically rather than by eye.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from dsgbr import DetectionConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
ALGORITHM_DOC = REPO_ROOT / "docs" / "algorithm.md"

# Fields whose documented value is prose rather than a literal default.
_PROSE_DEFAULTS = {"baseline_window"}


def _normalise(value: str) -> str:
    """Compare 0.001 with 0.001 and 9.0 with 9 without float parsing surprises."""
    text = value.strip().strip("`")
    try:
        return f"{float(text):.10g}"
    except ValueError:
        return text


def _readme_config_rows() -> list[tuple[str, str]]:
    pattern = re.compile(r"^\|\s*`(\w+)`\s*\|[^|]*\|\s*([^|]+?)\s*\|", re.MULTILINE)
    return pattern.findall(README.read_text(encoding="utf-8"))


def test_readme_config_table_matches_dataclass_defaults() -> None:
    """Every default in the README table must equal the dataclass default."""
    cfg = DetectionConfig()
    rows = _readme_config_rows()
    assert rows, "no configuration rows parsed from README -- has the table moved?"

    checked = 0
    mismatches = []
    for name, documented in rows:
        if not hasattr(cfg, name) or name in _PROSE_DEFAULTS:
            continue
        checked += 1
        actual = getattr(cfg, name)
        if _normalise(documented) != _normalise(str(actual)):
            mismatches.append(f"{name}: README says {documented!r}, actual {actual!r}")

    assert not mismatches, "README config table is stale:\n  " + "\n  ".join(mismatches)
    assert checked >= 10, f"only {checked} parameters cross-checked; parsing likely broke"


@pytest.mark.parametrize(
    "field",
    ["ratio_threshold", "distance_low", "distance_high", "switch_frequency", "max_peaks"],
)
def test_algorithm_doc_default_list_matches_dataclass(field: str) -> None:
    """docs/algorithm.md lists the defaults explicitly; keep them in step."""
    text = ALGORITHM_DOC.read_text(encoding="utf-8")
    match = re.search(rf"^- `{field} = ([^`]+)`", text, re.MULTILINE)
    assert match, f"no default entry for {field} in docs/algorithm.md"

    actual = getattr(DetectionConfig(), field)
    assert _normalise(match.group(1)) == _normalise(str(actual)), (
        f"docs/algorithm.md says {field} = {match.group(1)}, actual {actual}"
    )
