"""Utility helpers for dataset reporting and metadata."""

from __future__ import annotations

import pandas as pd


def parse_column_docstrings(doc: str | None) -> pd.DataFrame:
    """Parse column docstrings into a structured table.

    Args:
        doc: Docstring containing lines in the format
            ``- ``column``: dtype - description``.

    Returns:
        DataFrame with columns ``column``, ``type``, and ``description``.
    """
    rows: list[dict[str, str]] = []
    for raw_line in (doc or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("- ``") or " - " not in line:
            continue
        left, desc = line.split(" - ", 1)
        name_part, dtype_part = left.split(": ", 1)
        name = name_part.replace("- ``", "").replace("``", "").strip()
        rows.append(
            {"column": name, "type": dtype_part.strip(), "description": desc.strip()},
        )
    return pd.DataFrame(rows)
