"""Utility helpers for dataset reporting and metadata."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


DEFAULT_UNDP_SOCIO_COLUMNS: tuple[str, ...] = (
    "gnipc",
    "eys",
    "mys",
    "gdi",
    "gii",
    "coef_ineq",
    "loss",
    "ineq_edu",
    "ineq_inc",
    "pop_total",
    "co2_prod",
    "mmr",
    "abr",
    "se_f",
    "se_m",
    "pr_f",
    "pr_m",
    "lfpr_f",
    "lfpr_m",
)


def merge_undp_hdr_features(
    le_dataset,
    *,
    undp_dataset=None,
    years: Iterable[int] | None = None,
    columns: Iterable[str] | None = None,
    how: str = "left",
    add_iso3: bool = True,
):
    """Merge UNDP HDR features into a Life Expectancy dataset.

    This is a small convenience wrapper around ``LifeExpectancyDataset.merge_undp`` that:
      - loads UNDP HDR data if not provided
      - filters years and columns
      - returns a LifeExpectancyDataset instance with merged UNDP columns

    Args:
        le_dataset: LifeExpectancyDataset or DataFrame compatible with it.
        undp_dataset: Optional UNDPHDRDataset or DataFrame. If None, it is loaded via
            ``UNDPHDRDataset.from_csv(years=years)``.
        years: Optional iterable of years to load/filter the UNDP dataset.
        columns: Optional iterable of UNDP column names to merge (excluding iso3/year).
            If None, uses DEFAULT_UNDP_SOCIO_COLUMNS.
        how: Merge strategy (default: left).
        add_iso3: Whether to ensure ISO3 is present in the Life Expectancy dataset.

    Returns:
        LifeExpectancyDataset with UNDP HDR features merged in.
    """
    from .life_expectancy_dataset import LifeExpectancyDataset
    from .undp_hdr_columns import UNDPHDRColumn as HDRCol
    from .undp_hdr_dataset import UNDPHDRDataset

    if isinstance(le_dataset, pd.DataFrame):
        le_dataset = LifeExpectancyDataset(df=le_dataset)
    if not isinstance(le_dataset, LifeExpectancyDataset):
        raise TypeError("le_dataset must be a LifeExpectancyDataset or pd.DataFrame.")

    if undp_dataset is None:
        undp_dataset = UNDPHDRDataset.from_csv(years=years)

    if isinstance(undp_dataset, UNDPHDRDataset):
        undp_df = undp_dataset.df
    elif isinstance(undp_dataset, pd.DataFrame):
        undp_df = undp_dataset
    else:
        raise TypeError("undp_dataset must be a UNDPHDRDataset or pd.DataFrame.")

    keep_cols = [str(HDRCol.ISO3), str(HDRCol.YEAR)]
    cols = list(columns) if columns is not None else list(DEFAULT_UNDP_SOCIO_COLUMNS)
    cols = [c for c in cols if c in undp_df.columns]
    undp_df = undp_df.loc[:, [*keep_cols, *cols]].copy()

    if years is not None and str(HDRCol.YEAR) in undp_df.columns:
        undp_df = undp_df[undp_df[str(HDRCol.YEAR)].isin(list(years))]

    return le_dataset.merge_undp(
        undp_df,
        how=how,
        add_iso3=add_iso3,
    )


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
