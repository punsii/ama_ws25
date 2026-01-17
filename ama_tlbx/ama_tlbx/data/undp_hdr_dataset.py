"""Dataset loader for UNDP HDR composite time series data."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ama_tlbx.utils.paths import get_dataset_path

from .base_dataset import BaseDataset
from .undp_hdr_columns import UNDPHDRColumn as Col


class UNDPHDRDataset(BaseDataset):
    """Load and reshape UNDP HDR time series data to a long format."""

    Col = Col

    @classmethod
    def from_csv(
        cls,
        *,
        csv_path: str | Path | None = None,
        years: Iterable[int] = range(2000, 2016),
    ) -> UNDPHDRDataset:
        """Load UNDP HDR time series data and return long-format dataset.

        Args:
            csv_path: Path to the HDR CSV file. Defaults to bundled dataset.
            years: Years to extract from the wide HDR time series.
        """
        csv_path = get_dataset_path("undp_hdr") if csv_path is None else Path(csv_path)
        assert csv_path.exists(), f"CSV file not found at: {csv_path}"

        years_set = {int(y) for y in years}

        prefix_map = Col.time_series_prefix_map()

        def usecols(col: str) -> bool:
            c = col.lower()
            if c in ("iso3", "country", "hdicode", "region"):
                return True
            for prefix in prefix_map:
                prefix_token = f"{prefix}_"
                if not c.startswith(prefix_token):
                    continue
                suffix = c.removeprefix(prefix_token)
                if not suffix.isdigit():
                    return False
                return int(suffix) in years_set
            return False

        hdr_df = pd.read_csv(csv_path, encoding="latin1", usecols=usecols).rename(columns=str.lower)

        positive_only = {str(Col.GNI_PC), str(Col.GNI_PC_F), str(Col.GNI_PC_M)}
        series_frames: list[pd.DataFrame] = []
        for prefix, col in prefix_map.items():
            series_frames.append(
                cls._to_long(
                    hdr_df,
                    prefix=f"{prefix}_",
                    value_name=str(col),
                    positive_only=str(col) in positive_only,
                ),
            )

        base_cols = [c for c in ["iso3", "country", "hdicode", "region"] if c in hdr_df.columns]
        base = hdr_df[base_cols].drop_duplicates()

        if not series_frames:
            merged = base
        else:
            merged = series_frames[0]
            for frame in series_frames[1:]:
                merged = merged.merge(frame, on=[Col.ISO3, Col.YEAR], how="outer")
            if not base.empty:
                merged = merged.merge(base, on=Col.ISO3, how="left")

        ordered_cols = [Col.ISO3, Col.COUNTRY, Col.HDICODE, Col.REGION, Col.YEAR]
        ordered_cols += list(prefix_map.values())
        merged = merged.loc[:, [c for c in ordered_cols if c in merged.columns]]

        numeric_cols = [str(col) for col in prefix_map.values() if str(col) in merged.columns]
        merged = merged.assign(
            **{
                Col.YEAR: pd.to_numeric(merged[Col.YEAR], errors="coerce").astype("Int64"),
                **{col: pd.to_numeric(merged[col], errors="coerce") for col in numeric_cols},
            },
        )
        merged = merged.dropna(subset=[Col.YEAR]).assign(**{Col.YEAR: merged[Col.YEAR].astype(int)})

        return cls(df=merged)

    @staticmethod
    def _to_long(
        df: pd.DataFrame,
        *,
        prefix: str,
        value_name: str,
        positive_only: bool = False,
    ) -> pd.DataFrame:
        cols = [c for c in df.columns if c.startswith(prefix)]
        long_df = (
            df[["iso3", *cols]]
            .melt(id_vars=["iso3"], var_name=Col.YEAR, value_name=value_name)
            .assign(**{Col.YEAR: lambda d: d[Col.YEAR].str.replace(prefix, "", regex=False).astype(int)})
            .assign(**{value_name: lambda d: pd.to_numeric(d[value_name], errors="coerce")})
        )
        if positive_only:
            long_df = long_df.assign(**{value_name: lambda d: d[value_name].where(d[value_name] > 0)})
        return long_df.dropna(subset=[value_name])

    def with_iso3(self, *, overwrite: bool = False) -> pd.DataFrame:
        """Return a copy of the dataset with ISO3 codes derived from country names."""
        return self.add_iso3(country_col=Col.COUNTRY, iso3_col=Col.ISO3, overwrite=overwrite)

    def merge_life_expectancy(
        self,
        le_dataset: LifeExpectancyDataset,
        *,
        how: str = "inner",
        add_iso3: bool = True,
    ) -> pd.DataFrame:
        """Merge UNDP HDR data with the Life Expectancy dataset on ISO3 + year."""
        from .life_expectancy_columns import LifeExpectancyColumn as LECol
        from .life_expectancy_dataset import LifeExpectancyDataset

        if not isinstance(le_dataset, LifeExpectancyDataset):
            raise TypeError("le_dataset must be a LifeExpectancyDataset instance.")

        le_df = le_dataset.df
        if add_iso3:
            le_df = le_dataset.with_iso3()
        le_df = le_df.assign(year=lambda d: d[LECol.YEAR].dt.year.astype(int))
        return le_df.merge(self.df, on=[Col.ISO3, Col.YEAR], how=how)
