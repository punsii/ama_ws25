"""Refactored dataset class focused on data loading and preprocessing only."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pandas as pd

from ama_tlbx.utils.paths import get_dataset_path

from .base_dataset import BaseDataset
from .life_expectancy_columns import LifeExpectancyColumn as Col


class LifeExpectancyDataset(BaseDataset):
    """Loading, preprocessing and normalization for the [Life Expectancy dataset](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)."""

    Col = Col

    @classmethod
    def from_csv(
        cls,
        *,
        csv_path: str | Path | None = None,
        aggregate_by_country: bool | Literal["mean", 2014] = True,
        drop_missing_target: bool = True,
    ) -> "LifeExpectancyDataset":
        """Load and preprocess the Life Expectancy dataset from a CSV file.

        - Normalize column names
        - Convert data types

        Args:
            csv_path: Path to the CSV file
            aggregate_by_country: aggregate data by country (mean over years, label for other agg fn, or selected year)
                 defaults to selecting year 2014 based on previous analysis.
            drop_missing_target: If True, drop rows with missing life expectancy values

        Returns:
            LifeExpectancyDataset instance with loaded and cleaned data
        """
        csv_path = get_dataset_path("life_expectancy") if csv_path is None else Path(csv_path)

        le_df = pd.read_csv(csv_path).pipe(cls._normalize_col_names).pipe(cls._convert_data_types)

        if drop_missing_target:
            le_df = le_df.dropna(subset=[Col.TARGET])

        if aggregate_by_country:
            le_df = cls._aggregate_by_country(
                le_df,
                agg_by=2007 if aggregate_by_country is True else aggregate_by_country,
            )
            le_df.index.name = Col.COUNTRY

        return cls(df=le_df)

    @staticmethod
    def _normalize_col_names(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to match LifeExpectancyColumn enum.

        Converts original column names to regular snake_case by:
        Strip whitespace, convert to lowercase, replace spaces/slashes/hyphens with underscores, collapse multiple underscores
        """
        return df.set_axis(
            df.columns.str.strip()
            .str.lower()
            .str.replace(r"[\s/\-]+", "_", regex=True)
            .str.replace(r"_+", "_", regex=True),
            axis=1,
        )

    @staticmethod
    def _convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
        """Set appropriate data types for each col.

        Converts STATUS to binary: 0 = Developing, 1 = Developed.
        """
        identifier_cols = {Col.COUNTRY, Col.YEAR}
        numeric_cols = df.columns.difference(list(identifier_cols))

        converted = df.assign(
            year=pd.to_datetime(df[Col.YEAR].astype(str), format="%Y", errors="coerce"),
            status=(df[Col.STATUS].str.strip().str.lower() == "developed").astype(int),
        )
        return converted.assign(
            **{col: pd.to_numeric(converted[col], errors="coerce") for col in numeric_cols if col != Col.STATUS},
        )

    @staticmethod
    def _aggregate_by_country(df: pd.DataFrame, agg_by: Callable | str | int | None = None) -> pd.DataFrame:
        """Aggregate data by country, taking mean of numeric columns and imputing missing values.

        Args:
            df: Input DataFrame with multiple observations per country
            agg_by: Aggregation function or string (defaults to mean)

        Returns:
            Country-aggregated DataFrame with imputed missing values
        """
        if isinstance(agg_by, int):
            return df.assign(year=lambda d: d[Col.YEAR].dt.year).query("year == @agg_by").set_index(Col.COUNTRY)

        agg_by = agg_by or "mean"

        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.difference([Col.COUNTRY])
        non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.difference([Col.COUNTRY, Col.STATUS])

        agg_dict = {
            **dict.fromkeys(numeric_cols, agg_by),
            **dict.fromkeys(non_numeric_cols, "first"),
        }

        aggregated = df.groupby(Col.COUNTRY, as_index=False).agg(agg_dict)
        means = aggregated.loc[:, numeric_cols].mean()
        aggregated.loc[:, numeric_cols] = aggregated.loc[:, numeric_cols].fillna(means)
        return aggregated
