"""Refactored dataset class focused on data loading and preprocessing only."""

from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from .base_dataset import BaseDataset
from .column_definitions import LifeExpectancyColumn as Col


class LifeExpectancyDataset(BaseDataset):
    """Loading, preprocessing and normalization for the Life Expectancy dataset."""

    Col = Col  # Set the column enum class
    default_target_column = Col.TARGET.value
    identifier_columns = (Col.COUNTRY.value, Col.STATUS.value, Col.YEAR.value)

    @classmethod
    def from_csv(
        cls,
        filepath: str | Path,
        *,
        aggregate_by_country: bool = True,
        drop_missing_target: bool = True,
    ) -> "LifeExpectancyDataset":
        """Load and preprocess the Life Expectancy dataset from a CSV file.

        - Normalize column names
        - Convert data types

        Args:
            filepath: Path to the CSV file
            aggregate_by_country: If True, aggregate data by country (mean across years)
            drop_missing_target: If True, drop rows with missing life expectancy values

        Returns:
            LifeExpectancyDataset instance with loaded and cleaned data
        """
        filepath = Path(filepath)

        df = pd.read_csv(filepath).pipe(cls._normalize_col_names).pipe(cls._convert_data_types)

        if drop_missing_target:
            df = df.dropna(subset=[Col.TARGET.value])

        if aggregate_by_country:
            df = cls._aggregate_by_country(df)

        return cls(df=df)

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
        """Set appropriate data types for each col."""
        identifier_cols = {Col.COUNTRY.value, Col.STATUS.value, Col.YEAR.value}
        numeric_cols = df.columns.difference(list(identifier_cols))

        converted = df.assign(
            year=pd.to_datetime(df[Col.YEAR.value].astype(str), format="%Y", errors="coerce"),
        )
        return converted.assign(
            **{col: pd.to_numeric(converted[col], errors="coerce") for col in numeric_cols},
        )

    @staticmethod
    def _aggregate_by_country(df: pd.DataFrame, agg_fn: Callable | str | None = None) -> pd.DataFrame:
        """Aggregate data by country, taking mean of numeric columns and imputing missing values.

        Args:
            df: Input DataFrame with multiple observations per country
            agg_fn: Aggregation function or string (defaults to mean)

        Returns:
            Country-aggregated DataFrame with imputed missing values
        """
        agg_fn = agg_fn or "mean"

        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.difference([Col.COUNTRY.value])
        non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.difference([Col.COUNTRY.value])

        agg_dict = {
            **dict.fromkeys(numeric_cols, agg_fn),
            **dict.fromkeys(non_numeric_cols, "first"),
        }

        aggregated = df.groupby(Col.COUNTRY.value, as_index=False).agg(agg_dict)
        means = aggregated.loc[:, numeric_cols].mean()
        aggregated.loc[:, numeric_cols] = aggregated.loc[:, numeric_cols].fillna(means)
        return aggregated
