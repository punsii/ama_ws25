"""Refactored dataset class focused on data loading and preprocessing only."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from .base_dataset import BaseDataset
from .column_definitions import LifeExpectancyColumn as Col


class LifeExpectancyDataset(BaseDataset):
    """Loading, preprocessing and normalization for the Life Expectancy dataset.

    Columns:
    - country: str - Country name
    - year: datetime - Year of observation
    - status: str - Development status (Developing/Developed)
    - life_expectancy: float - Life expectancy in years (target variable)
    - adult_mortality: float - Adult mortality rate per 1000 population
    - infant_deaths: int - Number of infant deaths per 1000 population
    - alcohol: float - Alcohol consumption per capita (liters)
    - percentage_expenditure: float - Health expenditure as % of GDP per capita
    - hepatitis_b: float - Hepatitis B immunization coverage (%)
    - measles: int - Number of measles cases per 1000 population
    - bmi: float - Average Body Mass Index
    - under-five_deaths: int - Deaths of children under 5 per 1000 population
    - polio: float - Polio immunization coverage (%)
    - total_expenditure: float - Government health expenditure (% of total govt expenditure)
    - diphtheria: float - Diphtheria immunization coverage (%)
    - hiv/aids: float - Deaths per 1000 live births due to HIV/AIDS (0-4 years)
    - gdp: float - Gross Domestic Product per capita (USD)
    - population: int - Population of the country
    - thinness_1-19_years: float - Prevalence of thinness among children 10-19 years (%)
    - thinness_5-9_years: float - Prevalence of thinness among children 5-9 years (%)
    - income_composition_of_resources: float - Human Development Index (0-1)
    - schooling: float - Average years of schooling
    """

    target_col: Col = Col.LIFE_EXPECTANCY

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
            df = df.dropna(subset=[Col.LIFE_EXPECTANCY.value])

        if aggregate_by_country:
            df = cls._aggregate_by_country(df)

        return cls(df=df)

    @staticmethod
    def _normalize_col_names(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to match LifeExpectancyColumn enum; converts original column names to snake_case."""
        return df.set_axis(
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
            .str.replace("__+", "_", regex=True)
            .str.replace("/", "_", regex=False)
            .str.replace("-", "_", regex=False),
            axis=1,
        ).rename(
            columns={
                "hiv_aids": Col.HIV_AIDS.value,
                "thinness__1_19_years": Col.THINNESS_1_19_YEARS.value,
                "thinness_1_19_years": Col.THINNESS_1_19_YEARS.value,
                "thinness__5_9_years": Col.THINNESS_5_9_YEARS.value,
                "thinness_5_9_years": Col.THINNESS_5_9_YEARS.value,
                "under_five_deaths": Col.UNDER_FIVE_DEATHS.value,
            },
        )

    @staticmethod
    def _convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
        """Set appropriate data types for each col."""
        identifier_cols = {Col.COUNTRY.value, Col.STATUS.value, Col.YEAR.value}

        return df.assign(
            year=lambda d: pd.to_datetime(d[Col.YEAR.value].astype(str), format="%Y", errors="coerce"),
        ).pipe(
            lambda d: d.assign(
                **{col: pd.to_numeric(d[col], errors="coerce") for col in d.columns if col not in identifier_cols},
            ),
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
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

        # Remove country from lists if present
        if Col.COUNTRY.value in numeric_cols:
            numeric_cols.remove(Col.COUNTRY.value)
        if Col.COUNTRY.value in non_numeric_cols:
            non_numeric_cols.remove(Col.COUNTRY.value)

        # Build aggregation dict: numeric columns get agg_fn, non-numeric get first value
        agg_dict = dict.fromkeys(numeric_cols, agg_fn)
        agg_dict.update(dict.fromkeys(non_numeric_cols, "first"))

        # Aggregate
        aggregated = df.groupby(Col.COUNTRY.value, as_index=False).agg(agg_dict)

        # Impute missing values with mean for numeric columns
        for col in numeric_cols:
            aggregated[col] = aggregated[col].fillna(aggregated[col].mean())

        return aggregated

    def get_pretty_name(self, column_name: str) -> str:
        """Convert column name to pretty name for visualization.

        Args:
            column_name: The cleaned column name

        Returns:
            Pretty name suitable for plot labels and titles
        """
        try:
            col_enum = Col(column_name)
        except ValueError:
            # Fallback: capitalize and replace underscores if not in enum
            return column_name.replace("_", " ").title()
        else:
            return str(col_enum.pretty_name)
