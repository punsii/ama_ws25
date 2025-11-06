"""Refactored dataset class focused on data loading and preprocessing only."""

from collections.abc import Callable, Mapping
from pathlib import Path

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
        aggregate_by_country: bool = True,
        drop_missing_target: bool = True,
    ) -> "LifeExpectancyDataset":
        """Load and preprocess the Life Expectancy dataset from a CSV file.

        - Normalize column names
        - Convert data types

        TODO(@jd): Add aggregate_by_year parameter. https://github.com/punsii/ama_ws25/issues/TBD

        Args:
            csv_path: Path to the CSV file
            aggregate_by_country: If True, aggregate data by country (mean across years)
            drop_missing_target: If True, drop rows with missing life expectancy values

        Returns:
            LifeExpectancyDataset instance with loaded and cleaned data
        """
        csv_path = get_dataset_path("life_expectancy") if csv_path is None else Path(csv_path)

        le_df = pd.read_csv(csv_path).pipe(cls._normalize_col_names).pipe(cls._convert_data_types)

        if drop_missing_target:
            le_df = le_df.dropna(subset=[Col.TARGET])

        if aggregate_by_country:
            le_df = cls._aggregate_by_country(le_df)

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
        """Set appropriate data types for each col."""
        identifier_cols = {Col.COUNTRY, Col.STATUS, Col.YEAR}
        numeric_cols = df.columns.difference(list(identifier_cols))

        converted = df.assign(
            year=pd.to_datetime(df[Col.YEAR].astype(str), format="%Y", errors="coerce"),
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
        numeric_cols = df.select_dtypes(include=["number"]).columns.difference([Col.COUNTRY])
        non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.difference([Col.COUNTRY])

        agg_dict = {
            **dict.fromkeys(numeric_cols, agg_fn),
            **dict.fromkeys(non_numeric_cols, "first"),
        }

        aggregated = df.groupby(Col.COUNTRY, as_index=False).agg(agg_dict)
        means = aggregated.loc[:, numeric_cols].mean()
        aggregated.loc[:, numeric_cols] = aggregated.loc[:, numeric_cols].fillna(means)
        return aggregated

    def select_representative_year(
        self,
        *,
        detector: str = "iqr",
        detector_kwargs: Mapping[str, object] | None = None,
        agg_fn: Callable[[pd.Series], float] | str = "median",
        prefer_recent: bool = False,
    ) -> int:
        """Select a representative calendar year for downstream analyses.

        The method aggregates numeric features per calendar year, flags year-level profiles that
        behave like outliers, and returns the year whose aggregated profile is closest to the overall
        median among the remaining candidates. When all years are flagged as outliers, the year with
        the fewest outlier flags is considered instead.

        Args:
            detector: Outlier detection strategy. Supported values are ``iqr``, ``zscore``, and
                ``isolation_forest``.
            detector_kwargs: Optional keyword arguments forwarded to the detector constructor.
            agg_fn: Aggregation function applied to numeric columns before scoring (default: ``median``).
            prefer_recent: When True, prefer the most recent year if multiple candidates share the same
                score.

        Returns:
            Integer calendar year deemed most representative.

        Raises:
            ValueError: If no numeric data is available to determine a representative year.

        Examples:
            >>> dataset = LifeExpectancyDataset.from_csv(aggregate_by_country=False)
            >>> dataset.select_representative_year()
            2011
        """
        if self.Col.YEAR not in self.df.columns:
            raise ValueError("LifeExpectancyDataset requires a year column to select a representative year.")

        numeric_columns = list(self.numeric_cols)
        per_year = (
            self.df.assign(_year=self.df[self.Col.YEAR].dt.year)
            .groupby("_year")[numeric_columns]
            .agg(agg_fn)
            .sort_index()
            .rename_axis(self.Col.YEAR, axis=0)
        )

        if per_year.empty:
            raise ValueError("No numeric data available to evaluate a representative year.")

        per_year = per_year.fillna(per_year.median())
        from ama_tlbx.analysis.outlier_detector import (
            IQROutlierDetector,
            IsolationForestOutlierDetector,
            ZScoreOutlierDetector,
        )
        from ama_tlbx.data.views import DatasetView

        year_view = DatasetView(
            df=per_year,
            pretty_by_col={col: self.get_pretty_name(col) for col in per_year.columns},
            numeric_cols=list(per_year.columns),
            target_col=None,
        )

        detector_kwargs = dict(detector_kwargs or {})
        detector_name = detector.lower()
        if detector_name == "iqr":
            model = IQROutlierDetector(view=year_view, **detector_kwargs)
        elif detector_name == "zscore":
            model = ZScoreOutlierDetector(view=year_view, **detector_kwargs)
        elif detector_name == "isolation_forest":
            model = IsolationForestOutlierDetector(view=year_view, **detector_kwargs)
        else:
            raise ValueError(
                f"Unsupported detector '{detector}'. Use one of: 'iqr', 'zscore', 'isolation_forest'.",
            )

        result = model.fit().result()
        outlier_counts = result.n_outliers_per_row

        candidate_index = outlier_counts.loc[outlier_counts == 0].index
        if candidate_index.empty:
            min_outliers = int(outlier_counts.min())
            candidate_index = outlier_counts.loc[outlier_counts == min_outliers].index

        center = per_year.median()
        distances = per_year.sub(center).pow(2).mean(axis=1)
        candidate_distances = distances.loc[candidate_index]
        min_distance = candidate_distances.min()
        top_candidates = candidate_distances.loc[candidate_distances == min_distance].index

        if prefer_recent:
            selected_year = int(top_candidates.max())
        else:
            selected_year = int(top_candidates.min())

        return selected_year
