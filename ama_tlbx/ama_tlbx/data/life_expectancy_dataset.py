"""Refactored dataset class focused on data loading and preprocessing only."""

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

import pandas as pd

from ama_tlbx.utils.paths import get_dataset_path

from .base_dataset import BaseDataset
from .life_expectancy_columns import LifeExpectancyColumn as Col


class LifeExpectancyDataset(BaseDataset):
    """Loading, preprocessing and normalization for the [Life Expectancy dataset](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who).

    **Example workflows**:
    >>> from ama_tlbx.data import LifeExpectancyDataset, LECol
    >>> from ama_tlbx.analysis import FeatureGroup
    >>> from ama_tlbx.plotting.correlation_plots import (
    ...     plot_correlation_heatmap,
    ...     plot_top_correlated_pairs,
    ...     plot_target_correlations,
    ... )
    >>> ds = LifeExpectancyDataset.from_csv()
    >>> corr = ds.make_correlation_analyzer(standardized=True, include_target=False).fit().result()
    >>> pca = ds.make_pca_analyzer(standardized=True, exclude_target=True).fit().result()
    >>> groups = [
    ...     FeatureGroup("immunization", [LECol.POLIO, LECol.HEPATITIS_B, LECol.DIPHTHERIA]),
    ...     FeatureGroup("child_mortality", [LECol.INFANT_DEATHS, LECol.UNDER_FIVE_DEATHS]),
    ... ]
    >>> dimred = ds.make_pca_dim_reduction_analyzer(groups, min_var_explained=0.9).fit().result()
    >>> iqr = ds.make_iqr_outlier_detector(threshold=1.5).fit().result()
    >>> corr.matrix.shape, pca.scores.shape, dimred.reduced_df.shape, iqr.outlier_mask.shape


    Using a modified DataFrame (that still has a subset of the original columns) with analyzers:

    >>> cov_cols = [LECol.POLIO, LECol.HEPATITIS_B, LECol.DIPHTHERIA]
    >>> le_ds = LifeExpectancyDataset.from_csv()
    >>> df = le_ds.df.assign(**{col: 1 - le_ds.df[col] for col in cov_cols})
    >>> corr_result = (
    ...     LifeExpectancyDataset(df[[*cov_cols, LECol.TARGET]])
    ...     .make_correlation_analyzer()
    ...     .fit()
    ...     .result()
    ... )
    >>> _ = plot_correlation_heatmap(corr_result, figsize=(16, 16))
    >>> _ = plot_top_correlated_pairs(corr_result, n=15, threshold=0.7)
    >>> _ = plot_target_correlations(result=corr_result)
    """

    Col = Col

    @classmethod
    def from_csv(
        cls,
        *,
        csv_path: str | Path | None = None,
        aggregate_by_country: bool | Literal["mean", 2014] = True,
        drop_missing_target: bool = True,
        resolve_nand_pred: Literal["drop", "median"] = "drop",
    ) -> "LifeExpectancyDataset":
        """Load and preprocess the Life Expectancy dataset from a CSV file.

        - Normalize column names
        - Convert data types

        Args:
            csv_path: Path to the CSV file
            aggregate_by_country: aggregate data by country (mean over years, label for other agg fn, or selected year) defaults to selecting year 2014 based on previous analysis.
            drop_missing_target: If True, drop rows with missing life expectancy values
            resolve_nand_pred: Strategy for remaining NaNs in predictors when no aggregation is performed.
                - "drop": drop rows with any missing predictor/target
                - "median": median-impute numeric predictors, then drop residual NaNs

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

        le_df = cls._resolve_missing_predictors(le_df, strategy=resolve_nand_pred)

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

    @staticmethod
    def _resolve_missing_predictors(
        df: pd.DataFrame,
        *,
        strategy: Literal["drop", "median"],
    ) -> pd.DataFrame:
        """Handle missing predictor values when no country aggregation is used.

        Args:
            df: Input DataFrame.
            strategy: ``"drop"`` to drop rows with any missing predictor/target,
                ``"median"`` to median-impute numeric predictors then drop remaining NaNs.

        Returns:
            Cleaned DataFrame according to the chosen strategy.
        """
        identifier_cols = {Col.COUNTRY, Col.YEAR}
        pred_cols = df.columns.difference(list(identifier_cols))

        if strategy == "drop":
            return df.dropna(subset=pred_cols)

        numeric_cols = df.select_dtypes(include=["number"]).columns.difference(list(identifier_cols))
        medians = df[numeric_cols].median()
        imputed = df.copy()
        imputed.loc[:, numeric_cols] = imputed.loc[:, numeric_cols].fillna(medians)
        # Drop any rows still carrying NaNs (e.g., non-numeric leftovers)
        return imputed.dropna(subset=pred_cols)

    @property
    def numeric_cols(self) -> pd.Index:
        return super().numeric_cols.difference([Col.YEAR, Col.STATUS])

    def feature_columns(
        self,
        include_target: bool = False,
        extra_exclude: Iterable[str] | None = [Col.YEAR],
        extra_include: Iterable[str] | None = [Col.STATUS],
    ) -> list[str]:
        return super().feature_columns(
            include_target=include_target,
            extra_exclude=extra_exclude,
            extra_include=extra_include,
        )

    def tf_and_norm(self, tf_map: dict[Col, Callable] | None = None) -> pd.DataFrame:
        """Apply per-column transformations then z-score numeric columns.

        - Uses the default transforms from :class:`LifeExpectancyColumn` metadata.
        - Allows overrides via ``tf_map`` (set a value to ``None`` to drop a default).
        - Never transforms or standardizes the year column or the binary ``status`` indicator.

        Args:
            tf_map: Optional mapping from ``LifeExpectancyColumn`` to a callable that
                accepts and returns a ``pd.Series``. Custom entries override defaults.

        Returns:
            DataFrame with transforms applied and numeric columns standardized
            (mean=0, std=1) except for ``year`` and ``status``.
        """
        df = self.df.copy()

        # Defaults from metadata
        default_tf: dict[Col, Callable] = {col: col.transform for col in Col if col.transform is not None}

        # Apply overrides; None means remove the default transform
        transforms: dict[Col, Callable] = default_tf.copy()
        for col, fn in (tf_map or {}).items():
            if fn is None:
                transforms.pop(col, None)
            else:
                transforms[col] = fn

        for col, transform in transforms.items():
            if col in df.columns:
                transformed = transform(df[col])
                if isinstance(transformed, pd.DataFrame):
                    # drop original and join expanded columns (e.g., dummies, splines)
                    df = df.drop(columns=[col]).join(transformed)
                else:
                    df[col] = transformed

        # Recompute numeric columns after transformations (status/year excluded)
        numeric_cols = (
            df.select_dtypes(include=["number"])
            .columns.difference([self.Col.STATUS, self.Col.YEAR])
            .tolist()
        )
        if self.Col.TARGET in numeric_cols:
            numeric_cols.remove(self.Col.TARGET)

        if numeric_cols:
            # Median-impute numeric cols to avoid NaNs breaking StandardScaler
            df.loc[:, numeric_cols] = df.loc[:, numeric_cols].apply(lambda s: s.fillna(s.median()))
            df.loc[:, numeric_cols] = self.standardize(df)[numeric_cols]

        return df
