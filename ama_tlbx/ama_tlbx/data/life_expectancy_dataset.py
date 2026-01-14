"""Refactored dataset class focused on data loading and preprocessing only."""

import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal
from warnings import warn

import pandas as pd

from ama_tlbx.utils.paths import get_dataset_path

from .base_dataset import BaseDataset
from .life_expectancy_columns import LifeExpectancyColumn as Col


_COLNAME_DELIMS_RE = re.compile(r"[\s/\-]+")
_COLNAME_MULTI_UNDERSCORE_RE = re.compile(r"_+")


def _normalize_col_name(name: object) -> str:
    s = name if isinstance(name, str) else str(name)
    s = s.strip().lower()
    s = _COLNAME_DELIMS_RE.sub("_", s)
    s = _COLNAME_MULTI_UNDERSCORE_RE.sub("_", s)
    return s


_RAW_COLNAME_TO_ENUM: dict[str, str] = {_normalize_col_name(col.metadata().original_name): str(col) for col in Col}


class LifeExpectancyDataset(BaseDataset):
    """Loading, preprocessing and normalization for the [Life Expectancy dataset](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who).

    **Example workflows**:
    >>> from ama_tlbx.data import LifeExpectancyDataset, Col
    >>> from ama_tlbx.analysis import FeatureGroup
    >>> ds = LifeExpectancyDataset.from_csv()
    >>> corr = ds.make_correlation_analyzer(standardized=True, include_target=False).fit().result()
    >>> pca = ds.make_pca_analyzer(standardized=True, exclude_target=True).fit().result()
    >>> groups = [
    ...     FeatureGroup("immunization", [Col.POLIO, Col.HEPATITIS_B, Col.DIPHTHERIA]),
    ...     FeatureGroup("child_mortality", [Col.INFANT_DEATHS, Col.UNDER_FIVE_DEATHS]),
    ... ]
    >>> dimred = ds.make_pca_dim_reduction_analyzer(groups, min_var_explained=0.9).fit().result()
    >>> iqr = ds.make_iqr_outlier_detector(threshold=1.5).fit().result()
    >>> corr.matrix.shape, pca.scores.shape, dimred.reduced_df.shape, iqr.outlier_mask.shape


    Using a modified DataFrame (that still has a subset of the original columns) with analyzers:

    >>> cov_cols = [Col.POLIO, Col.HEPATITIS_B, Col.DIPHTHERIA]
    >>> le_ds = LifeExpectancyDataset.from_csv()
    >>> df = le_ds.df.assign(**{col: 1 - le_ds.df[col] for col in cov_cols})
    >>> corr_result = (
    ...     LifeExpectancyDataset(df[[*cov_cols, Col.TARGET]])
    ...     .make_correlation_analyzer()
    ...     .fit()
    ...     .result()
    ... )
    >>> _ = corr_result.plot_heatmap(figsize=(16, 16))
    >>> _ = corr_result.plot_top_pairs(n=15, threshold=0.7)
    >>> _ = corr_result.plot_target_correlations()
    """

    Col = Col

    def with_iso3(self, *, overwrite: bool = False) -> pd.DataFrame:
        """Return a copy of the dataset with ISO3 codes derived from country names."""
        return self.add_iso3(country_col=Col.COUNTRY, iso3_col="iso3", overwrite=overwrite)

    def merge_undp(
        self,
        undp_dataset: "UNDPHDRDataset",
        *,
        how: str = "inner",
        add_iso3: bool = True,
    ) -> pd.DataFrame:
        """Merge Life Expectancy data with UNDP HDR data on ISO3 + year."""
        from .undp_hdr_columns import UNDPHDRColumn as UCol
        from .undp_hdr_dataset import UNDPHDRDataset

        if not isinstance(undp_dataset, UNDPHDRDataset):
            raise TypeError("undp_dataset must be a UNDPHDRDataset instance.")

        le_df = self.df
        if add_iso3:
            le_df = self.with_iso3()
        le_df = le_df.assign(year=lambda d: d[Col.YEAR].dt.year.astype(int))
        return le_df.merge(undp_dataset.df, on=[UCol.ISO3, UCol.YEAR], how=how)

    @classmethod
    def from_csv(
        cls,
        *,
        csv_path: str | Path | None = None,
        aggregate_by_country: bool | str | int | Callable | None = True,
        drop_missing_target: bool = True,
        resolve_nand_pred: Literal["drop", "median", "mean", "carry_forward"] | bool = "carry_forward",
    ) -> "LifeExpectancyDataset":
        """Load and preprocess the Life Expectancy dataset from a CSV file.

        - Normalize column names
        - Convert data types

        Args:
            csv_path: Path to the CSV file
            aggregate_by_country: aggregate data by country (mean over years, label for other agg fn, or selected year).
                When ``True`` (default), the most recent valid year in the dataset is used.
            drop_missing_target: If True, drop rows with missing life expectancy values
            resolve_nand_pred: Strategy for remaining NaNs in predictors (applied after aggregation if enabled).
                - "carry_forward": forward-fill missing values per country using last valid observation
                - "drop": drop rows with any missing predictor/target
                - "median": median-impute numeric predictors, then drop residual NaNs
                - "mean": mean-impute numeric predictors, then drop residual NaNs

        Returns:
            LifeExpectancyDataset instance with loaded and cleaned data
        """
        csv_path = get_dataset_path("life_expectancy") if csv_path is None else Path(csv_path)
        assert csv_path.exists(), f"CSV file not found at: {csv_path}"

        le_df = pd.read_csv(csv_path).pipe(cls._normalize_col_names).pipe(cls._convert_data_types)

        if drop_missing_target:
            le_df = le_df.dropna(subset=[Col.TARGET])

        apply_carry_forward_pre_agg = resolve_nand_pred == "carry_forward" and isinstance(aggregate_by_country, int)
        if apply_carry_forward_pre_agg:
            # For year-specific slices, carry-forward needs the full panel history.
            le_df = cls._resolve_missing_predictors(
                le_df,
                strategy="carry_forward",
                drop_remaining=False,
            )

        if aggregate_by_country:
            agg_by = (
                cls._latest_valid_year(
                    le_df,
                    target_col=Col.TARGET if drop_missing_target else None,
                )
                if isinstance(aggregate_by_country, bool)
                else aggregate_by_country
            )
            le_df = cls._aggregate_by_country(
                le_df,
                agg_by=agg_by,
            )
            le_df.index.name = Col.COUNTRY

        if resolve_nand_pred and not apply_carry_forward_pre_agg:
            le_df = cls._resolve_missing_predictors(le_df, strategy=resolve_nand_pred)

        return cls(df=le_df)

    @staticmethod
    def _normalize_col_names(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to match LifeExpectancyColumn enum.

        Converts original column names to regular snake_case by:
        Strip whitespace, convert to lowercase, replace spaces/slashes/hyphens with underscores, collapse multiple underscores
        """
        normalized_cols = [_normalize_col_name(col) for col in df.columns]
        return df.set_axis(normalized_cols, axis=1).rename(columns=_RAW_COLNAME_TO_ENUM)

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
        """Aggregate data by country without imputing missing values.

        Args:
            df: Input DataFrame with multiple observations per country
            agg_by: Aggregation function or string (defaults to mean)

        Returns:
            Country-aggregated DataFrame (no imputation applied)
        """
        if isinstance(agg_by, int):
            filtered = df.assign(year=lambda d: d[Col.YEAR].dt.year).query("year == @agg_by").copy()
            return filtered.set_index(Col.COUNTRY)

        agg_by = agg_by or "mean"

        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.difference([Col.COUNTRY])
        non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.difference([Col.COUNTRY, Col.STATUS])

        agg_dict = {
            **dict.fromkeys(numeric_cols, agg_by),
            **dict.fromkeys(non_numeric_cols, "first"),
        }

        aggregated = df.groupby(Col.COUNTRY, as_index=False).agg(agg_dict)
        return aggregated

    @staticmethod
    def _latest_valid_year(df: pd.DataFrame, *, target_col: str | None = None) -> int:
        """Return the most recent year present in the dataset.

        If ``target_col`` is provided, only rows with a non-null target are
        considered when inferring the latest year. Raises ``ValueError`` when
        no valid year values are available.
        """
        years = df[Col.YEAR].dt.year
        if target_col is not None and target_col in df.columns:
            years = years[df[target_col].notna()]

        years = years.dropna()
        if years.empty:
            raise ValueError("Cannot infer latest year: no valid year values found.")

        return int(years.max())

    @staticmethod
    def _resolve_missing_predictors(
        df: pd.DataFrame,
        *,
        strategy: Literal["drop", "median", "mean", "carry_forward"] = "carry_forward",
        drop_remaining: bool = True,
    ) -> pd.DataFrame:
        """Handle missing predictor values (after optional aggregation).

        Args:
            df: Input DataFrame.
            strategy:
                - ``carry_forward`` to forward-fill missing values using the last valid observation per country (default),
                - ``"drop"`` to drop rows with any missing predictor/target,
                - ``"median"`` to median-impute numeric predictors then drop remaining NaNs.
                - ``"mean"`` to mean-impute numeric predictors then drop remaining NaNs.
            drop_remaining: When ``True`` (default), drop rows that still contain
                missing predictor values after applying the strategy.

        Returns:
            Cleaned DataFrame according to the chosen strategy.
        """
        identifier_cols = {Col.COUNTRY, Col.YEAR}
        pred_cols = df.columns.difference(list(identifier_cols))

        if strategy == "carry_forward":
            return LifeExpectancyDataset._carry_forward_predictors(
                df,
                pred_cols=pred_cols,
                drop_remaining=drop_remaining,
            )

        if strategy == "drop":
            return df.dropna(subset=pred_cols)

        numeric_cols = df.select_dtypes(include=["number"]).columns.difference(list(identifier_cols))
        fillers = df[numeric_cols].mean() if strategy == "mean" else df[numeric_cols].median()
        imputed = df.copy()
        imputed.loc[:, numeric_cols] = imputed.loc[:, numeric_cols].fillna(fillers)
        if drop_remaining:
            # Drop any rows still carrying NaNs (e.g., non-numeric leftovers)
            return imputed.dropna(subset=pred_cols)
        return imputed

    @staticmethod
    def _carry_forward_predictors(
        df: pd.DataFrame,
        *,
        pred_cols: pd.Index | list[str],
        drop_remaining: bool,
    ) -> pd.DataFrame:
        if len(pred_cols) == 0:
            return df

        filled = df.copy()

        # Forward fill within each country in chronological order.
        if Col.COUNTRY in filled.columns:
            # Avoid pandas ambiguity when index name equals country column.
            if filled.index.name == Col.COUNTRY:
                filled = filled.reset_index(drop=True)
            sort_cols: list[str] = [Col.COUNTRY]
            if Col.YEAR in filled.columns:
                sort_cols.append(Col.YEAR)
            filled = filled.sort_values(sort_cols)
            filled.loc[:, pred_cols] = filled.groupby(Col.COUNTRY, sort=False)[pred_cols].ffill()
            return filled.dropna(subset=pred_cols) if drop_remaining else filled

        if filled.index.name == Col.COUNTRY:
            if Col.YEAR in filled.columns:
                filled = filled.sort_values(Col.YEAR)
            filled.loc[:, pred_cols] = filled.groupby(level=0, sort=False)[pred_cols].ffill()
            return filled.dropna(subset=pred_cols) if drop_remaining else filled

        raise ValueError(
            "carry_forward strategy requires a country column or an index named 'country'.",
        )

    @property
    def numeric_cols(self) -> pd.Index:
        return super().numeric_cols.difference([Col.YEAR, Col.STATUS])

    def feature_columns(
        self,
        include_target: bool = False,
        extra_exclude: Iterable[str] | None = None,
        extra_include: Iterable[str] | None = None,
    ) -> list[str]:
        status_cols: list[str] = []
        if self.Col.STATUS in self.df.columns:
            status_cols.append(self.Col.STATUS)

        return super().feature_columns(
            include_target=include_target,
            extra_exclude=extra_exclude or [Col.YEAR],
            extra_include=extra_include or status_cols,
        )

    def tf_and_norm(self, tf_map: dict[Col, Callable] | None = None) -> pd.DataFrame:
        """Apply per-column transformations then z-score numeric columns.

        - Uses the default transforms from :class:`LifeExpectancyColumn` metadata.
        - Allows overrides via ``tf_map`` (set a value to ``None`` to drop a default).
        - Never standardizes the year column or status dummy columns.

        Args:
            tf_map: Optional mapping from ``LifeExpectancyColumn`` to a callable that
                accepts and returns a ``pd.Series``. Custom entries override defaults.

        Returns:
            DataFrame with transforms applied and numeric columns standardized
            (mean=0, std=1) except for ``year`` and status dummy columns.
        """
        df = self.tf_only(tf_map=tf_map)

        status_dummy_cols = [col for col in df.columns if col.startswith("status_")]

        # Recompute numeric columns after transformations (status/year/dummies excluded)
        numeric_cols = (
            df.select_dtypes(include=["number"])
            .columns.difference([self.Col.STATUS, self.Col.YEAR, *status_dummy_cols])
            .tolist()
        )
        if self.Col.TARGET in numeric_cols:
            numeric_cols.remove(self.Col.TARGET)

        if numeric_cols:
            # Median-impute numeric cols to avoid NaNs breaking StandardScaler
            has_missing = df[numeric_cols].isna().any().any()
            if has_missing:
                warn(
                    "tf_and_norm(): median-filled missing numeric values before standardization.",
                    stacklevel=2,
                )
            df.loc[:, numeric_cols] = df.loc[:, numeric_cols].apply(lambda s: s.fillna(s.median()))
            df.loc[:, numeric_cols] = self.standardize(df)[numeric_cols]

        return df

    def tf_only(self, tf_map: dict[Col, Callable] | None = None) -> pd.DataFrame:
        """Apply per-column transformations without standardization.

        This is a "transform-only" counterpart to :meth:`tf_and_norm`. It is
        useful when you need to:

        - fit scaling parameters on a reference split/year and apply them to a
          different dataset, or
        - keep variables on their transformed but original scale (e.g., log1p
          counts) for interpretability.

        Args:
            tf_map: Optional mapping from ``LifeExpectancyColumn`` to a callable that
                accepts and returns a ``pd.Series``. Custom entries override defaults.
                Set a value to ``None`` to drop a default transform.

        Returns:
            DataFrame with transforms applied. No z-scoring is performed.
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

        return df

    @staticmethod
    def country_count(df: pd.DataFrame) -> int:
        if Col.COUNTRY in df.columns:
            return int(df[Col.COUNTRY].nunique())
        if df.index.name == Col.COUNTRY:
            return int(df.index.nunique())
        return int(df.index.nunique())

    @staticmethod
    def year_bounds(df: pd.DataFrame) -> tuple[int | None, int | None]:
        if Col.YEAR not in df.columns:
            return None, None
        s = df[Col.YEAR]
        years = s.dt.year if pd.api.types.is_datetime64_any_dtype(s) else s
        return int(years.min()), int(years.max())

    @staticmethod
    def pct_missing_numeric(df: pd.DataFrame) -> float:
        numeric = df.select_dtypes(include=["number"])
        if numeric.empty:
            return 0.0
        return float(numeric.isna().mean().mean() * 100)
