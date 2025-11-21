"""Base dataset class for all dataset implementations."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


if TYPE_CHECKING:
    from ama_tlbx.analysis.correlation_analyzer import CorrelationAnalyzer
    from ama_tlbx.analysis.outlier_detector import (
        IQROutlierDetector,
        IsolationForestOutlierDetector,
        ZScoreOutlierDetector,
    )
    from ama_tlbx.analysis.pca_analyzer import PCAAnalyzer
    from ama_tlbx.analysis.pca_dim_reduction import FeatureGroup, PCADimReductionAnalyzer

from .base_columns import BaseColumn
from .views import DatasetView


class BaseDataset(ABC):
    """Abstract base class for dataset handlers used throughout the AMA module."""

    identifier_columns: Sequence[str] = ()
    Col: type[BaseColumn]

    def __init__(self, df: pd.DataFrame | None = None) -> None:
        """Initialize the base dataset.

        Args:
            df: Pre-loaded and cleaned DataFrame (optional)
        """
        self._df: pd.DataFrame | None = df
        self._scaler: StandardScaler | None = None
        self._df_standardized: pd.DataFrame | None = None

    @classmethod
    @abstractmethod
    def from_csv(cls, filepath: str | Path, **kwargs: object) -> "BaseDataset":
        """Load dataset from CSV file.

        Args:
            filepath: Path to the CSV file
            **kwargs: Additional loading parameters

        Returns:
            Dataset instance with loaded data
        """
        ...

    @property
    def df(self) -> pd.DataFrame:
        """Get the raw/cleaned DataFrame.

        Returns:
            The raw DataFrame

        Raises:
            ValueError: If dataset not loaded
        """
        if self._df is None:
            raise ValueError("Dataset not loaded. Use from_csv() to load data.")
        return self._df

    @property
    def df_pretty(self) -> pd.DataFrame:
        """Get the DataFrame with pretty column names.

        Returns:
            DataFrame with pretty column names
        """
        return self.df.rename(columns={col: self.get_pretty_name(col) for col in self.df.columns})

    @property
    def numeric_cols(self) -> pd.Index:
        """Get numeric column names.

        Default implementation filters columns by numeric dtypes.
        Subclasses can override for custom behavior.

        Returns:
            Index of numeric column names
        """
        return self.df.select_dtypes(include=["number"]).columns

    @property
    def df_standardized(self) -> pd.DataFrame:
        """Get the standardized DataFrame.

        X <- (X - E[X]) / Var(X)

        Returns:
            Standardized DataFrame
        """
        if self._df_standardized is None:
            self._df_standardized = self.standardize()
        return self._df_standardized

    def standardize(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Compute standardized version of the dataset using [sklearn's StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

        Default implementation uses StandardScaler on numeric columns.
        Subclasses can override for custom behavior.

        Returns:
            Standardized DataFrame with numeric columns scaled to mean=0, std=1
        """
        if df is None:
            df = self.df

        self._scaler = StandardScaler()
        scaled_data = self._scaler.fit_transform(df[self.numeric_cols])

        return pd.DataFrame(
            scaled_data,
            columns=self.numeric_cols,
            index=df.index,
        )

    def get_pretty_names(self, column_names: list[str] | None = None) -> list[str]:
        """Convert multiple column names to pretty names.

        Args:
            column_names: List of cleaned column names

        Returns:
            List of pretty names suitable for plot labels
        """
        return [self.get_pretty_name(name) for name in column_names or self.df.columns.to_list()]

    def view(
        self,
        columns: Iterable[str] | None = None,
        standardized: bool = False,
        target_col: str | None = None,
        missing_strategy: Literal["drop", "global_impute"] = "drop",
    ) -> DatasetView:
        """Build an immutable dataset view for analyzers and plotting layers.

        Args:
            columns: Columns to include in the view (defaults to all)
            standardized: Use standardized dataframe if available
            target_col: Optional target column reference
            missing_strategy: Strategy to handle missing values ("drop" or "global_impute")

        Returns:
            DatasetView containing selected data and metadata
        """
        frame = self.df_standardized if standardized else self.df

        if missing_strategy == "global_impute":
            # Impute numeric columns globally (column-wise median)
            numeric_cols = filter(lambda c: c in frame.columns, numeric_cols)
            if numeric_cols:
                frame[numeric_cols] = SimpleImputer(strategy="median").fit_transform(frame[numeric_cols])
        elif missing_strategy == "drop":
            # Drop rows after selecting the relevant columns
            pass
        else:
            raise ValueError(
                f"Invalid missing_strategy='{missing_strategy}'. Use 'drop' or 'global_impute'.",
            )

        selected_cols = list(columns or frame.columns.to_list())
        frame = frame.loc[:, selected_cols]

        if missing_strategy == "drop":
            # Drop rows with any missing values in the selected columns
            frame = frame.dropna(axis=0, how="any")

        return DatasetView(
            df=frame,
            pretty_by_col={col: self.get_pretty_name(col) for col in selected_cols},
            numeric_cols=[col for col in selected_cols if col in self.numeric_cols],
            target_col=target_col or self.Col.TARGET,
            is_standardized=standardized,
        )

    def feature_columns(
        self,
        include_target: bool = False,
        extra_exclude: Iterable[str] | None = None,
    ) -> list[str]:
        """Return numeric feature columns, optionally excluding identifiers and target."""
        exclude = set(self.Col.identifier_columns())
        if extra_exclude:
            exclude.update(extra_exclude)
        if not include_target and self.Col.TARGET:
            exclude.add(self.Col.TARGET)
        return [col for col in self.numeric_cols if col not in exclude]

    def analyzer_view(
        self,
        columns: Iterable[str] | None = None,
        standardized: bool = True,
        include_target: bool = True,
        missing_strategy: Literal["drop", "global_impute"] = "drop",
    ) -> DatasetView:
        """Build a dataset view tailored for downstream analyzers."""
        return self.view(
            columns=columns if columns is not None else self.feature_columns(include_target=include_target),
            standardized=standardized,
            target_col=self.Col.TARGET if include_target else None,
            missing_strategy=missing_strategy,
        )

    def get_pretty_name(self, column_name: str) -> str:
        """Convert column name to pretty name for visualization.

        Args:
            column_name: The cleaned column name

        Returns:
            Pretty name suitable for plot labels and titles
        """
        try:
            col_enum = self.Col(column_name)
        except ValueError:
            # Fallback: capitalize and replace underscores if not in enum
            return column_name.replace("_", " ").title()
        else:
            return str(col_enum.pretty_name)

    def make_correlation_analyzer(
        self,
        columns: Iterable[str] | None = None,
        standardized: bool = False,
        include_target: bool = True,
    ) -> "CorrelationAnalyzer":
        """Instantiate a correlation analyzer configured for this dataset."""
        from ama_tlbx.analysis.correlation_analyzer import CorrelationAnalyzer

        return CorrelationAnalyzer(
            self.analyzer_view(columns=columns, standardized=standardized, include_target=include_target),
        )

    def make_pca_analyzer(
        self,
        columns: Iterable[str] | None = None,
        standardized: bool = True,
        exclude_target: bool = True,
    ) -> "PCAAnalyzer":
        """Instantiate a PCA analyzer configured for this dataset."""
        from ama_tlbx.analysis.pca_analyzer import PCAAnalyzer

        return PCAAnalyzer(
            self.analyzer_view(
                columns=columns,
                standardized=standardized,
                include_target=not exclude_target,
            ),
        )

    def make_pca_dim_reduction_analyzer(
        self,
        feature_groups: list["FeatureGroup"],
        columns: Iterable[str] | None = None,
        standardized: bool = True,
        min_var_explained: float | list[float] = 0.7,
        missing_strategy: Literal["drop", "global_impute"] = "drop",
    ) -> "PCADimReductionAnalyzer":
        """Instantiate a PCA dimensionality reduction analyzer for grouped features.

        This analyzer automatically determines the number of principal components
        needed to explain a specified proportion of variance in each feature group.

        Args:
            feature_groups: List of FeatureGroup objects defining correlated feature sets
            columns: Optional column subset (groups must be within these columns)
            standardized: Use standardized data (recommended: True)
            min_var_explained: Minimum proportion of variance to explain per group.
                The analyzer will keep the minimum number of PCs needed to reach
                this threshold.
                - float: Same threshold for all groups (default: 0.7)
                - list[float]: Specific threshold for each group (must match length)
            missing_strategy: Strategy to handle missing values ("drop" or "global_impute")

        Returns:
            PCADimReductionAnalyzer instance

        Example:
            >>> from ama_tlbx.analysis.pca_dim_reduction import FeatureGroup
            >>> groups = [
            ...     FeatureGroup("Immunization", ["hepatitis_b", "polio", "diphtheria"]),
            ...     FeatureGroup("Mortality", ["infant_deaths", "under_five_deaths"]),
            ... ]
            >>> # Keep PCs explaining ≥80% variance (default)
            >>> analyzer = dataset.make_pca_dim_reduction_analyzer(groups)
            >>> result = analyzer.fit().result()
            >>> print(f"Group 1 kept {result.group_results[0].n_components} PCs")
            >>>
            >>> # Keep PCs explaining ≥95% variance for all groups
            >>> analyzer = dataset.make_pca_dim_reduction_analyzer(groups, min_var_explained=0.95)
            >>> result = analyzer.fit().result()
            >>>
            >>> # Different thresholds per group (90% for first, 70% for second)
            >>> analyzer = dataset.make_pca_dim_reduction_analyzer(groups, min_var_explained=[0.9, 0.7])
            >>> result = analyzer.fit().result()
        """
        from ama_tlbx.analysis.pca_dim_reduction import PCADimReductionAnalyzer

        view = self.analyzer_view(
            columns=columns,
            standardized=standardized,
            include_target=False,
            missing_strategy=missing_strategy,
        )
        return PCADimReductionAnalyzer(view=view, feature_groups=feature_groups, min_var_explained=min_var_explained)

    def make_iqr_outlier_detector(
        self,
        columns: Iterable[str] | None = None,
        standardized: bool = True,
        threshold: float = 1.5,
    ) -> "IQROutlierDetector":
        """Instantiate an IQR outlier detector configured for this dataset.

        Example:
            >>> from ama_tlbx.data.life_expectancy_dataset import LifeExpectancyDataset
            >>> ds = LifeExpectancyDataset.from_csv()
            >>> detector = ds.make_iqr_outlier_detector(threshold=1.5)
            >>> outlier_result = detector.fit().result()
            >>> mask = outlier_result.outlier_mask

        Args:
            columns: Columns to analyze (defaults to all numeric features)
            standardized: Use standardized data
            threshold: IQR multiplier for fence calculation (default: 1.5)

        Returns:
            IQROutlierDetector instance
        """
        from ama_tlbx.analysis.outlier_detector import IQROutlierDetector

        columns = list(columns or self.feature_columns(include_target=False))
        return IQROutlierDetector(
            view=self.view(columns=columns, standardized=standardized),
            threshold=threshold,
        )

    def make_zscore_outlier_detector(
        self,
        columns: Iterable[str] | None = None,
        standardized: bool = True,
        threshold: float = 3.0,
    ) -> "ZScoreOutlierDetector":
        """Instantiate a Z-score outlier detector configured for this dataset.

        Args:
            columns: Columns to analyze (defaults to all numeric features)
            standardized: Use standardized data
            threshold: Z-score threshold for outlier detection (default: 3.0)

        Returns:
            ZScoreOutlierDetector instance
        """
        from ama_tlbx.analysis.outlier_detector import ZScoreOutlierDetector

        columns = list(columns or self.feature_columns(include_target=False))
        return ZScoreOutlierDetector(
            view=self.view(columns=columns, standardized=standardized),
            threshold=threshold,
        )

    def make_isolation_forest_outlier_detector(
        self,
        columns: Iterable[str] | None = None,
        standardized: bool = True,
        contamination: float | str = "auto",
        random_state: int | None = None,
        n_estimators: int = 100,
    ) -> "IsolationForestOutlierDetector":
        """Instantiate an Isolation Forest outlier detector configured for this dataset.

        Args:
            columns: Columns to analyze (defaults to all numeric features)
            standardized: Use standardized data
            contamination: Expected proportion of outliers (default: "auto")
            random_state: Random seed for reproducibility (default: None)
            n_estimators: Number of trees in the forest (default: 100)

        Returns:
            IsolationForestOutlierDetector instance
        """
        from ama_tlbx.analysis.outlier_detector import IsolationForestOutlierDetector

        columns = list(columns or self.feature_columns(include_target=False))
        return IsolationForestOutlierDetector(
            view=self.view(columns=columns, standardized=standardized),
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
        )
