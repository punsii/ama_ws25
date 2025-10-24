"""Base dataset class for all dataset implementations."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.preprocessing import StandardScaler

from ama_tlbx.analysis.correlation_analyzer import CorrelationAnalyzer
from ama_tlbx.analysis.pca_analyzer import PCAAnalyzer


if TYPE_CHECKING:
    from ama_tlbx.analysis.outlier_detector import OutlierDetector

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
            self._df_standardized = self._compute_standardized()
        return self._df_standardized

    def _compute_standardized(self) -> pd.DataFrame:
        """Compute standardized version of the dataset using [sklearn's StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

        Default implementation uses StandardScaler on numeric columns.
        Subclasses can override for custom behavior.

        Returns:
            Standardized DataFrame with numeric columns scaled to mean=0, std=1
        """
        self._scaler = StandardScaler()
        scaled_data = self._scaler.fit_transform(self.df[self.numeric_cols])

        return pd.DataFrame(
            scaled_data,
            columns=self.numeric_cols,
            index=self.df.index,
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
        *,
        columns: Iterable[str] | None = None,
        standardized: bool = False,
        target_col: str | None = None,
    ) -> DatasetView:
        """Build an immutable dataset view for analyzers and plotting layers.

        Args:
            columns: Columns to include in the view (defaults to all)
            standardized: Use standardized dataframe if available
            target_col: Optional target column reference

        Returns:
            DatasetView containing selected data and metadata
        """
        frame = self.df_standardized if standardized else self.df
        selected_cols = list(columns or frame.columns.to_list())
        data = frame.loc[:, selected_cols].copy()
        pretty_by_col = {col: self.get_pretty_name(col) for col in selected_cols}
        numeric_cols = [col for col in selected_cols if col in self.numeric_cols]

        return DatasetView(
            data=data,
            pretty_by_col=pretty_by_col,
            numeric_cols=numeric_cols,
            target_col=target_col,
        )

    def feature_columns(
        self,
        *,
        include_target: bool = False,
        extra_exclude: Iterable[str] | None = None,
    ) -> list[str]:
        """Return numeric feature columns, optionally excluding identifiers and target."""
        exclude = set(self.identifier_columns)
        if extra_exclude:
            exclude.update(extra_exclude)
        if not include_target and self.Col.TARGET:
            exclude.add(self.Col.TARGET)
        return [col for col in self.numeric_cols if col not in exclude]

    def analyzer_view(
        self,
        *,
        columns: Iterable[str] | None = None,
        standardized: bool = True,
        include_target: bool = True,
    ) -> DatasetView:
        """Build a dataset view tailored for downstream analyzers."""
        if columns is None:
            columns = self.feature_columns(include_target=include_target)
        return self.view(
            columns=columns,
            standardized=standardized,
            target_col=self.Col.TARGET if include_target else None,
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
        *,
        columns: Iterable[str] | None = None,
        standardized: bool = True,
        include_target: bool = True,
    ) -> "CorrelationAnalyzer":
        """Instantiate a correlation analyzer configured for this dataset."""
        return CorrelationAnalyzer(
            self.analyzer_view(columns=columns, standardized=standardized, include_target=include_target),
        )

    def make_pca_analyzer(
        self,
        *,
        columns: Iterable[str] | None = None,
        standardized: bool = True,
        exclude_target: bool = True,
    ) -> "PCAAnalyzer":
        """Instantiate a PCA analyzer configured for this dataset."""
        return PCAAnalyzer(
            self.analyzer_view(
                columns=columns,
                standardized=standardized,
                include_target=not exclude_target,
            ),
        )

    def detect_outliers(
        self,
        detector: "OutlierDetector",
        *,
        columns: Iterable[str] | None = None,
        standardized: bool = True,
    ) -> pd.DataFrame:
        """Detect outliers with the provided detector for the chosen columns."""
        columns = list(columns or self.feature_columns(include_target=False))
        return detector.detect(data=self.view(columns=columns, standardized=standardized).data, columns=columns)
