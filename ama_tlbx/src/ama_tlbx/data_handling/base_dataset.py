"""Base dataset class for all dataset implementations."""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


class BaseDataset(ABC):
    """Abstract base class for dataset handlers used throughout the AMA module."""

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

    @abstractmethod
    def get_pretty_name(self, column_name: str) -> str:
        """Convert column name to pretty name for visualization.

        Args:
            column_name: The cleaned column name

        Returns:
            Pretty name suitable for plot labels and titles
        """
        ...

    def get_pretty_names(self, column_names: list[str] | None = None) -> list[str]:
        """Convert multiple column names to pretty names.

        Args:
            column_names: List of cleaned column names

        Returns:
            List of pretty names suitable for plot labels
        """
        return [self.get_pretty_name(name) for name in column_names or self.df.columns.to_list()]
