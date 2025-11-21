"""Task-specific views over dataset content."""

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetView:
    """Immutable snapshot of dataset data and related metadata.

    Attributes:
        df: Dataframe slice containing the relevant columns.
        pretty_by_col: Mapping from normalized column names to display-friendly labels.
        numeric_cols: Ordered list of numeric feature names present in ``data``.
        target_col: Optional name of the target variable used for analysis.
        is_standardized: Indicates if numeric features have been standardized (zero mean, unit variance).
    """

    df: pd.DataFrame
    """Dataframe slice containing the relevant columns."""
    pretty_by_col: Mapping[str, str]
    """Mapping from normalized column names to display-friendly labels."""
    numeric_cols: list[str]
    target_col: str | None = None
    is_standardized: bool | None = None
    """Indicates if numeric features have been standardized (zero mean, unit variance)."""

    @property
    def features(self) -> pd.DataFrame:
        """Return view over numeric feature columns."""
        cols = self.numeric_cols or self.df.columns.tolist()
        return self.df.loc[:, cols]
