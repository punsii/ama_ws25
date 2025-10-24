"""Domain-specific lightweight views over dataset content."""

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetView:
    """Immutable snapshot of dataset data and related metadata.

    Attributes:
        data: Materialized dataframe slice containing the requested columns.
        pretty_by_col: Mapping from canonical column names to display-friendly labels.
        numeric_cols: Ordered list of numeric feature names present in ``data``.
        target_col: Optional name of the target variable used for analysis.
    """

    data: pd.DataFrame
    pretty_by_col: Mapping[str, str]
    numeric_cols: list[str]
    target_col: str | None = None

    @property
    def features(self) -> pd.DataFrame:
        """Return view over numeric feature columns."""
        cols = self.numeric_cols or self.data.columns.tolist()
        return self.data.loc[:, cols]
