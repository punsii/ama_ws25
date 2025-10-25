"""Column Concatenation."""

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sklearn.decomposition import PCA

from ama_tlbx.data.views import DatasetView

@dataclass
class ColumnConcatenator:
    """Utility for concatenating multiple columns into a single column."""

    view: DatasetView

    def __init__(self, view: DatasetView):
        """Initialize the Column Concatenator."""
        self.view = view

    def concatenate(
        self,
        columns: list[str],
        new_column_name: str,
    ) -> pd.DataFrame:
        # copy the underlying DataFrame from the provided view and return after
        # concatenation. Use `self.view` (the dataclass field) rather than
        # an underscore-prefixed attribute which doesn't exist.
        df = self.view.df.copy()

        # Separate the columns to be concatenated and those to remain
        only_columns = df[columns]
        only_not_columns = df.drop(columns=columns)

        # Perform PCA to reduce to a single column
        pca = PCA(n_components=1)
        pc = pca.fit(only_columns)

        # build a tidy loadings DataFrame with one row per feature and a
        # single numeric 'loading' column (shape: n_features x 1), then
        # square the loading values as requested.
        loadings = (
            pd.DataFrame(pc.components_.T, index=only_columns.columns, columns=["loading"])  # (n_features, 1)
            .reset_index()
            .rename(columns={"index": "feature"})
        )

        # raise loadings to the power of 2 to get normalised importance
        loadings["loading"] = loadings["loading"] ** 2

        # Construct a weighted sum using the squared loadings.
        # - `loadings` has columns ['feature', 'loading']
        weights = loadings.set_index("feature")["loading"]

        
        available = [f for f in weights.index if f in only_columns.columns]
        if len(available) == 0:
            raise ValueError("None of the PCA features are present in the DataFrame columns")

        # Select the columns in the same order as weights, fill missing values
        # with 0 so they don't contribute to the weighted sum.
        selected = only_columns[available].fillna(0)

        # Perform weighted sum
        weighted_series = selected.mul(weights.loc[available], axis=1).sum(axis=1)

        only_not_columns[new_column_name] = weighted_series
        return only_not_columns