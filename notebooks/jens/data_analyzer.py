"""Small, notebook-friendly data analysis helpers.

Provides a lightweight `DataAnalyzer` class with a PCA-based
`concatColumns` helper to combine multiple numeric columns into a single
representative column (weights are derived from PCA loadings squared).

This is a simplified, dependency-light extraction adapted from
`ama_tlbx.analysis.column_concat.ColumnConcatenator` for use in notebooks.
"""
from typing import Iterable, Optional

import pandas as pd
from sklearn.decomposition import PCA


class DataAnalyzer:
    """Notebook helper providing analysis utilities.

    Methods are static so they can be used quickly from notebook cells.
    """

    @staticmethod
    def concatColumns(
        df: pd.DataFrame,
        columns: Optional[Iterable[str]] = None,
        new_column_name: str = "concatenated",
        drop_original: bool = True,
    ) -> pd.DataFrame:
        
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        # Choose columns: provided or all numeric
        if columns is None:
            num_df = df.select_dtypes(include=["number"]).copy()
            selected_columns = list(num_df.columns)
        else:
            selected_columns = [c for c in columns if c in df.columns]
            if len(selected_columns) == 0:
                raise ValueError("None of the specified columns are present in the DataFrame")
            num_df = df[selected_columns].copy()

        # Drop columns that are entirely NA (they don't contribute)
        num_df = num_df.dropna(axis=1, how="all")
        if num_df.shape[1] == 0:
            raise ValueError("No numeric columns available to combine after filtering; nothing to do.")

        # Fit PCA to reduce to one component
        pca = PCA(n_components=1)
        pc = pca.fit(num_df.fillna(0))

        # Build loadings: (n_features, 1) -> DataFrame with 'feature' and 'loading'
        loadings = pd.DataFrame(pc.components_.T, index=num_df.columns, columns=["loading"]).reset_index().rename(columns={"index": "feature"})

        # Square loadings to get positive importance weights
        loadings["loading"] **= 2

        print("PCA loadings, weighted")
        print(loadings)

        # Construct weighted sum using the squared loadings
        weights = loadings.set_index("feature")["loading"]

        available = [f for f in weights.index if f in num_df.columns]
        if len(available) == 0:
            raise ValueError("None of the PCA features are present in the DataFrame columns")

        selected = num_df[available].fillna(0)
        weighted_series = selected.mul(weights.loc[available], axis=1).sum(axis=1)

        # Compose resulting DataFrame
        if drop_original:
            result = df.drop(columns=available)
        else:
            result = df.copy()

        result[new_column_name] = weighted_series
        return result


__all__ = ["DataAnalyzer"]
