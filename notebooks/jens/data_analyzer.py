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
from sklearn.preprocessing import StandardScaler


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
        
        num_df = num_df.dropna(axis=1, how="all")
        if num_df.shape[1] == 0:
            raise ValueError("No numeric columns available to combine after filtering; nothing to do.")
        
        if num_df.shape[1] == 1:
            print(f"Only one column '{num_df.columns[0]}' selected for concatenation. No PCA performed.")
            if drop_original:
                result = df.drop(columns=num_df.columns[0])
            else:
                result = df.copy()
            result[new_column_name] = df[num_df.columns[0]] # Take from original df to handle potential NA in num_df
            # For the Quarto section, you'd state: "Only one column was available, so it was directly used as the new feature, explaining 100% of its own variance."
            return result

        # 1. Apply NaN handling (using fillna(0) as per your original logic)
        filled_num_df = num_df.fillna(0) 
        
        # 2. Standardize the data before PCA
        scaler = StandardScaler()
        scaled_num_df = pd.DataFrame(scaler.fit_transform(filled_num_df), 
                                     columns=filled_num_df.columns, 
                                     index=filled_num_df.index)

        pca = PCA(n_components=1)
        pc = pca.fit(scaled_num_df) 
        
        explained_variance_ratio_pc1 = pc.explained_variance_ratio_[0] * 100 # Convert to percentage
        print(f"PCA performed on {len(selected_columns)} columns (after NA filtering).")
        print(f"The first principal component (PC1) explains {explained_variance_ratio_pc1:.2f}% of the variance within these columns.")

        loadings = pd.DataFrame(pc.components_.T, index=num_df.columns, columns=["loading"]).reset_index().rename(columns={"index": "feature"})
        loadings["loading"] **= 2
        print("PCA loadings, weighted (squared for component contribution):")
        print(loadings)
        
        weights = loadings.set_index("feature")["loading"]
        available = [f for f in weights.index if f in num_df.columns]
        if len(available) == 0:
            raise ValueError("None of the PCA features are present in the DataFrame columns")
        selected = num_df[available].fillna(0) # Using fillna(0) for selected here too for consistency
        weighted_series = selected.mul(weights.loc[available], axis=1).sum(axis=1) # This is your new column

        if drop_original:
            result = df.drop(columns=available)
        else:
            result = df.copy()
        result[new_column_name] = weighted_series
        return result


__all__ = ["DataAnalyzer"]
