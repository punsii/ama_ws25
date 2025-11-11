"""Visualization helpers for the Jens notebooks.

Provides DataVisualiser with a convenience method to show a correlation
matrix for a pandas DataFrame.
"""
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns


class DataVisualiser:
    """Small collection of plotting helpers used in the notebooks/jens folder.

    Methods are static so they can be used without instantiating the class
    inside quick notebook cells.
    """

    @staticmethod
    def showCorrelationMatrix(df: pd.DataFrame, *, annot: bool = True, cmap: str = "coolwarm", figsize: Tuple[int, int] = (10, 8)) -> plt.Axes:
        """Display a correlation matrix heatmap for the given DataFrame.

        Args:
            df: Input data (will be converted to pandas.DataFrame if possible).
            annot: Whether to annotate each cell with the correlation value.
            cmap: Colormap for the heatmap.
            figsize: Figure size (width, height).

        Returns:
            The matplotlib Axes containing the heatmap.
        """
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        # Keep only numeric columns — seaborn / pandas correlation expects numeric data.
        num_df = df.select_dtypes(include=["number"]).copy()
        # drop columns that are entirely NA (they produce NaNs in corr)
        num_df = num_df.dropna(axis=1, how="all")
        if num_df.shape[1] == 0:
            raise ValueError("No numeric columns available to compute correlation. Ensure the dataframe contains numeric data.")

        corr = num_df.corr()

        plt.figure(figsize=figsize)
        ax = sns.heatmap(corr, annot=annot, cmap=cmap, fmt=".2f", vmin=-1, vmax=1, square=True)

        plt.title("Correlation matrix")
        plt.tight_layout()
        plt.show()
        return ax
