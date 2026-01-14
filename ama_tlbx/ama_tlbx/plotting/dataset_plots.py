"""Dataset visualization functions.

These helpers are used for quick EDA in notebooks and the Quarto submission.
They are intentionally lightweight and return matplotlib/seaborn objects for
easy composition.
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


if TYPE_CHECKING:
    from ama_tlbx.data.base_dataset import BaseDataset


def plot_histograms(
    df: pd.DataFrame,
    columns: list[str],
    *,
    bins: int = 30,
    figsize: tuple[int, int] = (16, 10),
    kde: bool = False,
) -> Figure:
    """Plot histograms for a set of columns.

    Args:
        df: Source DataFrame.
        columns: Columns to plot.
        bins: Number of histogram bins.
        figsize: Size of the created figure.
        kde: Whether to overlay a kernel density estimate.

    Returns:
        Matplotlib figure containing the histogram grid.
    """
    n_cols = min(3, len(columns))
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes_arr, columns, strict=False):
        sns.histplot(data=df, x=col, bins=bins, kde=kde, ax=ax)
        ax.set_title(f"Histogram: {col}")

    for ax in axes_arr[len(columns) :]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_standardization_comparison(
    dataset: "BaseDataset",
    figsize: tuple[int, int] = (20, 10),
) -> Figure:
    """Plot boxplots comparing non-standardized vs standardized data.

    Args:
        dataset: Dataset instance with data to visualize
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    numeric_cols = dataset.numeric_cols

    fig, axs = plt.subplots(2, 1, figsize=figsize)

    sns.boxplot(data=dataset.df[numeric_cols], ax=axs[0])
    axs[0].tick_params(axis="x", rotation=45)
    axs[0].set_title("Non-Standardized")

    sns.boxplot(data=dataset.df_standardized[numeric_cols], ax=axs[1])
    axs[1].tick_params(axis="x", rotation=45)
    axs[1].set_title("Standardized")

    plt.tight_layout()

    return fig
