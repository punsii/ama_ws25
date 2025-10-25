"""Dataset visualization functions."""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from ama_tlbx.data.base_dataset import BaseDataset


def plot_standardization_comparison(
    dataset: BaseDataset,
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
