"""Correlation analysis visualization functions."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from ama_tlbx.analysis.correlation_analyzer import CorrelationAnalyzer
from ama_tlbx.data_handling.base_dataset import BaseDataset


# TODO: No plotting function is allowed to do any computation, they should take precomputed data from the respective analyzer!
def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    pretty_labels: list[str],
    figsize: tuple[int, int] = (20, 20),
    **kwargs: object,
) -> Figure:
    """Plot correlation heatmap of all features.

    Args:
        corr_df: Correlation df from CorrelationAnalyzer.get_correlation_matrix()
        pretty_labels: Pretty names for features from dataset
        figsize: Figure size (width, height)
        **kwargs: Additional arguments passed to seaborn.heatmap

    Returns:
        matplotlib Figure object
    """
    # TODO: id doens't use the pretty_labels!
    fig = plt.figure(figsize=figsize)

    # Use pretty names for axis labels
    corr_renamed = corr_df.copy()
    corr_renamed.columns = pretty_labels
    corr_renamed.index = pretty_labels

    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        **kwargs,  # type: ignore[arg-type]
    )

    plt.tick_params(axis="both", rotation=45)
    plt.title("Country-Level Feature Correlations")
    plt.tight_layout()

    return fig


def plot_top_correlated_pairs(
    analyzer: CorrelationAnalyzer,
    n: int = 20,
    figsize: tuple[int, int] = (12, 10),
) -> tuple[Figure, Figure]:
    """Plot top positively and negatively correlated feature pairs.

    Args:
        analyzer: CorrelationAnalyzer instance
        n: Number of pairs to show
        figsize: Figure size (width, height)

    Returns:
        Tuple of (positive_correlations_fig, negative_correlations_fig)
    """
    corr_pairs_high = analyzer.get_top_correlated_pairs(n=n, ascending=False)
    corr_pairs_low = analyzer.get_top_correlated_pairs(n=n, ascending=True)

    # Top positive correlations
    fig1 = plt.figure(figsize=figsize)
    sns.barplot(
        data=corr_pairs_high,
        x="correlation",
        y="pair",
        hue="correlation",
        palette="coolwarm",
        legend=False,
    )
    plt.title(f"Top {n} Positively Correlated Feature Pairs")
    plt.xlabel("Pearson Correlation")
    plt.axvline(0, color="black", linewidth=1, linestyle="--")
    plt.tight_layout()

    # Top negative correlations
    fig2 = plt.figure(figsize=figsize)
    sns.barplot(
        data=corr_pairs_low,
        x="correlation",
        y="pair",
        hue="correlation",
        palette="coolwarm",
        legend=False,
    )
    plt.title(f"Top {n} Negatively Correlated Feature Pairs")
    plt.xlabel("Pearson Correlation")
    plt.axvline(0, color="black", linewidth=1, linestyle="--")
    plt.tight_layout()

    return fig1, fig2


def plot_target_correlations(
    analyzer: CorrelationAnalyzer,
    dataset: BaseDataset,
    target_col: str,
    n: int = 10,
    figsize: tuple[int, int] = (10, 6),
) -> tuple[Figure, Figure]:
    """Plot top positive and negative correlations with target variable.

    Args:
        analyzer: CorrelationAnalyzer instance
        dataset: Dataset instance for getting pretty names
        target_col: Name of target variable column
        n: Number of features to show
        figsize: Figure size (width, height)

    Returns:
        Tuple of (positive_correlations_fig, negative_correlations_fig)
    """
    target_corr = analyzer.get_target_correlations(target_col)

    # Top positive correlations
    fig1 = plt.figure(figsize=figsize)
    sns.barplot(
        data=target_corr.head(n),
        x="correlation",
        y="feature",
        palette="coolwarm",
    )
    plt.title(f"Top Positive Correlations with {dataset.get_pretty_name(target_col)}")
    plt.xlabel("Pearson Correlation")
    plt.axvline(0, color="black", linewidth=1, linestyle="--")
    plt.tight_layout()

    # Top negative correlations
    fig2 = plt.figure(figsize=figsize)
    sns.barplot(
        data=target_corr.tail(n).sort_values("correlation"),
        x="correlation",
        y="feature",
        palette="coolwarm",
    )
    plt.title(f"Top Negative Correlations with {dataset.get_pretty_name(target_col)}")
    plt.xlabel("Pearson Correlation")
    plt.axvline(0, color="black", linewidth=1, linestyle="--")
    plt.tight_layout()

    return fig1, fig2
