"""PCA visualization functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from ama_tlbx.analysis.pca_analyzer import PCAAnalyzer


def plot_explained_variance(
    analyzer: PCAAnalyzer,
    figsize: tuple[int, int] = (10, 6),
) -> Figure:
    """Plot explained variance ratio and cumulative variance for PCA components.

    Args:
        analyzer: Fitted PCAAnalyzer instance
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    if analyzer._pca_model is None:
        raise ValueError("PCA model not fitted. Call fit() first.")

    n_components = len(analyzer._pca_model.explained_variance_ratio_)

    fig, ax1 = plt.subplots(figsize=figsize)

    # Bar plot for individual variance
    ax1 = sns.barplot(
        x=np.arange(1, n_components + 1),
        y=analyzer._pca_model.explained_variance_ratio_,
        color="skyblue",
        ax=ax1,
    )
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Line plot for cumulative variance
    ax2 = ax1.twinx()
    ax2 = sns.lineplot(
        x=np.arange(1, n_components + 1),
        y=analyzer._pca_model.explained_variance_ratio_.cumsum(),
        marker="o",
        color="red",
        ax=ax2,
    )
    ax2.set_ylabel("Cumulative Variance Explained", color="red")
    ax2.set_ylim(0, 1)
    ax2.set_yticks(np.arange(0, 1.1, 0.1))
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title("PCA Explained Variance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_loadings_heatmap(
    analyzer: PCAAnalyzer,
    n_components: int = 3,
    top_n_features: int | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> Figure:
    """Plot heatmap of PCA loadings for top features.

    Args:
        analyzer: Fitted PCAAnalyzer instance
        n_components: Number of principal components to show
        top_n_features: Number of top features to display (None = all)
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    if analyzer._pca_model is None:
        raise ValueError("PCA model not fitted. Call fit() first.")

    loadings = analyzer.get_loading_vectors()
    if isinstance(loadings, pd.Series):
        raise TypeError("Expected DataFrame from get_loading_vectors()")

    pc_cols = [f"PC{i + 1}" for i in range(min(n_components, loadings.shape[1]))]

    if top_n_features is not None:
        top_features = list(analyzer.get_top_loading_features(n_components=n_components)[:top_n_features])
        loadings = loadings.loc[top_features]

    fig = plt.figure(figsize=figsize)
    sns.heatmap(
        loadings[pc_cols],
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
    )
    plt.title(f"PCA Loadings for Top {top_n_features or 'All'} Features")
    plt.tight_layout()

    return fig


# def plot_biplot(
#     analyzer: PCAAnalyzer,
#     *,
#     pc1: int = 1,
#     pc2: int = 2,
#     n_loadings: int = 4,
#     figsize: tuple[int, int] = (12, 8),
#     label_points: bool = True,
# ) -> Figure:
#     """Create a PCA biplot showing data points and feature loadings.

#     Args:
#         analyzer: Fitted PCAAnalyzer instance
#         pc1: First principal component to plot (1-indexed)
#         pc2: Second principal component to plot (1-indexed)
#         n_loadings: Number of top loading features to show as arrows
#         figsize: Figure size (width, height)
#         label_points: If True, label each data point

#     Returns:
#         matplotlib Figure object
#     """
#     if analyzer._pca_model is None:
#         raise ValueError("PCA model not fitted. Call fit() first.")

#     # Get PC scores
#     pca_scores = analyzer.transform()
#     pc1_vals = pca_scores[f"PC{pc1}"].to_numpy()
#     pc2_vals = pca_scores[f"PC{pc2}"].to_numpy()

#     # Get top loading features
#     top_features = analyzer.get_top_loading_features(n_components=max(pc1, pc2))[:n_loadings]
#     loadings = analyzer.get_loading_vectors()
#     if isinstance(loadings, pd.Series):
#         raise TypeError("Expected DataFrame from get_loading_vectors()")

#     fig = plt.figure(figsize=figsize)
#     plt.scatter(pc1_vals, pc2_vals, alpha=0.6)

#     # Label data points
#     if label_points:
#         for i, idx in enumerate(pca_scores.index):
#             plt.text(pc1_vals[i], pc2_vals[i], str(idx), fontsize=9, alpha=0.7)

#     # Add loading arrows for top features
#     for feature in top_features:
#         loading_pc1 = loadings.loc[feature, f"PC{pc1}"]
#         loading_pc2 = loadings.loc[feature, f"PC{pc2}"]

#         # Scale arrows for visibility
#         scale = 3.0
#         plt.arrow(
#             0,
#             0,
#             loading_pc1 * scale,
#             loading_pc2 * scale,
#             color="r",
#             alpha=0.5,
#             head_width=0.1,
#             head_length=0.1,
#         )
#         plt.text(
#             loading_pc1 * scale * 1.15,
#             loading_pc2 * scale * 1.15,
#             feature,
#             color="r",
#             fontsize=10,
#             weight="bold",
#         )

#     plt.xlabel(f"PC{pc1} ({analyzer._pca_model.explained_variance_ratio_[pc1 - 1]:.1%} variance)")
#     plt.ylabel(f"PC{pc2} ({analyzer._pca_model.explained_variance_ratio_[pc2 - 1]:.1%} variance)")
#     plt.title(f"PCA Biplot: PC{pc1} vs PC{pc2}")
#     plt.grid(True, alpha=0.3)
#     plt.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.3)
#     plt.axvline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.3)
#     plt.tight_layout()

#     return fig
