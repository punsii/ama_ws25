"""PCA dimensionality reduction visualization functions."""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from ama_tlbx.analysis.pca_dim_reduction import GroupPCAResult, PCADimReductionResult


def plot_group_variance_summary(
    result: PCADimReductionResult,
    figsize: tuple[int, int] = (10, 6),
    annotate: bool = True,
) -> Figure:
    """Plot cumulative explained variance per group.

    Each bar shows the cumulative variance explained by the retained PCs
    for that group. The variance thresholds (min_var_explained) are shown
    as a horizontal reference line when identical, or as annotations when different.

    Args:
        result: PCADimReductionResult from PCADimReductionAnalyzer.
        figsize: Size of the figure (width, height).
        annotate: If True, annotate bars with k (number of PCs).

    Returns:
        Matplotlib Figure.
    """
    if not result.group_results:
        raise ValueError("PCADimReductionResult contains no group_results.")

    data: list[dict[str, object]] = []
    for gr, threshold in zip(result.group_results, result.min_var_explained_per_group, strict=False):
        data.append(
            {
                "group": gr.group.name,
                "cumulative_explained": gr.cumulative_variance_explained,
                "threshold": threshold,
                "n_components": gr.n_components,
            },
        )

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=df,
        x="group",
        y="cumulative_explained",
        ax=ax,
        color="tab:blue",
    )

    ax.set_ylabel("Cumulative Variance Explained")
    ax.set_xlabel("Feature Group")
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([i / 10 for i in range(11)])
    ax.set_title("Variance Retained per Feature Group")

    # Annotate with number of PCs
    if annotate:
        for i, row in df.iterrows():
            ax.text(
                i,
                row["cumulative_explained"] + 0.02,
                f"k={row['n_components']}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Show threshold line if all equal
    unique_thresholds = df["threshold"].unique()
    if len(unique_thresholds) == 1:
        thr = float(unique_thresholds[0])
        ax.axhline(
            thr,
            color="tab:orange",
            linestyle="--",
            linewidth=1.5,
            label=f"min_var_explained={thr:.2f}",
        )
        ax.legend(loc="lower right")

    fig.tight_layout()
    return fig


def plot_group_compression(
    result: PCADimReductionResult,
    figsize: tuple[int, int] = (10, 6),
) -> Figure:
    """Plot original vs reduced dimensionality per feature group.

    For each group, shows:
      - Number of original features in the group
      - Number of retained principal components

    Args:
        result: PCADimReductionResult from PCADimReductionAnalyzer.
        figsize: Size of the figure (width, height).

    Returns:
        Matplotlib Figure.
    """
    if not result.group_results:
        raise ValueError("PCADimReductionResult contains no group_results.")

    rows: list[dict[str, object]] = []
    for gr in result.group_results:
        rows.append(
            {
                "group": gr.group.name,
                "kind": "original_features",
                "count": gr.n_features,
            },
        )
        rows.append(
            {
                "group": gr.group.name,
                "kind": "retained_pcs",
                "count": gr.n_components,
            },
        )

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=df,
        x="group",
        y="count",
        hue="kind",
        ax=ax,
    )

    ax.set_ylabel("Count")
    ax.set_xlabel("Feature Group")
    ax.set_title("Original vs Reduced Dimensionality")
    ax.legend(
        title="Type",
        labels=["Original features", "Retained PCs"],
        loc="upper right",
    )
    fig.tight_layout()

    return fig


def plot_group_loadings(
    result: PCADimReductionResult,
    figsize: tuple[int, int] = (12, 4),
) -> Figure:
    """Plot feature loadings on PC1 for all groups.

    Shows how original features contribute to the first principal component
    in each group.

    Args:
        result: PCADimReductionResult from PCADimReductionAnalyzer.
        figsize: Size of the figure per subplot row (width, height per row).

    Returns:
        Matplotlib Figure.
    """
    if not result.group_results:
        raise ValueError("PCADimReductionResult contains no group_results.")

    n_groups = len(result.group_results)
    fig, axes = plt.subplots(1, n_groups, figsize=(figsize[0], figsize[1]))

    if n_groups == 1:
        axes = [axes]

    for idx, gr in enumerate(result.group_results):
        ax = axes[idx]

        # Get ALL loadings from explained_variance (not just retained ones)
        # We need to reconstruct full loadings from the PCA result
        # The gr.loadings only contains retained PCs, but we want to show all
        all_loadings = gr.loadings.copy()

        # Use pretty names from result for row index (features)
        all_loadings.index = [result.pretty_by_col.get(f, f) for f in all_loadings.index]

        # Create custom y-axis labels to mark retained PCs
        n_retained = gr.n_components
        n_total = len(all_loadings.columns)
        y_labels = [f"{col} (y)" if i < n_retained else f"{col} (n)" for i, col in enumerate(all_loadings.columns)]

        # Heatmap (transpose so PCs are rows, features are columns)
        sns.heatmap(
            all_loadings.T,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0,
            cbar_kws={"label": "Loading"},
            ax=ax,
            vmin=-1,
            vmax=1,
            yticklabels=y_labels,
        )

        # Highlight retained PCs with background color
        for i in range(n_retained, n_total):
            ax.axhspan(i, i + 1, facecolor="lightgray", alpha=0.2, zorder=0)

        ax.set_title(
            f"{gr.group.name}\n({gr.cumulative_variance_explained:.1%} var., {n_retained}/{n_total} PCs)",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_ylabel("Principal Component", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=9)

    fig.suptitle("Feature Loadings on PC1 by Group", y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout()

    return fig
