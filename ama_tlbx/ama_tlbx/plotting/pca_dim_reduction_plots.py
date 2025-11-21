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


def _wrap_label(label: str, max_len: int) -> str:
    """Insert a line break into long labels to improve readability."""
    if len(label) <= max_len:
        return label

    parts: list[str] = []
    current: list[str] = []
    for word in label.split():
        projected = " ".join([*current, word])
        if len(projected) > max_len and current:
            parts.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        parts.append(" ".join(current))
    return "\n".join(parts)


def plot_group_biplot(
    group_result: GroupPCAResult,
    reduced_df: pd.DataFrame,
    *,
    target: pd.Series | None = None,
    pretty_by_col: Mapping[str, str] | None = None,
    figsize: tuple[int, int] = (9, 7),
    cmap: str = "RdYlGn",
    label_jitter: float = 0.05,
    arrow_scale: float | None = None,
    label_max_len: int = 24,
) -> Figure:
    """Biplot for a single feature group with minimal, readable layout."""

    pc1_col = f"{group_result.group.name}_PC1"
    pc2_col = f"{group_result.group.name}_PC2"
    if pc1_col not in reduced_df.columns:
        raise ValueError(f"{pc1_col} not found in reduced_df columns.")

    has_pc2 = group_result.n_components >= 2 and pc2_col in reduced_df.columns

    x = reduced_df[pc1_col]
    y = reduced_df[pc2_col] if has_pc2 else pd.Series(0.0, index=x.index)

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        x,
        y,
        alpha=0.6,
        s=32,
        c=target if target is not None else "#6baed6",
        cmap=cmap if target is not None else None,
        edgecolors="white",
        linewidth=0.5,
    )
    if target is not None:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Life Expectancy (years)", fontweight="bold")

    # Scale arrows by true PC spread
    base_range = 1.0
    if len(x):
        base_range = max(base_range, float(np.max(np.abs(x))))
    if has_pc2 and len(y):
        base_range = max(base_range, float(np.max(np.abs(y))))
    scale = arrow_scale or max(0.65 * base_range, 1.0)

    loadings = group_result.loadings
    for feature in loadings.index:
        pretty = pretty_by_col.get(feature, feature) if pretty_by_col else feature
        label = _wrap_label(pretty, max_len=label_max_len)

        lx = loadings.loc[feature, "PC1"] * scale
        ly = loadings.loc[feature, "PC2"] * scale if has_pc2 and "PC2" in loadings.columns else 0.0

        ax.arrow(
            0,
            0,
            lx,
            ly,
            head_width=0.08 * scale,
            head_length=0.12 * scale,
            fc="firebrick",
            ec="firebrick",
            linewidth=2.0,
            length_includes_head=True,
            alpha=0.9,
        )

        # Jitter labels only (not points/arrows) to reduce overlap
        jitter_y = np.random.uniform(-label_jitter, label_jitter) * scale
        jitter_x = np.random.uniform(-label_jitter, label_jitter) * scale

        ax.annotate(
            label,
            xy=(lx, ly),
            xytext=(lx + jitter_x, ly + jitter_y),
            fontsize=9,
            fontweight="bold",
            color="firebrick",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="firebrick", alpha=0.85),
            arrowprops=dict(arrowstyle="-", color="firebrick", alpha=0.8),
        )

    margin = scale * (1.4 if has_pc2 else 1.1)
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin if has_pc2 else margin * 0.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.35)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.35)

    ax.set_xlabel(
        f"PC1 ({group_result.explained_variance['explained_ratio'].iloc[0]:.1%} variance)",
        fontweight="bold",
    )
    ax.set_ylabel(
        f"PC2 ({group_result.explained_variance['explained_ratio'].iloc[1]:.1%} variance)"
        if has_pc2
        else "PC2 (not retained)",
        fontweight="bold",
    )
    ax.set_title(
        f"Biplot: {group_result.group.name.replace('_', ' ').title()} (n_components={group_result.n_components})",
        fontweight="bold",
        fontsize=12,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
