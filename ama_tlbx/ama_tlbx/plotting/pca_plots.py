"""PCA visualization functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.figure import Figure

from ama_tlbx.analysis.pca_analyzer import PCAResult
from ama_tlbx.analysis.pca_dim_reduction import GroupPCAResult, PCADimReductionResult


def plot_explained_variance(
    result: PCAResult,
    figsize: tuple[int, int] = (10, 6),
) -> Figure:
    """Plot explained variance ratio and cumulative variance for PCA components."""
    fig, ax1 = plt.subplots(figsize=figsize)

    variance_df = result.explained_variance
    component_index = np.arange(1, len(variance_df) + 1)

    sns.barplot(
        x=component_index,
        y=variance_df["explained_ratio"],
        color="skyblue",
        ax=ax1,
    )
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    sns.lineplot(
        x=component_index,
        y=variance_df["cumulative_ratio"],
        marker="o",
        color="red",
        ax=ax2,
    )
    ax2.set_ylabel("Cumulative Variance Explained", color="red")
    ax2.set_ylim(0, 1)
    ax2.set_yticks(np.arange(0, 1.1, 0.1))
    ax2.tick_params(axis="y", labelcolor="red")

    ax1.set_xticks(component_index)
    ax1.set_xticklabels(variance_df["PC"])

    ax1.set_title("PCA Explained Variance")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_loadings_heatmap(
    result: PCAResult,
    n_components: int = 3,
    top_n_features: int | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> Figure:
    """Plot heatmap of PCA loadings for top features."""
    loadings = result.loadings
    if isinstance(loadings, pd.Series):
        loadings = loadings.to_frame(name="PC1")

    pc_cols = list(loadings.columns[:n_components])

    ranked = loadings[pc_cols].abs().sum(axis=1).sort_values(ascending=False)
    if top_n_features is not None:
        selected_features = ranked.head(top_n_features).index
        loadings = loadings.loc[selected_features, pc_cols]
    else:
        loadings = loadings.loc[:, pc_cols]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        loadings,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
    )
    ax.set_title(f"PCA Loadings for Top {top_n_features or 'All'} Features")
    fig.tight_layout()

    return fig


def _prepare_biplot_data(
    result: PCAResult | PCADimReductionResult,
    *,
    group: str | None,
    dims: int,
    top_features: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str] | None]:
    """Extract scores, loadings, axis names, and pretty labels."""
    if dims not in (2, 3):
        raise ValueError("dims must be 2 or 3")

    if isinstance(result, PCAResult):
        scores = result.scores.copy()
        loadings = result.loadings.copy()
        pretty = None
    elif isinstance(result, PCADimReductionResult):
        if group is None:
            raise ValueError("group must be provided for PCADimReductionResult biplots.")
        try:
            gr: GroupPCAResult = next(gr for gr in result.group_results if gr.group.name == group)
        except StopIteration as exc:
            raise ValueError(f"group '{group}' not found in PCADimReductionResult") from exc

        scores = gr.pc_scores.copy()
        # Normalize column names to PC1.. for plotting
        rename_map = {col: f"PC{i + 1}" for i, col in enumerate(scores.columns)}
        scores = scores.rename(columns=rename_map)
        loadings = gr.loadings.copy()
        pretty = result.pretty_by_col
    else:
        raise TypeError("result must be PCAResult or PCADimReductionResult")

    pc_cols = [f"PC{i}" for i in range(1, dims + 1)]
    available = [c for c in pc_cols if c in scores.columns and c in loadings.columns]
    if len(available) < dims:
        raise ValueError(f"Requested {dims}D biplot but only found columns {available}")

    # Limit loadings to strongest features if requested
    if top_features is not None:
        norms = (loadings[pc_cols] ** 2).sum(axis=1).pow(0.5)
        keep = norms.nlargest(top_features).index
        loadings = loadings.loc[keep]

    return scores[pc_cols], loadings[pc_cols], pc_cols, pretty


def plot_biplot_plotly(
    result: PCAResult | PCADimReductionResult,
    *,
    group: str | None = None,
    dims: int = 2,
    top_features: int | None = 8,
    color: pd.Series | np.ndarray | None = None,
    color_palette: str | list | None = "Viridis",
    point_labels: bool = False,
    height: int = 700,
    width: int = 900,
) -> go.Figure:
    """Interactive biplot (2D/3D) for PCAResult or PCADimReductionResult using Plotly.

    Args:
        result: PCAResult or PCADimReductionResult.
        group: Required when `result` is PCADimReductionResult (group name).
        dims: 2 or 3 for the plot dimensionality.
        top_features: Number of strongest-loading features to draw (None = all).
        color: Optional array/Series for point coloring (aligned to scores index).
        color_palette: Plotly colorscale name or list applied when `color` is provided.
        point_labels: If True, show observation labels.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        plotly.graph_objects.Figure
    """
    scores, loadings, pc_cols, pretty = _prepare_biplot_data(
        result,
        group=group,
        dims=dims,
        top_features=top_features,
    )

    # Scale arrows relative to score spread
    score_span = float(np.abs(scores.to_numpy()).max() or 1.0)
    loading_span = float(np.abs(loadings.to_numpy()).max() or 1.0)
    scale = (score_span / loading_span) * 0.85 if loading_span else 1.0

    # Prepare score traces
    text = scores.index.astype(str) if point_labels else None
    marker_kwargs = dict(
        color=color,
        colorscale=color_palette if color is not None else None,
        size=7 if dims == 2 else 4,
        opacity=0.7,
    )

    if dims == 2:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=scores[pc_cols[0]],
                y=scores[pc_cols[1]],
                mode="markers+text" if point_labels else "markers",
                text=text,
                textposition="top center",
                marker=marker_kwargs,
                name="Scores",
            ),
        )

        # Arrows for loadings
        for feat, row in loadings.iterrows():
            lx, ly = row[pc_cols[0]] * scale, row[pc_cols[1]] * scale
            label = pretty.get(feat, feat) if pretty else feat
            fig.add_trace(
                go.Scatter(
                    x=[0, lx],
                    y=[0, ly],
                    mode="lines",
                    line=dict(color="firebrick", width=2),
                    showlegend=False,
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=[lx],
                    y=[ly],
                    mode="markers+text",
                    marker=dict(color="firebrick", size=6),
                    text=[label],
                    textposition="top center",
                    showlegend=False,
                ),
            )

        fig.update_xaxes(title=pc_cols[0])
        fig.update_yaxes(title=pc_cols[1])

    else:  # dims == 3
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=scores[pc_cols[0]],
                y=scores[pc_cols[1]],
                z=scores[pc_cols[2]],
                mode="markers+text" if point_labels else "markers",
                text=text,
                marker=marker_kwargs,
                name="Scores",
            ),
        )

        for feat, row in loadings.iterrows():
            lx, ly, lz = (row[c] * scale for c in pc_cols)
            label = pretty.get(feat, feat) if pretty else feat
            fig.add_trace(
                go.Scatter3d(
                    x=[0, lx],
                    y=[0, ly],
                    z=[0, lz],
                    mode="lines",
                    line=dict(color="firebrick", width=4),
                    showlegend=False,
                ),
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[lx],
                    y=[ly],
                    z=[lz],
                    mode="text",
                    text=[label],
                    textposition="top center",
                    showlegend=False,
                ),
            )

        fig.update_layout(
            scene=dict(
                xaxis_title=pc_cols[0],
                yaxis_title=pc_cols[1],
                zaxis_title=pc_cols[2],
            ),
        )

    fig.update_layout(
        title="PCA Biplot",
        width=width,
        height=height,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
