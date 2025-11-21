"""PCA visualization functions."""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.figure import Figure

from ama_tlbx.analysis.pca_analyzer import PCAResult


def plot_explained_variance(
    result: PCAResult,
    figsize: tuple[int, int] = (10, 6),
    bar: Literal["explained_ratio", "variance"] = "variance",
) -> Figure:
    """Plot explained variance (or explained ratio) with cumulative curve."""
    fig, ax1 = plt.subplots(figsize=figsize)
    x = np.arange(len(result.explained_variance))
    sns.barplot(x=x, y=result.explained_variance[bar], ax=ax1, color="skyblue")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel(bar.replace("_", " ").title(), color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    sns.lineplot(x=x, y=result.explained_variance["cumulative_ratio"], marker="o", color="red", ax=ax2)
    ax2.set_ylabel("Cumulative Variance Explained", color="red")
    ax2.set_yticks(np.arange(0, 1.1, 0.1))
    ax2.tick_params(axis="y", labelcolor="red")

    ax1.set_xticks(x)
    ax1.set_xticklabels(result.explained_variance["PC"])
    ax1.set_title("PCA Explained Variance")
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_loadings_heatmap(
    result: PCAResult,
    n_components: int = 3,
    top_n_features: int | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> Figure:
    """Heatmap of loadings ordered by precomputed ranking."""
    loadings = result.loadings
    if isinstance(loadings, pd.Series):
        loadings = loadings.to_frame(name="PC1")
    pc_cols = list(loadings.columns[:n_components])
    ranked = result.top_features_global
    if top_n_features is not None:
        ranked = ranked[:top_n_features]
    loadings = loadings.loc[ranked, pc_cols]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(f"PCA Loadings for Top {top_n_features or 'All'} Features")
    fig.tight_layout()
    return fig


def plot_biplot_plotly(
    result: PCAResult,
    *,
    dims: int = 2,
    top_features: int | None = 8,
    color: pd.Series | np.ndarray | None = None,
    color_palette: str | list | None = "orrd",
    hover_metadata: pd.DataFrame | None = None,
    point_labels: bool = False,
    height: int = 700,
    width: int = 900,
) -> go.Figure:
    """Interactive biplot (2D/3D) with optional hover metadata."""
    if dims not in (2, 3):
        raise ValueError("dims must be 2 or 3")

    scores = result.scores.copy()
    loadings = result.loadings.copy()
    pc_cols = [f"PC{i}" for i in range(1, dims + 1)]
    if any(pc not in scores.columns or pc not in loadings.columns for pc in pc_cols):
        raise ValueError(f"Requested {dims}D biplot but only found columns {list(scores.columns)}")

    if top_features is not None and hasattr(result, "top_features_global"):
        loadings = loadings.loc[result.top_features_global[:top_features]]

    score_span = float(np.abs(scores[pc_cols].to_numpy()).max() or 1.0)
    loading_span = float(np.abs(loadings[pc_cols].to_numpy()).max() or 1.0)
    scale = (score_span / loading_span) * 0.85 if loading_span else 1.0

    text = scores.index.astype(str) if point_labels else None
    hover_df = hover_metadata.reindex(scores.index) if hover_metadata is not None else None
    hover_cols = list(hover_df.columns) if hover_df is not None else []

    marker_kwargs = dict(
        color=color,
        colorscale=color_palette if color is not None else None,
        size=7 if dims == 2 else 4,
        opacity=0.8,
    )

    fig = go.Figure()
    if dims == 2:
        hover_lines = [
            f"{pc_cols[0]}: %{{x:.3f}}",
            f"{pc_cols[1]}: %{{y:.3f}}",
        ] + [f"{col}: %{{customdata[{i}]}}" for i, col in enumerate(hover_cols)]

        fig.add_trace(
            go.Scatter(
                x=scores[pc_cols[0]],
                y=scores[pc_cols[1]],
                mode="markers+text" if point_labels else "markers",
                text=text,
                textposition="top center",
                marker=marker_kwargs,
                customdata=hover_df.to_numpy() if hover_df is not None else None,
                hovertemplate="<br>".join(hover_lines + ["<extra></extra>"]) if hover_df is not None else None,
                name="Scores",
            ),
        )
        for feat, row in loadings.iterrows():
            lx, ly = row[pc_cols[0]] * scale, row[pc_cols[1]] * scale
            fig.add_trace(
                go.Scatter(x=[0, lx], y=[0, ly], mode="lines", line=dict(color="firebrick", width=2), showlegend=False),
            )
            fig.add_trace(
                go.Scatter(
                    x=[lx],
                    y=[ly],
                    mode="markers+text",
                    marker=dict(color="firebrick", size=6),
                    text=[feat],
                    textposition="top center",
                    showlegend=False,
                ),
            )
        fig.update_xaxes(title=pc_cols[0])
        fig.update_yaxes(title=pc_cols[1])
    else:
        hover_lines = [
            f"{pc_cols[0]}: %{{x:.3f}}",
            f"{pc_cols[1]}: %{{y:.3f}}",
            f"{pc_cols[2]}: %{{z:.3f}}",
        ] + [f"{col}: %{{customdata[{i}]}}" for i, col in enumerate(hover_cols)]

        fig.add_trace(
            go.Scatter3d(
                x=scores[pc_cols[0]],
                y=scores[pc_cols[1]],
                z=scores[pc_cols[2]],
                mode="markers+text" if point_labels else "markers",
                text=text,
                marker=marker_kwargs,
                customdata=hover_df.to_numpy() if hover_df is not None else None,
                hovertemplate="<br>".join(hover_lines + ["<extra></extra>"]) if hover_df is not None else None,
                name="Scores",
            ),
        )
        for feat, row in loadings.iterrows():
            lx, ly, lz = (row[c] * scale for c in pc_cols)
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
                    text=[feat],
                    textposition="top center",
                    showlegend=False,
                ),
            )
        fig.update_layout(scene=dict(xaxis_title=pc_cols[0], yaxis_title=pc_cols[1], zaxis_title=pc_cols[2]))

    fig.update_layout(
        title="PCA Biplot",
        width=width,
        height=height,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
