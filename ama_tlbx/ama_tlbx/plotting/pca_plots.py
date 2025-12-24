"""PCA visualization functions."""

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.figure import Figure

from ama_tlbx.analysis.pca_analyzer import PCAResult
from ama_tlbx.analysis.pca_dim_reduction import GroupPCAResult


def _ensure_loadings_df(loadings: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Guarantee a DataFrame of loadings."""
    if isinstance(loadings, pd.Series):
        return loadings.to_frame(name=loadings.name or "PC1")
    return loadings


def _compute_top_features(loadings: pd.DataFrame) -> pd.Index:
    """Rank features by L2 norm across PCs."""
    return (loadings.pow(2).sum(axis=1) ** 0.5).sort_values(ascending=False).index


def _standardize_pca_result(
    result: PCAResult | GroupPCAResult,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Index]:
    """Return (scores, loadings, explained, top_features_global) with unified column names."""
    if isinstance(result, GroupPCAResult):
        loadings = _ensure_loadings_df(result.loadings.copy())
        scores = result.pc_scores.copy()
        prefix = f"{result.group.name}_"
        scores = scores.rename(columns=lambda c: c.removeprefix(prefix))
        explained = result.explained_variance
        top_features_global = _compute_top_features(loadings)
    elif isinstance(result, PCAResult):
        loadings = _ensure_loadings_df(result.loadings.copy())
        scores = result.scores.copy()
        explained = result.explained_variance
        top_features_global = getattr(result, "top_features_global", _compute_top_features(loadings))
    else:
        raise TypeError("result must be PCAResult or GroupPCAResult")

    return scores, loadings, explained, top_features_global


def _normalize_pc_name(pc: str | int) -> str:
    """Convert component identifiers to canonical 'PCk' strings."""
    if isinstance(pc, int):
        return f"PC{pc}"
    pc_str = str(pc)
    return pc_str if pc_str.upper().startswith("PC") else f"PC{pc_str}"


def _select_pc_columns(
    available: pd.Index,
    n_components: int,
    pc_subset: Sequence[str | int] | None = None,
    required_len: int | None = None,
) -> list[str]:
    """Resolve which PC columns to use, validating availability and length."""
    if pc_subset is None:
        pc_cols = [f"PC{i}" for i in range(1, n_components + 1)]
    else:
        pc_cols = [_normalize_pc_name(pc) for pc in pc_subset]

    if required_len is not None and len(pc_cols) != required_len:
        raise ValueError(f"Expected {required_len} components, got {len(pc_cols)}")

    missing = [pc for pc in pc_cols if pc not in available]
    if missing:
        raise ValueError(f"Requested components {missing} not available. Available: {list(available)}")

    return pc_cols


def plot_explained_variance(
    result: PCAResult,
    figsize: tuple[int, int] = (10, 6),
    bar: Literal["explained_ratio", "variance"] = "variance",
) -> Figure:
    """Plot explained variance (or explained ratio) with cumulative curve.

    Combines [:func:`seaborn.barplot`](https://seaborn.pydata.org/generated/seaborn.barplot.html) and
    [:func:`seaborn.lineplot`](https://seaborn.pydata.org/generated/seaborn.lineplot.html) to mirror the
    classic PCA scree plot.
    """
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
    result: PCAResult | GroupPCAResult,
    n_components: int = 3,
    top_n_features: int | None = None,
    figsize: tuple[int, int] = (12, 6),
    pc_subset: Sequence[str | int] | None = None,
) -> Figure:
    """Heatmap of loadings ordered by feature importance.

    Displays component loadings via [:func:`seaborn.heatmap`](https://seaborn.pydata.org/generated/seaborn.heatmap.html), optionally
    restricted to the strongest features.

    Args:
        result: PCAResult or GroupPCAResult to visualize.
        n_components: Number of leading PCs to show (ignored when pc_subset is provided).
        top_n_features: Limit to the top-N features by loading magnitude across the selected PCs.
        figsize: Figure size.
        pc_subset: Explicit list of PC names or indices to plot (e.g., ["PC1", "PC3"] or [1, 3]).
    """
    _, loadings, _, top_features_global = _standardize_pca_result(result)
    pc_cols = _select_pc_columns(loadings.columns, n_components, pc_subset)
    ranked = top_features_global
    if top_n_features is not None:
        ranked = ranked[:top_n_features]
    loadings = loadings.loc[ranked, pc_cols]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(f"PCA Loadings for Top {top_n_features or 'All'} Features")
    fig.tight_layout()
    return fig


def plot_biplot_plotly(
    result: PCAResult | GroupPCAResult,
    *,
    dims: int = 2,
    pc_axes: Sequence[str | int] | None = None,
    top_features: int | None = 8,
    color: pd.Series | np.ndarray | None = None,
    color_palette: str | list | None = "orrd",
    hover_metadata: pd.DataFrame | None = None,
    point_labels: bool = False,
    height: int = 700,
    width: int = 900,
) -> go.Figure:
    """Interactive biplot (2D/3D) with optional hover metadata.

    Implemented with Plotly's [:class:`plotly.graph_objects.Scatter`](https://plotly.com/python/line-and-scatter/) /
    [:class:`plotly.graph_objects.Scatter3d`](https://plotly.com/python/line-and-scatter/) for scores and loadings.
    """
    if dims not in (2, 3):
        raise ValueError("dims must be 2 or 3")

    scores, loadings, _, top_features_global = _standardize_pca_result(result)
    pc_cols = _select_pc_columns(loadings.columns, dims, pc_subset=pc_axes, required_len=dims)
    if any(pc not in scores.columns for pc in pc_cols):
        raise ValueError(f"Requested {dims}D biplot but only found score columns {list(scores.columns)}")

    if top_features is not None:
        loadings = loadings.loc[top_features_global[:top_features]]

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
        title=f"{getattr(result, 'group', None).name if isinstance(result, GroupPCAResult) else 'PCA'} Biplot",
        width=width,
        height=height,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
