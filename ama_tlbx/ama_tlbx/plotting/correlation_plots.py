"""Correlation analysis visualization functions."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import legend
from matplotlib.figure import Figure

from ama_tlbx.analysis.correlation_analyzer import CorrelationResult


def plot_correlation_heatmap(
    result: CorrelationResult,
    figsize: tuple[int, int] = (20, 20),
    **kwargs: object,
) -> Figure:
    """Plot correlation heatmap of all features."""
    fig, ax = plt.subplots(figsize=figsize)

    label_map = {col: result.pretty_by_col.get(col, col) for col in result.matrix.columns}

    sns.heatmap(
        result.matrix.rename(index=label_map, columns=label_map),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
        **kwargs,  # type: ignore[arg-type]
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="y", rotation=0)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()

    return fig


def _prettify_pair_columns(
    pairs: pd.DataFrame,
    pretty_by_col: dict[str, str],
) -> pd.DataFrame:
    """Attach pretty labels for plotting convenience."""

    def _pretty_pair(row: pd.Series) -> str:
        a = pretty_by_col.get(row["feature_a"], row["feature_a"])
        b = pretty_by_col.get(row["feature_b"], row["feature_b"])
        return f"{a} vs {b}"

    return pairs.assign(
        pretty_pair=lambda d: d.apply(_pretty_pair, axis=1),
    )


def plot_top_correlated_pairs(
    result: CorrelationResult,
    n: int = 20,
    threshold: float | None = 0.8,
    figsize: tuple[int, int] = (12, 10),
) -> tuple[Figure, Figure]:
    """Plot top positively and negatively correlated feature pairs.

    Args:
        result: CorrelationResult from CorrelationAnalyzer.
        n: Number of top correlated pairs to display in each plot.
        threshold: Minimum absolute correlation to justify grouping features.
        figsize: Figure size for each plot.
    """
    pairs = _prettify_pair_columns(result.feature_pairs, result.pretty_by_col)
    positive = pairs.query("correlation > 0").nlargest(n, "abs_correlation")
    negative = pairs.query("correlation < 0").nlargest(n, "abs_correlation")

    fig_pos, ax_pos = plt.subplots(figsize=figsize)
    sns.barplot(data=positive, x="correlation", y="pretty_pair", color="tab:red", ax=ax_pos)
    ax_pos.set_title(f"Top {len(positive)} Positive Correlations")
    ax_pos.set_xlabel("Pearson Correlation")
    ax_pos.axvline(0, color="black", linewidth=1, linestyle="--")
    if threshold is not None:
        ax_pos.axvline(threshold, color="tab:orange", linewidth=2, linestyle="--")
    fig_pos.tight_layout()

    fig_neg, ax_neg = plt.subplots(figsize=figsize)
    sns.barplot(data=negative, x="correlation", y="pretty_pair", color="tab:blue", ax=ax_neg)
    ax_neg.set_title(f"Top {len(negative)} Negative Correlations")
    ax_neg.set_xlabel("Pearson Correlation")
    ax_neg.axvline(0, color="black", linewidth=1, linestyle="--")
    if threshold is not None:
        ax_neg.axvline(-threshold, color="tab:orange", linewidth=2, linestyle="--")
    fig_neg.tight_layout()

    return fig_pos, fig_neg


def plot_target_correlations(
    result: CorrelationResult,
    n: int = 10,
    figsize: tuple[int, int] = (10, 6),
) -> tuple[Figure, Figure]:
    """Plot top positive and negative correlations with target variable."""
    if result.target_correlations is None:
        msg = "CorrelationResult does not include target correlations."
        raise ValueError(msg)

    target_corr = result.target_correlations.copy()
    target_corr["pretty_feature"] = target_corr["feature"].map(lambda c: result.pretty_by_col.get(c, c))

    positive = target_corr.nlargest(n, "correlation")
    negative = target_corr.nsmallest(n, "correlation").sort_values("correlation", ascending=True)

    fig_pos, ax_pos = plt.subplots(figsize=figsize)
    sns.barplot(data=positive, x="correlation", y="pretty_feature", color="#d62728", ax=ax_pos)
    ax_pos.set_title("Top Positive Correlations with Target")
    ax_pos.set_xlabel("Pearson Correlation")
    ax_pos.axvline(0, color="black", linewidth=1, linestyle="--")
    fig_pos.tight_layout()

    fig_neg, ax_neg = plt.subplots(figsize=figsize)
    sns.barplot(data=negative, x="correlation", y="pretty_feature", color="#1f77b4", ax=ax_neg)
    ax_neg.set_title("Top Negative Correlations with Target")
    ax_neg.set_xlabel("Pearson Correlation")
    ax_neg.axvline(0, color="black", linewidth=1, linestyle="--")
    fig_neg.tight_layout()

    return fig_pos, fig_neg
