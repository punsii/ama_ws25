"""Correlation analysis for dataset features."""

from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd

from ama_tlbx.data.views import DatasetView

from .base_analyser import BaseAnalyser


@dataclass(frozen=True)
class CorrelationResult:
    """Correlation analysis outputs grouped for plotting and reporting.

    Attributes:
        matrix: Full Pearson correlation matrix (DataFrame; rows/cols = features in view).
        pretty_by_col: Mapping from raw feature names to presentation labels.
        feature_pairs: DataFrame with columns `feature_a`, `feature_b`, `correlation`,
            `abs_correlation`, `pair`; sorted by strongest absolute correlations.
        target_correlations: Optional DataFrame with columns `feature`, `correlation`
            for feature-vs-target correlations (sorted descending).
    """

    matrix: pd.DataFrame
    pretty_by_col: dict[str, str]
    feature_pairs: pd.DataFrame
    target_correlations: pd.DataFrame | None = None

    # ------------------------------------------------------------------ plotting shortcuts
    def plot_heatmap(self, **kwargs: object):
        """Plot correlation heatmap using the plotting helper."""
        from ama_tlbx.plotting.correlation_plots import plot_correlation_heatmap  # noqa: PLC0415

        return plot_correlation_heatmap(self, **kwargs)

    def plot_top_pairs(self, **kwargs: object):
        """Plot top positive/negative correlated pairs."""
        from ama_tlbx.plotting.correlation_plots import plot_top_correlated_pairs  # noqa: PLC0415

        return plot_top_correlated_pairs(self, **kwargs)

    def plot_target_correlations(self, **kwargs: object):
        """Plot correlations with the target variable."""
        from ama_tlbx.plotting.correlation_plots import plot_target_correlations  # noqa: PLC0415

        return plot_target_correlations(self, **kwargs)


class CorrelationAnalyzer(BaseAnalyser):
    """Analyzer for computing feature correlations.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from ama_tlbx.data.life_expectancy_dataset import LifeExpectancyDataset
        >>> from ama_tlbx.plotting.correlation_plots import (
        ...     plot_correlation_heatmap,
        ...     plot_top_correlated_pairs,
        ...     plot_target_correlations,
        ... )
        >>> ds = LifeExpectancyDataset.from_csv()
        >>> corr_res = ds.make_correlation_analyzer(standardized=True, include_target=True).fit().result()
        >>> fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        >>> _ = plot_correlation_heatmap(corr_res, ax=axes[0])
        >>> pos_fig, neg_fig = plot_top_correlated_pairs(corr_res, n=10, axes=axes)
        >>> # Target correlations on separate axes
        >>> fig_tc, axes_tc = plt.subplots(1, 2, figsize=(18, 6))
        >>> _ = plot_target_correlations(corr_res, axes=axes_tc)
    """

    def __init__(self, view: DatasetView):
        """Initialize the correlation analyzer with a dataset view."""
        self._view = view
        self._corr_mat: pd.DataFrame | None = None

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Compute the Pearson correlation matrix via :meth:`pandas.DataFrame.corr`."""
        if self._corr_mat is None:
            self._corr_mat = self._view.df.corr(numeric_only=True)
        return self._corr_mat

    def get_top_correlated_pairs(self, n: int = 20) -> pd.DataFrame:
        """Return the strongest absolute Pearson correlations between feature pairs.

        The implementation vectorizes the symmetric matrix by masking the upper triangle
        (excluding the diagonal) using :func:`np.triu`, then stacks the remaining
        values for efficient sorting.
        """
        corr_matrix = self.get_correlation_matrix()
        mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)

        pairs = (
            corr_matrix.where(mask)
            .melt(ignore_index=False, var_name="feature_b", value_name="correlation")
            .dropna()
            .reset_index()
            .rename(columns={"index": "feature_a"})
            .assign(
                abs_correlation=lambda d: d.correlation.abs(),
                pair=lambda d: d.feature_a + " vs " + d.feature_b,
            )
            .sort_values("abs_correlation", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )
        return pairs

    def get_target_correlations(self) -> pd.DataFrame:
        """
        Return Pearson correlations between each feature and the configured target.

        This uses the correlation matrix computed above and therefore shares the same
        statistical interpretation. See [Wikipdia :: Pearson Correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
        for the underlying theory.

        Returns:
            DataFrame of features and their correlation with the target variable.
        """
        if not self._view.target_col:
            raise ValueError("Dataset view has no target column configured.")

        corr_matrix = self.get_correlation_matrix()

        if self._view.target_col not in corr_matrix.index:
            raise ValueError(f"Target column '{self._view.target_col}' not found in data")

        assert self._view.target_col is not None, "get_target_correlations requires a target_col"

        return (
            corr_matrix.loc[self._view.target_col]
            .drop(self._view.target_col)
            .sort_values(ascending=False)
            .to_frame(name="correlation")
            .assign(feature=lambda d: d.index)
            .reset_index(drop=True)
        )

    def fit(self) -> Self:
        """Compute correlation matrix."""
        self.get_correlation_matrix()

        return self

    def result(self, *, top_n_pairs: int = 20) -> CorrelationResult:
        matrix = self.get_correlation_matrix()
        pairs = self.get_top_correlated_pairs(n=top_n_pairs)

        target_corr = (
            self.get_target_correlations() if self._view.target_col and self._view.target_col in matrix.index else None
        )

        return CorrelationResult(
            matrix=matrix,
            pretty_by_col=self._view.pretty_by_col,
            feature_pairs=pairs,
            target_correlations=target_corr,
        )
