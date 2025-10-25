"""Correlation analysis for dataset features."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ama_tlbx.data.views import DatasetView


@dataclass(frozen=True)
class CorrelationResult:
    """Correlation analysis outputs grouped for plotting and reporting.

    Attributes:
        matrix: Full Pearson correlation matrix for the selected view.
        pretty_by_col: Mapping from raw feature names to presentation labels.
        feature_pairs: Sorted table of top absolute correlations between features.
        target_correlations: Optional correlations between each feature and the target.
    """

    matrix: pd.DataFrame
    pretty_by_col: dict[str, str]
    feature_pairs: pd.DataFrame
    target_correlations: pd.DataFrame | None = None


class CorrelationAnalyzer:
    """Analyzer for computing feature correlations."""

    def __init__(self, view: DatasetView):
        """Initialize the correlation analyzer with a dataset view."""
        self._view = view

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Compute the Pearson correlation matrix via :meth:`pandas.DataFrame.corr`."""
        return self._view.data.corr(numeric_only=True)

    def get_top_correlated_pairs(self, n: int = 20) -> pd.DataFrame:
        """Return the strongest absolute Pearson correlations between feature pairs.

        The implementation vectorizes the symmetric matrix by masking the upper triangle
        (excluding the diagonal) using :func:`np.triu`, then stacks the remaining
        values for efficient sorting.
        """
        corr_matrix = self.get_correlation_matrix()
        mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        # Use melt instead of stack for better pandas compatibility
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
        statistical interpretation. See https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
        for the underlying theory.

        Returns:
            DataFrame of features and their correlation with the target variable.
        """
        if not self._view.target_col:
            msg = "Dataset view has no target column configured."
            raise ValueError(msg)

        corr_matrix = self.get_correlation_matrix()

        if self._view.target_col not in corr_matrix.index:
            msg = f"Target column '{self._view.target_col}' not found in data"
            raise ValueError(msg)

        correlations = corr_matrix.loc[self._view.target_col].drop(self._view.target_col)
        return (
            correlations.sort_values(ascending=False)
            .to_frame(name="correlation")
            .assign(feature=lambda d: d.index)
            .reset_index(drop=True)
        )

    def compute(self, *, top_n_pairs: int = 20) -> CorrelationResult:
        """Assemble correlation results for downstream plotting and reporting."""
        matrix = self.get_correlation_matrix()
        pairs = self.get_top_correlated_pairs(n=top_n_pairs)
        target_corr = None
        if self._view.target_col:
            target_corr = self.get_target_correlations()
        return CorrelationResult(
            matrix=matrix,
            pretty_by_col=dict(self._view.pretty_by_col),
            feature_pairs=pairs,
            target_correlations=target_corr,
        )
