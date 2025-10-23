"""Correlation analysis for dataset features."""

import pandas as pd


class CorrelationAnalyzer:
    """Analyzer for computing feature correlations.

    This class provides methods for calculating correlation matrices,
    identifying highly correlated feature pairs, and analyzing relationships
    with target variables.
    """

    def __init__(self, df: pd.DataFrame):
        """Initialize the correlation analyzer.

        Args:
            df: DataFrame with numeric features to analyze
        """
        self.df = df

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix for all numeric features."""
        return self.df.corr()

    def get_top_correlated_pairs(self, n: int = 20, ascending: bool = False) -> pd.DataFrame:
        """Get top N correlated feature pairs.

        Args:
            n: Number of pairs to return
            ascending: If False, return highest correlations; if True, return lowest

        Returns:
            DataFrame with correlated pairs and their correlation values
        """
        corr_matrix = self.df.corr()

        return (
            corr_matrix.where(
                pd.DataFrame(
                    [[i < j for j in range(len(corr_matrix))] for i in range(len(corr_matrix))],
                    index=corr_matrix.index,
                    columns=corr_matrix.columns,
                ),
            )
            .melt(ignore_index=False, var_name="level_1", value_name="correlation")
            .reset_index(names="level_0")
            .dropna()
            .assign(pair=lambda d: d.level_0 + " vs " + d.level_1)
            .sort_values("correlation", ascending=ascending)
            .head(n)
            .reset_index(drop=True)
        )

    def get_target_correlations(self, target_col: str) -> pd.DataFrame:
        """Get correlations of all features with a target variable.

        Args:
            target_col: Name of target variable column

        Returns:
            DataFrame with features and their correlation with target
        """
        corr_matrix = self.get_correlation_matrix()

        if target_col not in corr_matrix.index:
            msg = f"Target column '{target_col}' not found in data"
            raise ValueError(msg)

        return (
            corr_matrix.loc[target_col]
            .drop(target_col)
            .sort_values(ascending=False)  # type: ignore
            .to_frame(name="correlation")
            .assign(feature=lambda d: d.index)
            .reset_index(drop=True)
        )
