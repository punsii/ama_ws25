"""PCA-based dimensionality reduction for grouped correlated features."""

from dataclasses import dataclass
from typing import Self

import pandas as pd

from ama_tlbx.data.base_columns import BaseColumn
from ama_tlbx.data.views import DatasetView

from .base_analyser import BaseAnalyser
from .pca_analyzer import PCAAnalyzer


@dataclass(frozen=True)
class FeatureGroup:
    """A group of correlated features to be reduced via PCA.

    Attributes:
        name: Interpretable name for this group (e.g., "Immunization", "Mortality")
        features: List of feature names in this group - i.e. the columns in the original data.
    """

    name: str
    features: list[str | BaseColumn]


@dataclass(frozen=True)
class GroupPCAResult:
    """PCA dimensionality reduction results for a single feature group.

    Attributes:
        group: The feature group that was analyzed.
        pc_scores: DataFrame of retained PC scores for this group; columns named
            `{group.name}_PC1..k`, index aligned to the original data.
        explained_variance: DataFrame with columns `PC`, `variance`,
            `explained_ratio`, `cumulative_ratio` for all PCs prior to trimming.
        loadings: DataFrame of feature loadings for all PCs; index = original
            feature names, columns `PC1..PCm`.
        n_features: Number of original features in the group
        n_components: Number of principal components retained
        min_var_explained: Minimum variance threshold used to determine n_components
    """

    group: FeatureGroup
    pc_scores: pd.DataFrame
    explained_variance: pd.DataFrame
    loadings: pd.DataFrame
    n_features: int
    n_components: int
    min_var_explained: float

    @property
    def explained_variance_retained(self) -> pd.Series:
        """Explained variance ratio for the retained principal components."""
        return self.explained_variance["explained_ratio"].iloc[: self.n_components]

    @property
    def loadings_retained(self) -> pd.DataFrame:
        """Feature loadings on retained PCs only."""
        return self.loadings.iloc[:, : self.n_components]

    @property
    def pc1_scores(self) -> pd.Series:
        return self.pc_scores.iloc[:, 0]

    @property
    def explained_variance_pc1(self) -> float:
        return float(self.explained_variance_retained.iloc[0])

    @property
    def cumulative_variance_explained(self) -> float:
        return float(self.explained_variance_retained.sum())


@dataclass(frozen=True)
class PCADimReductionResult:
    """Complete results from PCA-based dimensionality reduction.

    Attributes:
        group_results: List of `GroupPCAResult` objects (one per group).
        reduced_df: DataFrame concatenating all retained PCs; columns are
            `{group}_PCj`, index aligned to the source data.
        original_n_features: Total number of features before reduction.
        reduced_n_features: Number after reduction (sum of retained PCs).
        pretty_by_col: Mapping from original column names to pretty names.
        min_var_explained_per_group: Variance threshold(s) used per group.
    """

    group_results: list[GroupPCAResult]
    reduced_df: pd.DataFrame
    original_n_features: int
    reduced_n_features: int
    pretty_by_col: dict[str, str]
    min_var_explained_per_group: list[float]

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio: original features / reduced features."""
        return self.original_n_features / self.reduced_n_features if self.reduced_n_features > 0 else 0.0

    @property
    def mean_explained_variance(self) -> float:
        """Mean variance explained by all retained PCs across all groups."""
        if not self.group_results:
            return 0.0
        return sum(gr.cumulative_variance_explained for gr in self.group_results) / len(self.group_results)

    @property
    def mean_explained_variance_pc1(self) -> float:
        """Mean variance explained by PC1 only across all groups (for backward compatibility)."""
        if not self.group_results:
            return 0.0
        return sum(gr.explained_variance_pc1 for gr in self.group_results) / len(self.group_results)


class PCADimReductionAnalyzer(BaseAnalyser):
    r"""Dimensionality reduction via PCA on correlated feature groups.

    This analyzer groups highly correlated features and reduces each group to
    the minimum number of principal components needed to explain a specified
    proportion of variance. This is useful when multiple features measure similar
    underlying constructs and can be combined without significant information loss.

    **Theory:**
    For a group of k correlated features :math:`X \in \mathbb{R}^{n \times k}`, PCA finds orthogonal bases of maximum variance. If features are highly correlated, the first
    few PCs will capture most of the variance. By setting a minimum explained
    variance threshold (e.g., 0.8), it determines how many PCs are needed to explain a cumulative variance of at least that threshold.

    **When to use:**
    - Multiple features measure highly related constructs (e.g., different immunization rates or economic indicators)
    - High correlation between features (|r| > 0.7)
    - Need to reduce multicollinearity before regression
    - Want interpretable dimension reduction (vs. all-feature PCA)
    - Want data-driven component selection based on information retention

    **Example:**
    >>> from ama_tlbx.data import LifeExpectancyDataset, LECol
    >>> from ama_tlbx.analysis import FeatureGroup
    >>> dataset = LifeExpectancyDataset.from_csv()
    >>> groups = [
    ...     FeatureGroup("immunization", [LECol.POLIO, LECol.HEPATITIS_B, LECol.DIPHTHERIA]),
    ...     FeatureGroup("child_mortality", [LECol.INFANT_DEATHS, LECol.UNDER_FIVE_DEATHS]),
    ... ]
    >>> result = (
    ...     dataset
    ...     .make_pca_dim_reduction_analyzer(groups, min_var_explained=0.9)
    ...     .fit()
    ...     .result()
    ... )
    >>> reduced_df = result.reduced_df

    Different thresholds per group:

    >>> result = dataset.make_pca_dim_reduction_analyzer(groups, min_var_explained=[0.9, 0.7]).fit().result()
    """

    def __init__(
        self,
        view: DatasetView,
        feature_groups: list[FeatureGroup],
        min_var_explained: float | list[float] = 0.8,
    ):
        """Dimensionality reduction via PCA on correlated feature groups.

        Args:
            view: Immutable dataset view (should contain standardized features)
            feature_groups: List of feature groups to reduce
            min_var_explained: Minimum proportion of variance to explain per group.
                The analyzer will keep the minimum number of PCs needed to reach
                this threshold.
                - float: Same threshold for all groups (default: 0.8)
                - list[float]: Specific threshold for each group (must match length)
                Values must be between 0 and 1.

        Raises:
            ValueError: If feature groups are empty, contain invalid columns, or min_var_explained specification is invalid
        """
        self._view = view
        self._group_results: list[GroupPCAResult] = []
        self._fitted = False

        if not self._view.is_standardized:
            raise ValueError("DatasetView must contain standardized features for PCA dimensionality reduction.")

        if not feature_groups:
            raise ValueError("At least one feature group is required")

        # Normalize feature groups -> strings
        normalized_groups: list[FeatureGroup] = []
        for group in feature_groups:
            if not group.features:
                raise ValueError(f"Feature group '{group.name}' is empty")
            feats = [f.value if isinstance(f, BaseColumn) else f for f in group.features]
            normalized_groups.append(FeatureGroup(name=group.name, features=feats))
        self._feature_groups = normalized_groups

        # Validate that features exist in the dataset
        all_features = set(self._view.features.columns)
        for group in self._feature_groups:
            invalid = set(group.features) - all_features
            if invalid:
                raise ValueError(f"Group '{group.name}' contains invalid features: {invalid}")

        # Normalize thresholds -> list[float] with one per group
        thresholds = (
            min_var_explained
            if isinstance(min_var_explained, list)
            else [min_var_explained] * len(self._feature_groups)
        )

        if len(thresholds) != len(self._feature_groups):
            raise ValueError(
                f"Length of min_var_explained list ({len(thresholds)}) must match "
                f"number of feature groups ({len(self._feature_groups)})",
            )

        if not all(0 < t <= 1 for t in thresholds):
            raise ValueError("All min_var_explained values must be between 0 and 1.")

        self._min_var_explained = thresholds

    def fit(self) -> Self:
        """Fit PCA on each feature group and dynamically determine n_components.

        For each group:
        1. Extract the features belonging to that group
        2. Fit PCA with all components
        3. Determine minimum number of PCs needed to reach variance threshold
        4. Extract PC scores, explained variance, and loadings
        5. Store results for interpretation

        Returns:
            Self for method chaining
        """
        self._group_results = []

        for i, group in enumerate(self._feature_groups):
            # Determine variance threshold for this group
            threshold = self._min_var_explained[i]

            # Create a view with only this group's features
            group_view = DatasetView(
                df=self._view.df[group.features].copy(),
                pretty_by_col={k: v for k, v in self._view.pretty_by_col.items() if k in group.features},
                numeric_cols=group.features,
                target_col=None,
            )

            # Fit PCA on this group (always fit all components first to analyze variance)
            pca_analyzer = PCAAnalyzer(group_view)
            pca_analyzer.fit(n_components=None)
            pca_result = pca_analyzer.result()

            # Determine number of components needed to reach variance threshold
            cumulative_var = pca_result.explained_variance["explained_ratio"].cumsum()
            n_comp = int((cumulative_var >= threshold).idxmax() + 1)  # +1 because index is 0-based

            # Extract PC scores (keep only the determined number)
            pc_scores = pca_result.scores.iloc[:, :n_comp].copy()
            # Rename columns with group name
            pc_scores.columns = [f"{group.name}_PC{j + 1}" for j in range(n_comp)]

            # Get all loadings (not just retained ones)
            all_loadings = pca_result.loadings.copy()
            all_loadings.columns = [f"PC{j + 1}" for j in range(len(all_loadings.columns))]

            # Store results
            group_result = GroupPCAResult(
                group=group,
                pc_scores=pc_scores,
                explained_variance=pca_result.explained_variance,
                loadings=all_loadings,
                n_features=len(group.features),
                n_components=n_comp,
                min_var_explained=threshold,
            )
            self._group_results.append(group_result)

        self._fitted = True
        return self

    def result(self) -> PCADimReductionResult:
        """Assemble dimensionality reduction results.

        Returns:
            PCADimReductionResult containing all group results and reduced DataFrame
        """
        if not self._fitted:
            raise ValueError("Analyzer not fitted. Call fit() first.")

        # Combine PC scores from all groups into a single DataFrame
        reduced_df = pd.concat(map(lambda gr: gr.pc_scores, self._group_results), axis=1)
        reduced_df.index = self._view.df.index

        return PCADimReductionResult(
            group_results=self._group_results,
            reduced_df=reduced_df,
            original_n_features=len({feature for group in self._feature_groups for feature in group.features}),
            reduced_n_features=sum(map(lambda gr: gr.n_components, self._group_results)),
            pretty_by_col=dict(self._view.pretty_by_col),
            min_var_explained_per_group=self._min_var_explained,
        )
