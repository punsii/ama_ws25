"""PCA analysis for dimensionality reduction and feature interpretation."""

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sklearn.decomposition import PCA

from ama_tlbx.data_handling.views import DatasetView


@dataclass(frozen=True)
class PCAResult:
    """PCA outputs packaged for downstream visualization and reporting.

    Attributes:
        scores: Observation coordinates in principal-component space.
        loadings: Feature loadings relating original variables to components.
        explained_variance: Variance, ratio, and cumulative ratio per component.
    """

    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance: pd.DataFrame


class PCAAnalyzer:
    """Analyzer for Principal Component Analysis (PCA).

    Provides methods for fitting PCA models, transforming data, and
    summarizing component loadings.
    """

    def __init__(self, view: DatasetView):
        """Initialize the PCA analyzer."""
        self._view = view
        self._pca_model: PCA | None = None
        self._feature_names: list[str] = []

    def fit(
        self,
        n_components: int | None = None,
        exclude_cols: list[str] | None = None,
    ) -> "PCAAnalyzer":
        r"""Fit a PCA model using :class:`sklearn.decomposition.PCA`.

        Principal Component Analysis projects the standardized data matrix
        :math:`\mathbf{X}` onto orthogonal directions of maximal variance, enabling
        dimensionality reduction. See
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        for details.
        """
        features = self._view.features.copy()

        # Exclude specified columns
        exclude_cols = exclude_cols or []
        features = features.drop(columns=exclude_cols, errors="ignore")

        if features.empty:
            raise ValueError("No features remaining after exclusions for PCA fitting.")

        self._feature_names = features.columns.tolist()
        self._pca_model = PCA(n_components=n_components)
        self._pca_model.fit(features)

        return self

    def transform(self, n_components: int | None = None) -> pd.DataFrame:
        """Project observations into principal-component space via :meth:`PCA.transform`."""
        if self._pca_model is None:
            raise ValueError("PCA model not fitted. Call fit() first.")

        features = self._view.features.loc[:, self._feature_names]
        transformed = self._pca_model.transform(features)

        if n_components is not None:
            transformed = transformed[:, :n_components]

        return pd.DataFrame(
            transformed,
            columns=[f"PC{i + 1}" for i in range(transformed.shape[1])],
            index=self._view.data.index,
        )

    def get_explained_variance(self) -> pd.DataFrame:
        """Summarize component-wise variance contributions and cumulative totals."""
        if self._pca_model is None:
            raise ValueError("PCA model not fitted. Call fit() first.")

        n_components = len(self._pca_model.explained_variance_ratio_)

        return pd.DataFrame(
            {
                "PC": [f"PC{i + 1}" for i in range(n_components)],
                "variance": self._pca_model.explained_variance_,
                "explained_ratio": self._pca_model.explained_variance_ratio_,
                "cumulative_ratio": self._pca_model.explained_variance_ratio_.cumsum(),
            },
        )

    def get_loading_vectors(
        self,
        component: int | None = None,
    ) -> pd.DataFrame | pd.Series:
        """Return PCA loading vectors linking original features to component axes."""
        if self._pca_model is None:
            raise ValueError("PCA model not fitted. Call fit() first.")

        n_components = self._pca_model.n_components_

        loadings_df = pd.DataFrame(
            self._pca_model.components_.T,
            index=self._feature_names,
            columns=[f"PC{i}" for i in range(1, n_components + 1)],
        )

        if component is None:
            return loadings_df

        if not isinstance(component, int):
            msg = f"Component must be an integer, got {type(component).__name__}"
            raise TypeError(msg)

        pc_name = f"PC{component}"
        if pc_name not in loadings_df.columns:
            msg = f"Component {component} not found. Available: PC1-PC{n_components}"
            raise ValueError(msg)

        return loadings_df[pc_name]

    def get_top_loading_features(
        self,
        n_components: int = 3,
        method: Literal["sum", "max", "l2", "euclidean"] = "sum",
    ) -> pd.Index:
        """Rank features by aggregated loading strength across leading components."""
        if self._pca_model is None:
            raise ValueError("PCA model not fitted. Call fit() first.")

        loadings = self.get_loading_vectors()
        if isinstance(loadings, pd.Series):
            raise TypeError("Expected DataFrame from get_loading_vectors()")

        pc_cols = [f"PC{i + 1}" for i in range(min(n_components, loadings.shape[1]))]

        method_key = method.lower()
        if method_key == "sum":
            importance = loadings[pc_cols].abs().sum(axis=1)
        elif method_key == "max":
            importance = loadings[pc_cols].abs().max(axis=1)
        elif method_key in {"l2", "euclidean"}:
            importance = (loadings[pc_cols] ** 2).sum(axis=1).pow(0.5)
        else:
            msg = "method must be one of {'sum', 'max', 'l2'}"
            raise ValueError(msg)

        return importance.sort_values(ascending=False).index

    def result(self) -> PCAResult:
        """Collect PCA scores, loadings, and variance diagnostics for downstream use."""
        if self._pca_model is None:
            raise ValueError("PCA model not fitted. Call fit() first.")

        scores = self.transform()
        loadings = self.get_loading_vectors()
        if isinstance(loadings, pd.Series):
            loadings = loadings.to_frame(name="PC1")
        explained = self.get_explained_variance()
        return PCAResult(scores=scores, loadings=loadings, explained_variance=explained)
