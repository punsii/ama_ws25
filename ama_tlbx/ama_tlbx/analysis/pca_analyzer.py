"""PCA analysis for dimensionality reduction and feature interpretation."""

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sklearn.decomposition import PCA

from ama_tlbx.data.views import DatasetView


@dataclass(frozen=True)
class PCAResult:
    """PCA outputs packaged for downstream visualization and reporting.

    Attributes:
        scores: DataFrame of observation coordinates w.r.t. the principal components; columns named `PC1..PCk`, same index as in the input data.
        loadings: DataFrame of feature loadings; index = original feature names, columns `PC1..PCk` matching the fitted components. Each `PCi` column is the unit-length eigenvector of the sample covariance matrix **X** associated with the i-th largest eigenvalue. The j-th value of each eigenvector expresses the relative weight of each feature in that orthonormal basis vector; hence ||PCi||_2 = 1. The sings of the loadings are arbitrary.
        explained_variance: DataFrame with columns `PC`, `variance`,
            `explained_ratio`, `cumulative_ratio` for each component; the `variance` of each PC equals their eigenvalues, ordered from largest to smallest, describing the variance captured along each orthogonal principal axis.
        top_features_global: Features ranked by overall loading strength (L2 across PCs).
        top_features_per_pc: Per-component ranking by absolute loading.
    """

    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance: pd.DataFrame
    top_features_global: pd.Index
    top_features_per_pc: dict[str, pd.Index]

    # ------------------------------------------------------------------ plotting shortcuts
    def plot_explained_variance(self, **kwargs: object):
        """Plot explained variance using shared plotting helper."""
        from ama_tlbx.plotting.pca_plots import plot_explained_variance  # noqa: PLC0415

        return plot_explained_variance(self, **kwargs)

    def plot_loadings_heatmap(self, **kwargs: object):
        """Plot loadings heatmap (optionally top-N features)."""
        from ama_tlbx.plotting.pca_plots import plot_loadings_heatmap  # noqa: PLC0415

        return plot_loadings_heatmap(self, **kwargs)

    def plot_biplot(self, **kwargs: object):
        """Plot 2D/3D biplot."""
        from ama_tlbx.plotting.pca_plots import plot_biplot_plotly  # noqa: PLC0415

        return plot_biplot_plotly(self, **kwargs)


class PCAAnalyzer:
    """Analyzer for Principal Component Analysis (PCA).

    Provides methods for fitting PCA models, transforming data, and
    summarizing component loadings.

    Example:
        >>> from ama_tlbx.data import LifeExpectancyDataset
        >>> from ama_tlbx.plotting import plot_explained_variance
        >>> pca_result = (
        ...     LifeExpectancyDataset.from_csv()
        ...     .make_pca_analyzer(standardized=True, exclude_target=True)
        ...     .fit(n_components=None)
        ...     .result()
        ... )
        >>> pca_result.explained_variance.head()
        >>> fig = plot_explained_variance(pca_result)
        >>> fig = plot_biplot_plotly(pca_result, color=le_ds[LECol.Target], ...)
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

        Principal Component Analysis projects the data matrix **X** onto a new orthonormal
        basis of maximal variance. The columns of this basis are eigenvectors of Cov(**X**, **X**), ordered by descending eigenvalue; each successive component explains the largest remaining variance while staying orthogonal to the previous ones.
        See [scikit-learn PCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) for details.
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
            index=self._view.df.index,
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

    @property
    def model(self) -> PCA:
        """Return the fitted scikit-learn PCA model."""
        if self._pca_model is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
        return self._pca_model

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
        method: Literal["max", "l2"] = "l2",
    ) -> pd.Index:
        """Rank features by aggregated loading strength across leading components."""
        if self._pca_model is None:
            raise ValueError("PCA model not fitted. Call fit() first.")

        loadings = self.get_loading_vectors()
        if isinstance(loadings, pd.Series):
            raise TypeError("Expected DataFrame from get_loading_vectors()")

        pc_cols = [f"PC{i + 1}" for i in range(min(n_components, loadings.shape[1]))]

        method_key = method.lower()
        if method_key == "max":
            importance = loadings[pc_cols].abs().max(axis=1)
        elif method_key == "l2":
            importance = (loadings[pc_cols] ** 2).sum(axis=1).pow(0.5)
        else:
            raise ValueError("method must be one of {'max', 'l2'}")

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
        top_global = self.get_top_loading_features(n_components=loadings.shape[1])
        top_per_pc: dict[str, pd.Index] = {
            pc: loadings[pc].abs().sort_values(ascending=False).index for pc in loadings.columns
        }
        return PCAResult(
            scores=scores,
            loadings=loadings,
            explained_variance=explained,
            top_features_global=top_global,
            top_features_per_pc=top_per_pc,
        )
