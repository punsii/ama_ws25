"""PCA analysis for dimensionality reduction and feature interpretation."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class PCAAnalyzer:
    """Analyzer for Principal Component Analysis (PCA).

    This class provides methods for fitting PCA models, transforming data,
    analyzing loading vectors, and computing geometric relationships in PC space.
    """

    def __init__(self, data: pd.DataFrame):
        """Initialize the PCA analyzer.

        Args:
            data: Standardized DataFrame with numeric features
        """
        self.data = data
        self._pca_model: PCA | None = None
        self._feature_names: list[str] = []

    def fit(
        self,
        n_components: int | None = None,
        exclude_cols: list[str] | None = None,
    ) -> "PCAAnalyzer":
        """Fit PCA model on the data.

        Args:
            n_components: Number of components to keep (None = all)
            exclude_cols: Columns to exclude from PCA (e.g., ['life_expectancy'])

        Returns:
            Self for method chaining
        """
        features = self.data.copy()

        # Exclude specified columns
        exclude_cols = exclude_cols or []
        for col in exclude_cols:
            if col in features.columns:
                features = features.drop(columns=[col])

        self._feature_names = features.columns.tolist()
        self._pca_model = PCA(n_components=n_components)
        self._pca_model.fit(features)

        return self

    def transform(self, n_components: int | None = None) -> pd.DataFrame:
        """Transform data using fitted PCA model.

        Args:
            n_components: Number of components to return (None = all)

        Returns:
            DataFrame with PCA-transformed data
        """
        if self._pca_model is None:
            raise ValueError("PCA model not fitted. Call fit() first.")

        features = self.data[self._feature_names]
        transformed = self._pca_model.transform(features)

        if n_components is not None:
            transformed = transformed[:, :n_components]

        return pd.DataFrame(
            transformed,
            columns=[f"PC{i + 1}" for i in range(transformed.shape[1])],
            index=self.data.index,
        )

    def get_explained_variance(self) -> pd.DataFrame:
        """Get explained variance for each principal component.

        Returns:
            DataFrame with variance, explained ratio, and cumulative variance
        """
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
        """Get PCA loading vectors for specific component(s).

        Args:
            component: Component number (1-indexed) to get "PC1", "PC2", ...
                      If None, returns all loading vectors as DataFrame.

        Returns:
            Series with feature loadings for single component, or
            DataFrame with all components if component is None.

        Examples:
            >>> analyzer.fit()
            >>> pc1_loadings = analyzer.get_loading_vectors(1)
            >>> all_loadings = analyzer.get_loading_vectors()
        """
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

    def get_loading_magnitudes(
        self,
        component: int | str | None = None,
    ) -> pd.Series | float:
        """Get L2 norm (magnitude) of loading vector(s).

        Args:
            component: Component number (1-indexed) or "PC1", "PC2", etc.
                      If None, returns magnitudes for all components.

        Returns:
            Float for single component, or Series for all components.

        Examples:
            >>> analyzer.fit()
            >>> pc1_magnitude = analyzer.get_loading_magnitudes(1)
            >>> all_magnitudes = analyzer.get_loading_magnitudes()
        """
        if self._pca_model is None:
            raise ValueError("PCA model not fitted. Call fit() first.")

        loadings = self.get_loading_vectors(component=None)

        if isinstance(loadings, pd.Series):
            # Single component
            return float(np.linalg.norm(loadings.to_numpy()))

        if component is None:
            # Return magnitude for all components
            return pd.Series(
                np.linalg.norm(loadings.to_numpy(), axis=0),
                index=loadings.columns,
                name="magnitude",
            )

        # Return magnitude for specific component
        loading_vector = self.get_loading_vectors(component)
        if isinstance(loading_vector, pd.Series):
            return float(np.linalg.norm(loading_vector.to_numpy()))
        msg = f"Unexpected type for loading vector: {type(loading_vector)}"
        raise ValueError(msg)

    def get_top_loading_features(
        self,
        n_components: int = 3,
        method: str = "sum",
    ) -> pd.Index:
        """Get features with highest loadings across top N principal components.

        Args:
            n_components: Number of PCs to consider
            method: 'sum' for sum of absolute loadings, 'max' for max absolute loading

        Returns:
            Index of feature names sorted by importance
        """
        if self._pca_model is None:
            raise ValueError("PCA model not fitted. Call fit() first.")

        loadings = self.get_loading_vectors()
        if isinstance(loadings, pd.Series):
            raise TypeError("Expected DataFrame from get_loading_vectors()")

        pc_cols = [f"PC{i + 1}" for i in range(min(n_components, loadings.shape[1]))]

        # Calculate feature importance across components
        importance = loadings[pc_cols].abs().sum(axis=1) if method == "sum" else loadings[pc_cols].abs().max(axis=1)

        return importance.sort_values(ascending=False).index

    def compute_pca_dot_product(
        self,
        index1: str | int,
        index2: str | int,
        n_components: int | None = None,
    ) -> float:
        """Compute dot product between PCA coordinates of two data points.

        The dot product measures similarity in the PC space. Higher values
        indicate more similar data points in terms of their principal components.

        Args:
            index1: Row index for first data point
            index2: Row index for second data point
            n_components: Number of PCs to use (None = all fitted components)

        Returns:
            Dot product value

        Examples:
            >>> analyzer.fit()
            >>> similarity = analyzer.compute_pca_dot_product("USA", "Canada")
        """
        pca_scores = self.transform(n_components=n_components)

        try:
            vec1 = pca_scores.loc[index1].to_numpy()
            vec2 = pca_scores.loc[index2].to_numpy()
        except KeyError as e:
            msg = f"Index not found: {e}"
            raise ValueError(msg) from e

        return float(np.dot(vec1, vec2))

    def compute_pca_cross_product(
        self,
        index1: str | int,
        index2: str | int,
    ) -> np.ndarray:
        """Compute cross product between PCA coordinates (only for 3D).

        The cross product gives a vector perpendicular to both input vectors.
        Only works when using exactly 3 principal components.

        Args:
            index1: Row index for first data point
            index2: Row index for second data point

        Returns:
            3D numpy array representing the cross product vector

        Raises:
            ValueError: If PCA not fitted with exactly 3 components

        Examples:
            >>> analyzer.fit()
            >>> orthogonal_vec = analyzer.compute_pca_cross_product("USA", "Canada")
        """
        required_components = 3
        pca_scores = self.transform(n_components=required_components)

        if pca_scores.shape[1] != required_components:
            msg = (
                f"Cross product requires exactly {required_components} components, "
                f"got {pca_scores.shape[1]}. "
                f"Fit with n_components={required_components} or "
                f"transform with n_components={required_components}."
            )
            raise ValueError(msg)

        try:
            vec1 = pca_scores.loc[index1].to_numpy()
            vec2 = pca_scores.loc[index2].to_numpy()
        except KeyError as e:
            msg = f"Index not found: {e}"
            raise ValueError(msg) from e

        return np.cross(vec1, vec2)

    def compute_loading_dot_product(
        self,
        component1: int,
        component2: int,
    ) -> float:
        """Compute dot product between two loading vectors (should be 0 if orthogonal).

        PCA components are orthogonal, so their loading vectors should have
        dot product ≈ 0. This method can verify orthogonality.

        Args:
            component1: First component number (1-indexed) or "PC1", "PC2", etc.
            component2: Second component number (1-indexed) or "PC1", "PC2", etc.

        Returns:
            Dot product value (should be ~0 for different components)

        Examples:
            >>> analyzer.fit()
            >>> orthogonality = analyzer.compute_loading_dot_product(1, 2)
            >>> print(f"Orthogonality check: {orthogonality:.10f}")  # Should be ~0
        """
        vec1_series = self.get_loading_vectors(component1)
        vec2_series = self.get_loading_vectors(component2)

        if not isinstance(vec1_series, pd.Series) or not isinstance(vec2_series, pd.Series):
            raise TypeError("Expected Series from get_loading_vectors with single component")

        return float(np.dot(vec1_series.to_numpy(), vec2_series.to_numpy()))
