"""Tests for PCAAnalyzer."""

import numpy as np
import pandas as pd
import pytest

from ama_tlbx.analysis.pca_analyzer import PCAAnalyzer, PCAResult
from ama_tlbx.data.views import DatasetView


class TestPCAAnalyzer:
    """Test PCAAnalyzer functionality."""

    @pytest.fixture
    def sample_view(self) -> DatasetView:
        """Create a sample DatasetView for testing."""
        # Create data with known variance structure
        np.random.seed(42)
        n_samples = 100
        # First component has high variance
        comp1 = np.random.normal(0, 3, n_samples)
        # Second component has medium variance
        comp2 = np.random.normal(0, 1, n_samples)
        # Third component has low variance
        comp3 = np.random.normal(0, 0.5, n_samples)

        data = pd.DataFrame(
            {
                "feature1": comp1 + 0.5 * comp2,
                "feature2": comp1 - 0.5 * comp2,
                "feature3": comp2 + 0.3 * comp3,
                "feature4": comp3,
            },
        )

        return DatasetView(
            data=data,
            pretty_by_col={
                "feature1": "Feature 1",
                "feature2": "Feature 2",
                "feature3": "Feature 3",
                "feature4": "Feature 4",
            },
            numeric_cols=["feature1", "feature2", "feature3", "feature4"],
            target_col=None,
        )

    def test_analyzer_creation(self, sample_view: DatasetView) -> None:
        """Test creating a PCAAnalyzer."""
        analyzer = PCAAnalyzer(sample_view)
        assert isinstance(analyzer, PCAAnalyzer)

    def test_fit_computes_pca(self, sample_view: DatasetView) -> None:
        """Test that fit computes PCA."""
        analyzer = PCAAnalyzer(sample_view)
        analyzer.fit(n_components=2)

        # Should have PCA model after fitting
        assert analyzer._pca_model is not None

    def test_fit_returns_self(self, sample_view: DatasetView) -> None:
        """Test that fit returns self for chaining."""
        analyzer = PCAAnalyzer(sample_view)
        result = analyzer.fit(n_components=2)

        assert result is analyzer

    def test_transform_before_fit_raises_error(self, sample_view: DatasetView) -> None:
        """Test that transform before fit raises error."""
        analyzer = PCAAnalyzer(sample_view)

        with pytest.raises(ValueError, match=r"PCA model not fitted"):
            analyzer.transform()

    def test_transform_reduces_dimensions(self, sample_view: DatasetView) -> None:
        """Test that transform reduces dimensions."""
        analyzer = PCAAnalyzer(sample_view)
        analyzer.fit(n_components=2)
        transformed = analyzer.transform()

        # Should have 2 components
        assert transformed.shape == (100, 2)
        assert list(transformed.columns) == ["PC1", "PC2"]

    def test_get_explained_variance(self, sample_view: DatasetView) -> None:
        """Test getting explained variance."""
        analyzer = PCAAnalyzer(sample_view)
        analyzer.fit(n_components=3)
        variance_df = analyzer.get_explained_variance()

        # Should have 3 rows
        assert len(variance_df) == 3

        # Should have required columns (actual column names from implementation)
        assert "PC" in variance_df.columns
        assert "explained_ratio" in variance_df.columns
        assert "cumulative_ratio" in variance_df.columns

        # Variance should be decreasing
        assert (
            variance_df["explained_ratio"].iloc[:-1].to_numpy() >= variance_df["explained_ratio"].iloc[1:].to_numpy()
        ).all()

        # Cumulative should be increasing
        assert (
            variance_df["cumulative_ratio"].iloc[:-1].to_numpy() <= variance_df["cumulative_ratio"].iloc[1:].to_numpy()
        ).all()

    def test_get_loading_vectors(self, sample_view: DatasetView) -> None:
        """Test getting loading vectors."""
        analyzer = PCAAnalyzer(sample_view)
        analyzer.fit(n_components=2)
        loadings = analyzer.get_loading_vectors()

        # Should have 4 rows (features) and 2 columns (components)
        assert loadings.shape == (4, 2)
        assert list(loadings.columns) == ["PC1", "PC2"]

    def test_get_loading_vectors_specific_component(self, sample_view: DatasetView) -> None:
        """Test getting loading vectors for specific component."""
        analyzer = PCAAnalyzer(sample_view)
        analyzer.fit(n_components=3)
        loadings = analyzer.get_loading_vectors(component=1)

        # Should be a Series for single component
        assert isinstance(loadings, pd.Series)
        assert loadings.name == "PC1"

    def test_get_top_loading_features_sum(self, sample_view: DatasetView) -> None:
        """Test getting top loading features using sum method."""
        analyzer = PCAAnalyzer(sample_view)
        analyzer.fit(n_components=2)
        top_features = analyzer.get_top_loading_features(n_components=2, method="sum")

        # Returns pd.Index of feature names
        assert isinstance(top_features, pd.Index)
        assert len(top_features) == 4  # All features returned, sorted

    def test_get_top_loading_features_max(self, sample_view: DatasetView) -> None:
        """Test getting top loading features using max method."""
        analyzer = PCAAnalyzer(sample_view)
        analyzer.fit(n_components=2)
        top_features = analyzer.get_top_loading_features(n_components=2, method="max")

        assert isinstance(top_features, pd.Index)

    def test_get_top_loading_features_l2(self, sample_view: DatasetView) -> None:
        """Test getting top loading features using l2 method."""
        analyzer = PCAAnalyzer(sample_view)
        analyzer.fit(n_components=2)
        top_features = analyzer.get_top_loading_features(n_components=2, method="l2")

        assert isinstance(top_features, pd.Index)

    def test_result_returns_pca_result(self, sample_view: DatasetView) -> None:
        """Test that result returns PCAResult."""
        analyzer = PCAAnalyzer(sample_view)
        analyzer.fit(n_components=2)
        result = analyzer.result()

        assert isinstance(result, PCAResult)
        assert result.scores is not None
        assert result.explained_variance is not None
        assert result.loadings is not None

    def test_result_before_fit_raises_error(self, sample_view: DatasetView) -> None:
        """Test that result before fit raises error."""
        analyzer = PCAAnalyzer(sample_view)

        with pytest.raises(ValueError, match=r"PCA model not fitted"):
            analyzer.result()

    def test_n_components_none(self, sample_view: DatasetView) -> None:
        """Test n_components=None uses all components."""
        analyzer = PCAAnalyzer(sample_view)
        analyzer.fit(n_components=None)
        transformed = analyzer.transform()

        # Should have 4 components (all features)
        assert transformed.shape[1] == 4

    def test_n_components_int(self, sample_view: DatasetView) -> None:
        """Test n_components as integer."""
        analyzer = PCAAnalyzer(sample_view)
        analyzer.fit(n_components=2)

        assert analyzer._pca_model.n_components_ == 2

    def test_pca_result_dataclass(self) -> None:
        """Test PCAResult dataclass."""
        scores = pd.DataFrame({"PC1": [1, 2], "PC2": [3, 4]})
        variance = pd.DataFrame({"PC": ["PC1"], "explained_ratio": [0.8]})
        loadings = pd.DataFrame({"PC1": [0.7, 0.7]})

        result = PCAResult(
            scores=scores,
            explained_variance=variance,
            loadings=loadings,
        )

        assert result.scores is not None
        assert result.explained_variance is not None
        assert result.loadings is not None

    def test_standardization_impact(self, sample_view: DatasetView) -> None:
        """Test that data is standardized before PCA."""
        analyzer = PCAAnalyzer(sample_view)
        analyzer.fit(n_components=2)

        # PCA should work well with standardized data
        variance = analyzer.get_explained_variance()

        # First component should explain most variance
        assert variance.iloc[0]["explained_ratio"] > 0.3

    def test_component_ordering(self, sample_view: DatasetView) -> None:
        """Test that components are ordered by explained variance."""
        analyzer = PCAAnalyzer(sample_view)
        analyzer.fit(n_components=3)
        variance = analyzer.get_explained_variance()

        # Check that variance is sorted in descending order
        variances = variance["explained_ratio"].to_numpy()
        assert np.all(variances[:-1] >= variances[1:])

    def test_fit_transform_chain(self, sample_view: DatasetView) -> None:
        """Test chaining fit and transform."""
        analyzer = PCAAnalyzer(sample_view)
        transformed = analyzer.fit(n_components=2).transform()

        assert transformed.shape == (100, 2)

    def test_multiple_transforms(self, sample_view: DatasetView) -> None:
        """Test that transform can be called multiple times."""
        analyzer = PCAAnalyzer(sample_view)
        analyzer.fit(n_components=2)

        transformed1 = analyzer.transform()
        transformed2 = analyzer.transform()

        # Should produce identical results
        pd.testing.assert_frame_equal(transformed1, transformed2)
