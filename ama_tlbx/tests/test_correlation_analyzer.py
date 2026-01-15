"""Tests for CorrelationAnalyzer."""

import numpy as np
import pandas as pd
import pytest

from ama_tlbx.analysis.correlation_analyzer import CorrelationAnalyzer, CorrelationResult
from ama_tlbx.data.views import DatasetView


class TestCorrelationAnalyzer:
    """Test CorrelationAnalyzer functionality."""

    @pytest.fixture
    def sample_view(self) -> DatasetView:
        """Create a sample DatasetView for testing."""
        data = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [2.0, 4.0, 6.0, 8.0, 10.0],  # Perfect positive correlation
                "feature3": [5.0, 4.0, 3.0, 2.0, 1.0],  # Perfect negative correlation with feature1
                "target": [10.0, 15.0, 20.0, 25.0, 30.0],
            },
        )
        return DatasetView(
            df=data,
            pretty_by_col={
                "feature1": "Feature 1",
                "feature2": "Feature 2",
                "feature3": "Feature 3",
                "target": "Target",
            },
            numeric_cols=["feature1", "feature2", "feature3", "target"],
            target_col="target",
        )

    @pytest.fixture
    def view_without_target(self) -> DatasetView:
        """Create a DatasetView without a target column."""
        data = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0],
                "feature2": [2.0, 4.0, 6.0, 8.0],
            },
        )
        return DatasetView(
            df=data,
            pretty_by_col={"feature1": "Feature 1", "feature2": "Feature 2"},
            numeric_cols=["feature1", "feature2"],
            target_col=None,
        )

    def test_analyzer_creation(self, sample_view: DatasetView) -> None:
        """Test creating a CorrelationAnalyzer."""
        analyzer = CorrelationAnalyzer(sample_view)
        assert isinstance(analyzer, CorrelationAnalyzer)

    def test_get_correlation_matrix(self, sample_view: DatasetView) -> None:
        """Test getting correlation matrix."""
        analyzer = CorrelationAnalyzer(sample_view)
        corr_matrix = analyzer.get_correlation_matrix()

        # Check shape
        assert corr_matrix.shape == (4, 4)

        # Check diagonal is 1.0
        assert np.allclose(np.diag(corr_matrix), 1.0)

        # Check symmetry
        assert np.allclose(corr_matrix, corr_matrix.T)

        # Check perfect correlations
        assert np.isclose(corr_matrix.loc["feature1", "feature2"], 1.0)
        assert np.isclose(corr_matrix.loc["feature1", "feature3"], -1.0)

    def test_get_top_correlated_pairs(self, sample_view: DatasetView) -> None:
        """Test getting top correlated pairs."""
        analyzer = CorrelationAnalyzer(sample_view)
        top_pairs = analyzer.get_top_correlated_pairs(n=3)

        # Should have 3 rows
        assert len(top_pairs) == 3

        # Should have required columns (actual column names from implementation)
        assert "feature_a" in top_pairs.columns
        assert "feature_b" in top_pairs.columns
        assert "correlation" in top_pairs.columns
        assert "abs_correlation" in top_pairs.columns

        # First pair should have highest absolute correlation
        assert np.isclose(top_pairs.iloc[0]["abs_correlation"], 1.0)

    def test_get_top_correlated_pairs_default_n(self, sample_view: DatasetView) -> None:
        """Test get_top_correlated_pairs with default n."""
        analyzer = CorrelationAnalyzer(sample_view)
        top_pairs = analyzer.get_top_correlated_pairs()

        # Default n should be 10, but we only have 6 unique pairs
        assert len(top_pairs) <= 10

    def test_get_target_correlations(self, sample_view: DatasetView) -> None:
        """Test getting correlations with target."""
        analyzer = CorrelationAnalyzer(sample_view)
        target_corrs = analyzer.get_target_correlations()

        # Should have 3 features (excluding target itself)
        assert len(target_corrs) == 3

        # Should have required columns (actual column names)
        assert "correlation" in target_corrs.columns
        assert "feature" in target_corrs.columns

    def test_get_target_correlations_no_target(self, view_without_target: DatasetView) -> None:
        """Test get_target_correlations raises error without target."""
        analyzer = CorrelationAnalyzer(view_without_target)

        with pytest.raises(ValueError, match=r"Dataset view has no target column"):
            analyzer.get_target_correlations()

    def test_compute_returns_result(self, sample_view: DatasetView) -> None:
        """Test that compute returns CorrelationResult."""
        analyzer = CorrelationAnalyzer(sample_view)
        fitted = analyzer.fit()
        result = analyzer.result()

        assert fitted is analyzer
        assert isinstance(result, CorrelationResult)
        assert result.matrix is not None
        assert result.feature_pairs is not None
        assert result.target_correlations is not None

    def test_compute_without_target(self, view_without_target: DatasetView) -> None:
        """Test compute works without target column."""
        analyzer = CorrelationAnalyzer(view_without_target)
        analyzer.fit()
        result = analyzer.result()

        assert isinstance(result, CorrelationResult)
        assert result.matrix is not None
        assert result.feature_pairs is not None
        assert result.target_correlations is None  # No target

    def test_correlation_result_dataclass(self) -> None:
        """Test CorrelationResult dataclass."""
        corr_matrix = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]])
        top_pairs = pd.DataFrame({"feature_a": ["A"], "feature_b": ["B"], "correlation": [0.5]})
        target_corrs = pd.DataFrame({"feature": ["A"], "correlation": [0.8]})
        pretty_by_col = {"A": "Feature A", "B": "Feature B"}

        result = CorrelationResult(
            matrix=corr_matrix,
            pretty_by_col=pretty_by_col,
            feature_pairs=top_pairs,
            target_correlations=target_corrs,
        )

        assert result.matrix is not None
        assert result.feature_pairs is not None
        assert result.target_correlations is not None

    def test_correlation_with_missing_values(self) -> None:
        """Test correlation handles missing values."""
        data = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, np.nan, 4.0, 5.0],
                "feature2": [2.0, 4.0, 6.0, 8.0, 10.0],
                "target": [10.0, 15.0, 20.0, 25.0, 30.0],
            },
        )
        view = DatasetView(
            df=data,
            pretty_by_col={"feature1": "Feature 1", "feature2": "Feature 2", "target": "Target"},
            numeric_cols=["feature1", "feature2", "target"],
            target_col="target",
        )

        analyzer = CorrelationAnalyzer(view)
        analyzer.fit()
        result = analyzer.result()

        # Should still compute correlations (using pairwise complete observations)
        assert result.matrix is not None
        assert not result.matrix.isna().all().all()
