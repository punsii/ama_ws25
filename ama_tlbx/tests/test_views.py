"""Tests for DatasetView."""

import pandas as pd
import pytest

from ama_tlbx.data_handling.views import DatasetView


class TestDatasetView:
    """Test DatasetView functionality."""

    @pytest.fixture
    def sample_view(self) -> DatasetView:
        """Create a sample DatasetView for testing."""
        data = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
                "category": ["A", "B", "C"],
            },
        )
        pretty_by_col = {
            "feature1": "Feature One",
            "feature2": "Feature Two",
            "category": "Category",
        }
        numeric_cols = ["feature1", "feature2"]
        return DatasetView(
            data=data,
            pretty_by_col=pretty_by_col,
            numeric_cols=numeric_cols,
            target_col="feature1",
        )

    def test_view_creation(self, sample_view: DatasetView) -> None:
        """Test creating a DatasetView."""
        assert len(sample_view.data) == 3
        assert list(sample_view.data.columns) == ["feature1", "feature2", "category"]
        assert sample_view.target_col == "feature1"

    def test_view_is_frozen(self, sample_view: DatasetView) -> None:
        """Test that DatasetView is immutable."""
        with pytest.raises(AttributeError):
            sample_view.target_col = "feature2"  # type: ignore[misc]

    def test_features_property(self, sample_view: DatasetView) -> None:
        """Test features property returns numeric columns."""
        features = sample_view.features
        assert list(features.columns) == ["feature1", "feature2"]
        assert len(features) == 3

    def test_features_property_with_no_numeric_cols(self) -> None:
        """Test features property when numeric_cols is empty."""
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        view = DatasetView(
            data=data,
            pretty_by_col={"col1": "Column 1", "col2": "Column 2"},
            numeric_cols=[],
        )
        features = view.features
        # Should return all columns when numeric_cols is empty
        assert list(features.columns) == ["col1", "col2"]

    def test_pretty_by_col_mapping(self, sample_view: DatasetView) -> None:
        """Test pretty name mapping."""
        assert sample_view.pretty_by_col["feature1"] == "Feature One"
        assert sample_view.pretty_by_col["feature2"] == "Feature Two"

    def test_view_without_target(self) -> None:
        """Test creating a view without a target column."""
        data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        view = DatasetView(
            data=data,
            pretty_by_col={"x": "X", "y": "Y"},
            numeric_cols=["x", "y"],
            target_col=None,
        )
        assert view.target_col is None
        assert len(view.features) == 3
