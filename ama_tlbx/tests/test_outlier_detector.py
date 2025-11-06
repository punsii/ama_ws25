"""Tests for outlier detection strategies."""

import numpy as np
import pandas as pd
import pytest

from ama_tlbx.analysis.outlier_detector import (
    IQROutlierDetector,
    IsolationForestOutlierDetector,
    OutlierDetectionResult,
    ZScoreOutlierDetector,
)
from ama_tlbx.data.views import DatasetView


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data with known outliers."""
    np.random.seed(42)
    normal_data = np.random.normal(loc=50, scale=10, size=95)
    outliers = np.array([0, 100, 150, 200, -50])  # Clear outliers
    data = np.concatenate([normal_data, outliers])
    return pd.DataFrame(
        {
            "feature1": data,
            "feature2": np.random.normal(loc=20, scale=5, size=100),
        },
    )


@pytest.fixture
def sample_view(sample_data: pd.DataFrame) -> DatasetView:
    """Create a DatasetView from sample data."""
    return DatasetView(
        df=sample_data,
        pretty_by_col={"feature1": "Feature 1", "feature2": "Feature 2"},
        numeric_cols=["feature1", "feature2"],
        target_col=None,
    )


class TestIQROutlierDetector:
    """Test IQR-based outlier detection."""

    def test_iqr_detector_creation(self, sample_view: DatasetView) -> None:
        """Test creating an IQR detector with custom threshold."""
        detector = IQROutlierDetector(sample_view, threshold=1.5)
        assert detector.threshold == 1.5

    def test_iqr_detector_default_threshold(self, sample_view: DatasetView) -> None:
        """Test default threshold is 1.5."""
        detector = IQROutlierDetector(sample_view)
        assert detector.threshold == 1.5

    def test_iqr_detect_outliers(self, sample_view: DatasetView) -> None:
        """Test IQR outlier detection."""
        detector = IQROutlierDetector(sample_view, threshold=1.5)
        result = detector.fit().result()

        assert isinstance(result, OutlierDetectionResult)
        assert isinstance(result.outlier_mask, pd.DataFrame)
        assert result.outlier_mask.shape == sample_view.df.shape
        assert result.outlier_mask.dtypes.all() == bool or all(result.outlier_mask.dtypes == bool)
        # Should detect some outliers
        assert result.total_outliers > 0

    def test_iqr_result_statistics(self, sample_view: DatasetView) -> None:
        """Test that result contains proper statistics."""
        detector = IQROutlierDetector(sample_view, threshold=1.5)
        result = detector.fit().result()

        assert isinstance(result.n_outliers_per_column, pd.Series)
        assert isinstance(result.n_outliers_per_row, pd.Series)
        assert isinstance(result.total_outliers, int)
        assert isinstance(result.outlier_percentage, float)
        assert 0 <= result.outlier_percentage <= 100

    def test_iqr_stricter_threshold(self, sample_view: DatasetView) -> None:
        """Test that stricter threshold detects more outliers."""
        loose_detector = IQROutlierDetector(sample_view, threshold=3.0)
        strict_detector = IQROutlierDetector(sample_view, threshold=1.0)

        loose_result = loose_detector.fit().result()
        strict_result = strict_detector.fit().result()

        # Stricter threshold should detect more outliers
        assert strict_result.total_outliers >= loose_result.total_outliers


class TestZScoreOutlierDetector:
    """Test Z-score based outlier detection."""

    def test_zscore_detector_creation(self, sample_view: DatasetView) -> None:
        """Test creating a Z-score detector."""
        detector = ZScoreOutlierDetector(sample_view, threshold=3.0)
        assert detector.threshold == 3.0

    def test_zscore_default_parameters(self, sample_view: DatasetView) -> None:
        """Test default parameters."""
        detector = ZScoreOutlierDetector(sample_view)
        assert detector.threshold == 3.0

    def test_zscore_detect_outliers(self, sample_view: DatasetView) -> None:
        """Test Z-score outlier detection."""
        detector = ZScoreOutlierDetector(sample_view, threshold=2.5)
        result = detector.fit().result()

        assert isinstance(result, OutlierDetectionResult)
        assert isinstance(result.outlier_mask, pd.DataFrame)
        assert result.outlier_mask.shape == sample_view.df.shape
        assert result.total_outliers > 0

    def test_zscore_result_contains_statistics(self, sample_view: DatasetView) -> None:
        """Test Z-score result statistics."""
        detector = ZScoreOutlierDetector(sample_view, threshold=2.0)
        result = detector.fit().result()

        assert result.total_outliers > 0
        assert len(result.column_names) == 2

    def test_zscore_stricter_threshold(self, sample_view: DatasetView) -> None:
        """Test that lower threshold detects more outliers."""
        loose_detector = ZScoreOutlierDetector(sample_view, threshold=4.0)
        strict_detector = ZScoreOutlierDetector(sample_view, threshold=2.0)

        loose_result = loose_detector.fit().result()
        strict_result = strict_detector.fit().result()

        assert strict_result.total_outliers >= loose_result.total_outliers


class TestIsolationForestOutlierDetector:
    """Test Isolation Forest outlier detection."""

    def test_isolation_forest_creation(self, sample_view: DatasetView) -> None:
        """Test creating an Isolation Forest detector."""
        detector = IsolationForestOutlierDetector(
            sample_view,
            contamination=0.1,
            random_state=42,
            n_estimators=50,
        )
        assert detector.contamination == 0.1
        assert detector.random_state == 42
        assert detector.n_estimators == 50

    def test_isolation_forest_default_parameters(self, sample_view: DatasetView) -> None:
        """Test default parameters."""
        detector = IsolationForestOutlierDetector(sample_view)
        assert detector.contamination == "auto"
        assert detector.random_state is None
        assert detector.n_estimators == 100

    def test_isolation_forest_detect_outliers(self, sample_view: DatasetView) -> None:
        """Test Isolation Forest outlier detection."""
        detector = IsolationForestOutlierDetector(
            sample_view,
            contamination=0.05,
            random_state=42,
        )
        result = detector.fit().result()

        assert isinstance(result, OutlierDetectionResult)
        assert isinstance(result.outlier_mask, pd.DataFrame)
        assert result.outlier_mask.shape == sample_view.df.shape
        # Should detect some outliers
        assert result.total_outliers > 0

    def test_isolation_forest_reproducible(self, sample_view: DatasetView) -> None:
        """Test that results are reproducible with random_state."""
        detector1 = IsolationForestOutlierDetector(sample_view, random_state=42)
        detector2 = IsolationForestOutlierDetector(sample_view, random_state=42)

        result1 = detector1.fit().result()
        result2 = detector2.fit().result()

        pd.testing.assert_frame_equal(result1.outlier_mask, result2.outlier_mask)

    def test_isolation_forest_result_statistics(self, sample_view: DatasetView) -> None:
        """Test Isolation Forest result statistics."""
        detector = IsolationForestOutlierDetector(sample_view, random_state=42)
        result = detector.fit().result()

        assert result.total_outliers >= 0
        assert len(result.column_names) == 2


class TestOutlierDetectorComparison:
    """Test comparing different outlier detection methods."""

    def test_all_detectors_work_on_same_data(self, sample_view: DatasetView) -> None:
        """Test that all detectors can process the same data."""
        iqr = IQROutlierDetector(sample_view)
        zscore = ZScoreOutlierDetector(sample_view)
        iforest = IsolationForestOutlierDetector(sample_view, random_state=42)

        iqr_result = iqr.fit().result()
        zscore_result = zscore.fit().result()
        iforest_result = iforest.fit().result()

        # All should return OutlierDetectionResult with same shape masks
        assert iqr_result.outlier_mask.shape == zscore_result.outlier_mask.shape == iforest_result.outlier_mask.shape
        assert iqr_result.outlier_mask.shape == sample_view.df.shape

    def test_detectors_return_boolean_masks(self, sample_view: DatasetView) -> None:
        """Test that all detectors return boolean DataFrames."""
        detectors = [
            IQROutlierDetector(sample_view),
            ZScoreOutlierDetector(sample_view),
            IsolationForestOutlierDetector(sample_view, random_state=42),
        ]

        for detector in detectors:
            result = detector.fit().result()
            assert result.outlier_mask.dtypes.all() == bool or all(result.outlier_mask.dtypes == bool)
