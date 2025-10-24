"""Tests for outlier detection strategies."""

import numpy as np
import pandas as pd
import pytest

from ama_tlbx.analysis.outlier_detector import (
    IQROutlierDetector,
    IsolationForestOutlierDetector,
    ZScoreOutlierDetector,
)


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


class TestIQROutlierDetector:
    """Test IQR-based outlier detection."""

    def test_iqr_detector_creation(self) -> None:
        """Test creating an IQR detector with custom threshold."""
        detector = IQROutlierDetector(threshold=1.5)
        assert detector.threshold == 1.5

    def test_iqr_detector_default_threshold(self) -> None:
        """Test default threshold is 1.5."""
        detector = IQROutlierDetector()
        assert detector.threshold == 1.5

    def test_iqr_detect_outliers(self, sample_data: pd.DataFrame) -> None:
        """Test IQR outlier detection."""
        detector = IQROutlierDetector(threshold=1.5)
        outliers = detector.detect(sample_data)

        assert isinstance(outliers, pd.DataFrame)
        assert outliers.shape == sample_data.shape
        assert outliers.dtypes.all() == bool or all(outliers.dtypes == bool)
        # Should detect some outliers
        assert outliers.any().any()

    def test_iqr_detect_single_column(self, sample_data: pd.DataFrame) -> None:
        """Test IQR detection on a single column."""
        detector = IQROutlierDetector(threshold=1.5)
        outliers = detector.detect(sample_data, columns=["feature1"])

        assert list(outliers.columns) == ["feature1"]
        assert len(outliers) == len(sample_data)

    def test_iqr_stricter_threshold(self, sample_data: pd.DataFrame) -> None:
        """Test that stricter threshold detects fewer outliers."""
        loose_detector = IQROutlierDetector(threshold=3.0)
        strict_detector = IQROutlierDetector(threshold=1.0)

        loose_outliers = loose_detector.detect(sample_data)
        strict_outliers = strict_detector.detect(sample_data)

        # Stricter threshold should detect more outliers
        assert strict_outliers.sum().sum() >= loose_outliers.sum().sum()


class TestZScoreOutlierDetector:
    """Test Z-score based outlier detection."""

    def test_zscore_detector_creation(self) -> None:
        """Test creating a Z-score detector."""
        detector = ZScoreOutlierDetector(threshold=3.0, ddof=1)
        assert detector.threshold == 3.0
        assert detector.ddof == 1

    def test_zscore_default_parameters(self) -> None:
        """Test default parameters."""
        detector = ZScoreOutlierDetector()
        assert detector.threshold == 3.0
        assert detector.ddof == 0

    def test_zscore_detect_outliers(self, sample_data: pd.DataFrame) -> None:
        """Test Z-score outlier detection."""
        detector = ZScoreOutlierDetector(threshold=2.5)
        outliers = detector.detect(sample_data)

        assert isinstance(outliers, pd.DataFrame)
        assert outliers.shape == sample_data.shape
        assert outliers.any().any()

    def test_zscore_detect_specific_columns(self, sample_data: pd.DataFrame) -> None:
        """Test Z-score detection on specific columns."""
        detector = ZScoreOutlierDetector(threshold=2.0)
        outliers = detector.detect(sample_data, columns=["feature2"])

        assert list(outliers.columns) == ["feature2"]

    def test_zscore_stricter_threshold(self, sample_data: pd.DataFrame) -> None:
        """Test that lower threshold detects more outliers."""
        loose_detector = ZScoreOutlierDetector(threshold=4.0)
        strict_detector = ZScoreOutlierDetector(threshold=2.0)

        loose_outliers = loose_detector.detect(sample_data)
        strict_outliers = strict_detector.detect(sample_data)

        assert strict_outliers.sum().sum() >= loose_outliers.sum().sum()


class TestIsolationForestOutlierDetector:
    """Test Isolation Forest outlier detection."""

    def test_isolation_forest_creation(self) -> None:
        """Test creating an Isolation Forest detector."""
        detector = IsolationForestOutlierDetector(
            contamination=0.1,
            random_state=42,
            n_estimators=50,
        )
        assert detector.contamination == 0.1
        assert detector.random_state == 42
        assert detector.n_estimators == 50

    def test_isolation_forest_default_parameters(self) -> None:
        """Test default parameters."""
        detector = IsolationForestOutlierDetector()
        assert detector.contamination == "auto"
        assert detector.random_state is None
        assert detector.n_estimators == 100

    def test_isolation_forest_detect_outliers(self, sample_data: pd.DataFrame) -> None:
        """Test Isolation Forest outlier detection."""
        detector = IsolationForestOutlierDetector(
            contamination=0.05,
            random_state=42,
        )
        outliers = detector.detect(sample_data)

        assert isinstance(outliers, pd.DataFrame)
        assert outliers.shape == sample_data.shape
        # Should detect some outliers
        assert outliers.any().any()

    def test_isolation_forest_reproducible(self, sample_data: pd.DataFrame) -> None:
        """Test that results are reproducible with random_state."""
        detector1 = IsolationForestOutlierDetector(random_state=42)
        detector2 = IsolationForestOutlierDetector(random_state=42)

        outliers1 = detector1.detect(sample_data)
        outliers2 = detector2.detect(sample_data)

        pd.testing.assert_frame_equal(outliers1, outliers2)

    def test_isolation_forest_single_column(self, sample_data: pd.DataFrame) -> None:
        """Test Isolation Forest on a single column."""
        detector = IsolationForestOutlierDetector(random_state=42)
        outliers = detector.detect(sample_data, columns=["feature1"])

        assert list(outliers.columns) == ["feature1"]


class TestOutlierDetectorComparison:
    """Test comparing different outlier detection methods."""

    def test_all_detectors_work_on_same_data(self, sample_data: pd.DataFrame) -> None:
        """Test that all detectors can process the same data."""
        iqr = IQROutlierDetector()
        zscore = ZScoreOutlierDetector()
        iforest = IsolationForestOutlierDetector(random_state=42)

        iqr_result = iqr.detect(sample_data)
        zscore_result = zscore.detect(sample_data)
        iforest_result = iforest.detect(sample_data)

        # All should return DataFrames of the same shape
        assert iqr_result.shape == zscore_result.shape == iforest_result.shape
        assert iqr_result.shape == sample_data.shape

    def test_detectors_return_boolean_masks(self, sample_data: pd.DataFrame) -> None:
        """Test that all detectors return boolean DataFrames."""
        detectors = [
            IQROutlierDetector(),
            ZScoreOutlierDetector(),
            IsolationForestOutlierDetector(random_state=42),
        ]

        for detector in detectors:
            result = detector.detect(sample_data)
            assert result.dtypes.all() == bool or all(result.dtypes == bool)
