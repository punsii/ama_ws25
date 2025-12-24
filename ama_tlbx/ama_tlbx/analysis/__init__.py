"""Analysis modules for dataset processing and statistical methods."""

from .column_concat import ColumnConcatenator
from .correlation_analyzer import CorrelationAnalyzer, CorrelationResult
from .hierachical_clustering import suggest_groups_from_correlation
from .model_registry import ModelEntry, ModelRegistry
from .ols_helper import RegressionResult, fit_ols_design, fit_ols_formula
from .outlier_detector import (
    IQROutlierDetector,
    IsolationForestOutlierDetector,
    OutlierDetectionResult,
    ZScoreOutlierDetector,
)
from .pca_analyzer import PCAAnalyzer, PCAResult
from .pca_dim_reduction import (
    FeatureGroup,
    GroupPCAResult,
    PCADimReductionAnalyzer,
    PCADimReductionResult,
)


__all__ = [
    "ColumnConcatenator",
    "CorrelationAnalyzer",
    "CorrelationResult",
    "FeatureGroup",
    "GroupPCAResult",
    "IQROutlierDetector",
    "IsolationForestOutlierDetector",
    "ModelEntry",
    "ModelRegistry",
    "OutlierDetectionResult",
    "PCAAnalyzer",
    "PCADimReductionAnalyzer",
    "PCADimReductionResult",
    "PCAResult",
    "RegressionResult",
    "ZScoreOutlierDetector",
    "fit_ols_design",
    "fit_ols_formula",
    "suggest_groups_from_correlation",
]
