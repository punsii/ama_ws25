"""Tests for PCA dimensionality reduction analyzer."""

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from ama_tlbx.analysis.pca_dim_reduction import (
    FeatureGroup,
    GroupPCAResult,
    PCADimReductionAnalyzer,
    PCADimReductionResult,
)
from ama_tlbx.data.views import DatasetView


@pytest.fixture
def sample_view():
    """Create a sample dataset view for testing."""
    np.random.seed(42)
    n_samples = 100

    # Create correlated features
    # Group 1: highly correlated features (immunization-like)
    base1 = np.random.randn(n_samples)
    feat1a = base1 + np.random.randn(n_samples) * 0.1
    feat1b = base1 + np.random.randn(n_samples) * 0.1
    feat1c = base1 + np.random.randn(n_samples) * 0.1

    # Group 2: moderately correlated features (mortality-like)
    base2 = np.random.randn(n_samples)
    feat2a = base2 + np.random.randn(n_samples) * 0.3
    feat2b = base2 + np.random.randn(n_samples) * 0.3

    df = pd.DataFrame(
        {
            "feat1a": feat1a,
            "feat1b": feat1b,
            "feat1c": feat1c,
            "feat2a": feat2a,
            "feat2b": feat2b,
        },
    )

    return DatasetView(
        df=df,
        pretty_by_col={col: col.upper() for col in df.columns},
        numeric_cols=df.columns.tolist(),
        target_col=None,
        is_standardized=True,
    )


@pytest.fixture
def sample_groups():
    """Create sample feature groups."""
    return [
        FeatureGroup(name="Group1", features=["feat1a", "feat1b", "feat1c"]),
        FeatureGroup(name="Group2", features=["feat2a", "feat2b"]),
    ]


def test_feature_group_creation():
    """Test FeatureGroup dataclass."""
    group = FeatureGroup(name="Test", features=["a", "b", "c"])
    assert group.name == "Test"
    assert group.features == ["a", "b", "c"]


def test_analyzer_initialization(sample_view, sample_groups):
    """Test analyzer initialization."""
    analyzer = PCADimReductionAnalyzer(view=sample_view, feature_groups=sample_groups)
    assert analyzer._view == sample_view
    assert analyzer._feature_groups == sample_groups
    assert not analyzer._fitted


def test_analyzer_initialization_empty_groups(sample_view):
    """Test that empty groups raise ValueError."""
    with pytest.raises(ValueError, match="At least one feature group is required"):
        PCADimReductionAnalyzer(view=sample_view, feature_groups=[])


def test_analyzer_initialization_invalid_features(sample_view):
    """Test that invalid feature names raise ValueError."""
    invalid_groups = [FeatureGroup(name="Invalid", features=["nonexistent"])]
    with pytest.raises(ValueError, match="contains invalid features"):
        PCADimReductionAnalyzer(view=sample_view, feature_groups=invalid_groups)


def test_analyzer_fit(sample_view, sample_groups):
    """Test analyzer fitting."""
    analyzer = PCADimReductionAnalyzer(view=sample_view, feature_groups=sample_groups)
    result = analyzer.fit()

    assert result == analyzer  # Check method chaining
    assert analyzer._fitted
    assert len(analyzer._group_results) == len(sample_groups)


def test_result_before_fit(sample_view, sample_groups):
    """Test that calling result() before fit() raises ValueError."""
    analyzer = PCADimReductionAnalyzer(view=sample_view, feature_groups=sample_groups)
    with pytest.raises(ValueError, match="not fitted"):
        analyzer.result()


def test_group_pca_result_structure(sample_view, sample_groups):
    """Test GroupPCAResult structure."""
    analyzer = PCADimReductionAnalyzer(view=sample_view, feature_groups=sample_groups)
    analyzer.fit()

    for gr in analyzer._group_results:
        assert isinstance(gr, GroupPCAResult)
        assert isinstance(gr.group, FeatureGroup)
        assert isinstance(gr.pc_scores, pd.DataFrame)
        assert isinstance(gr.explained_variance, pd.DataFrame)
        assert isinstance(gr.pc1_scores, pd.Series)  # backward compatibility property
        assert isinstance(gr.explained_variance_pc1, float)  # backward compatibility property
        assert isinstance(gr.cumulative_variance_explained, float)
        assert isinstance(gr.loadings, pd.DataFrame)
        assert isinstance(gr.loadings_retained, pd.DataFrame)  # new property for retained PCs only
        assert isinstance(gr.n_features, int)
        assert isinstance(gr.n_components, int)
        assert isinstance(gr.min_var_explained, float)
        assert 0 <= gr.explained_variance_pc1 <= 1
        assert 0 < gr.min_var_explained <= 1


def test_pca_dim_reduction_result(sample_view, sample_groups):
    """Test complete PCADimReductionResult."""
    analyzer = PCADimReductionAnalyzer(view=sample_view, feature_groups=sample_groups)
    result = analyzer.fit().result()

    assert isinstance(result, PCADimReductionResult)
    assert len(result.group_results) == 2
    assert result.reduced_df.shape == (100, 2)  # 100 samples, 2 groups
    assert result.original_n_features == 5  # 3 + 2 features
    assert result.reduced_n_features == 2  # 2 groups
    assert result.compression_ratio == 2.5  # 5 / 2


def test_reduced_df_structure(sample_view, sample_groups):
    """Test that reduced_df has correct structure."""
    analyzer = PCADimReductionAnalyzer(view=sample_view, feature_groups=sample_groups)
    result = analyzer.fit().result()

    # Check columns are named correctly
    assert "Group1_PC1" in result.reduced_df.columns
    assert "Group2_PC1" in result.reduced_df.columns

    # Check no missing values
    assert not result.reduced_df.isna().any().any()

    # Check index matches original
    assert result.reduced_df.index.equals(sample_view.df.index)


def test_explained_variance_high_correlation(sample_view):
    """Test that highly correlated features have high PC1 variance."""
    # Group with very high correlation
    groups = [FeatureGroup(name="HighCorr", features=["feat1a", "feat1b", "feat1c"])]
    analyzer = PCADimReductionAnalyzer(view=sample_view, feature_groups=groups)
    result = analyzer.fit().result()

    # PC1 should explain >80% for highly correlated features
    assert result.group_results[0].explained_variance_pc1 > 0.80


def test_loadings_sum_of_squares(sample_view, sample_groups):
    """Test that loadings are normalized (sum of squares = 1) for retained PCs."""
    analyzer = PCADimReductionAnalyzer(view=sample_view, feature_groups=sample_groups)
    result = analyzer.fit().result()

    for gr in result.group_results:
        loadings_sq_sum = (gr.loadings_retained**2).sum()
        assert np.isclose(loadings_sq_sum, 1.0, atol=1e-6)


def test_compression_ratio_property(sample_view, sample_groups):
    """Test compression_ratio property."""
    analyzer = PCADimReductionAnalyzer(view=sample_view, feature_groups=sample_groups)
    result = analyzer.fit().result()

    expected = result.original_n_features / result.reduced_n_features
    assert result.compression_ratio == expected


def test_mean_explained_variance_property(sample_view, sample_groups):
    """Test mean_explained_variance property."""
    analyzer = PCADimReductionAnalyzer(view=sample_view, feature_groups=sample_groups)
    result = analyzer.fit().result()

    expected = sum(gr.explained_variance_pc1 for gr in result.group_results) / len(result.group_results)
    assert result.mean_explained_variance == expected


def test_pretty_names_preserved(sample_view, sample_groups):
    """Test that pretty names are preserved in result."""
    analyzer = PCADimReductionAnalyzer(view=sample_view, feature_groups=sample_groups)
    result = analyzer.fit().result()

    assert result.pretty_by_col == dict(sample_view.pretty_by_col)


def test_single_group(sample_view):
    """Test with a single feature group."""
    groups = [FeatureGroup(name="OnlyGroup", features=["feat1a", "feat1b"])]
    analyzer = PCADimReductionAnalyzer(view=sample_view, feature_groups=groups)
    result = analyzer.fit().result()

    assert len(result.group_results) == 1
    assert result.reduced_df.shape == (100, 1)
    assert result.compression_ratio == 2.0


def test_pc1_scores_match_pca_transform(sample_view, sample_groups):
    """Test that PC1 scores match direct PCA transformation."""
    analyzer = PCADimReductionAnalyzer(view=sample_view, feature_groups=sample_groups)
    result = analyzer.fit().result()

    # Manually verify for first group
    group1_features = sample_groups[0].features
    group1_data = sample_view.df[group1_features]

    pca = PCA(n_components=1)
    manual_pc1 = pca.fit_transform(group1_data).flatten()

    # PC1 scores should match (allow sign flip)
    result_pc1 = result.group_results[0].pc1_scores.to_numpy()
    correlation = np.corrcoef(manual_pc1, result_pc1)[0, 1]

    assert np.abs(correlation) > 0.99  # Should be nearly identical (sign may differ)


def test_dim_reduction_result_transform_reproduces_training_scores(sample_view, sample_groups):
    """transform() should reproduce stored reduced_df when applied to the training data."""
    analyzer = PCADimReductionAnalyzer(view=sample_view, feature_groups=sample_groups)
    result = analyzer.fit().result()

    transformed = result.transform(sample_view.df)

    assert list(transformed.columns) == list(result.reduced_df.columns)
    assert transformed.index.equals(result.reduced_df.index)
    assert np.allclose(transformed.to_numpy(), result.reduced_df.to_numpy())
