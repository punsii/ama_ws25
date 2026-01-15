"""Smoke tests for plotting utilities using real data results."""

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from ama_tlbx.analysis import pca_dim_reduction
from ama_tlbx.data import LECol
from ama_tlbx.plotting.correlation_plots import (
    plot_correlation_heatmap,
    plot_target_correlations,
    plot_top_correlated_pairs,
)
from ama_tlbx.plotting.dataset_plots import plot_standardization_comparison
from ama_tlbx.plotting.pca_dim_reduction_plots import (
    plot_group_compression,
    plot_group_loadings,
    plot_group_variance_summary,
)
from ama_tlbx.plotting.pca_plots import plot_biplot_plotly, plot_explained_variance, plot_loadings_heatmap


def test_correlation_plots_real_data(life_expectancy_dataset) -> None:
    """Correlation plots should render without errors on the real dataset."""
    corr_result = (
        life_expectancy_dataset.make_correlation_analyzer(standardized=True, include_target=True).fit().result()
    )
    heat_fig = plot_correlation_heatmap(corr_result, figsize=(6, 6))
    pos_fig, neg_fig = plot_top_correlated_pairs(corr_result, n=5, threshold=0.5, figsize=(6, 4))
    targ_pos, targ_neg = plot_target_correlations(corr_result, n=5, figsize=(6, 4))

    for fig in (heat_fig, pos_fig, neg_fig, targ_pos, targ_neg):
        assert fig.axes  # at least one axis exists
        plt.close(fig)


def test_dataset_standardization_plot(life_expectancy_dataset) -> None:
    """Standardization comparison should produce two boxplots (raw vs scaled)."""
    fig = plot_standardization_comparison(life_expectancy_dataset, figsize=(8, 6))
    assert len(fig.axes) == 2
    plt.close(fig)


def test_pca_plots_with_real_data(life_expectancy_dataset) -> None:
    """PCA plot helpers should work off the real PCA result."""
    pca_result = life_expectancy_dataset.make_pca_analyzer(standardized=True, exclude_target=True).fit().result()

    var_fig = plot_explained_variance(pca_result, figsize=(6, 4))
    load_fig = plot_loadings_heatmap(pca_result, n_components=2, top_n_features=6, figsize=(6, 4))
    biplot = plot_biplot_plotly(pca_result, dims=2, top_features=5)

    assert isinstance(biplot, go.Figure)
    plt.close(var_fig)
    plt.close(load_fig)


def test_pca_dim_reduction_plots_real_data(life_expectancy_dataset) -> None:
    """Dimensionality-reduction plots should visualize grouped PCs."""
    groups = [
        pca_dim_reduction.FeatureGroup("immunization", [LECol.POLIO, LECol.HEPATITIS_B, LECol.DIPHTHERIA]),
        pca_dim_reduction.FeatureGroup("mortality", [LECol.INFANT_DEATHS, LECol.UNDER_FIVE_DEATHS]),
    ]
    dimred_result = (
        life_expectancy_dataset.make_pca_dim_reduction_analyzer(groups, min_var_explained=0.8).fit().result()
    )

    var_fig = plot_group_variance_summary(dimred_result, figsize=(6, 4))
    comp_fig = plot_group_compression(dimred_result, figsize=(6, 4))
    load_fig = plot_group_loadings(dimred_result, figsize=(10, 3))

    for fig in (var_fig, comp_fig, load_fig):
        assert fig.axes
        plt.close(fig)
