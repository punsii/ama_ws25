"""Real-data tests for column concatenation and correlation-driven grouping."""

import pandas as pd

from ama_tlbx.analysis.column_concat import ColumnConcatenator
from ama_tlbx.analysis.hierachical_clustering import suggest_groups_from_correlation
from ama_tlbx.data import LECol


def test_column_concatenator_creates_weighted_feature(life_expectancy_dataset) -> None:
    """Concatenate immunization coverage columns into a weighted PCA feature."""
    cols = [LECol.HEPATITIS_B, LECol.POLIO, LECol.DIPHTHERIA]
    concat = ColumnConcatenator(life_expectancy_dataset)

    new_ds = concat.concatenate(columns=cols, new_column_name="immunization_pca")

    assert new_ds is not life_expectancy_dataset
    assert "immunization_pca" in new_ds.df.columns
    assert all(col not in new_ds.df.columns for col in cols)
    assert concat.explained_variance > 0
    assert set(concat.loadings["feature"]) == set(cols)
    assert new_ds.df["immunization_pca"].notna().all()


def test_suggest_groups_from_correlation_real_data(life_expectancy_dataset) -> None:
    """Hierarchical clustering on real correlations should group related features."""
    feature_subset = [
        LECol.THINNESS_1_19_YEARS,
        LECol.THINNESS_5_9_YEARS,
        LECol.INFANT_DEATHS,
        LECol.UNDER_FIVE_DEATHS,
    ]
    corr_mat = life_expectancy_dataset.df[feature_subset].corr()

    groups, summary = suggest_groups_from_correlation(
        corr_mat,
        threshold=0.6,
        min_group_size=2,
        return_summary=True,
    )

    assert groups
    assert isinstance(summary, pd.DataFrame)
    assert {"group", "size", "features", "mean_abs_corr", "min_abs_corr"}.issubset(summary.columns)
    assert any({LECol.THINNESS_1_19_YEARS, LECol.THINNESS_5_9_YEARS}.issubset(set(g.features)) for g in groups)
    assert all(set(g.features).issubset(corr_mat.columns) for g in groups)
