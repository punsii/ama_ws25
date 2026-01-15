"""Integration tests for regression diagnostics helpers using real data."""

import pytest

from ama_tlbx.analysis import fit_ols_design, fit_ols_formula
from ama_tlbx.data import LECol


def test_regression_diagnostics_real_data(life_expectancy_dataset) -> None:
    """Analyzer should compute metrics, assumptions, VIF, and CV on real data."""
    view = life_expectancy_dataset.analyzer_view(
        columns=None,
        standardized=True,
        include_target=True,
    )
    result = fit_ols_design(
        view.df,
        target_col=LECol.TARGET,
        cv_folds=4,
        shuffle_cv=True,
        random_state=0,
    )

    assert result.metrics.rmse > 0
    assert 0 <= result.metrics.r2 <= 1
    assert result.metrics.cv_scores is not None
    assert len(result.metrics.cv_scores) == 4
    assert result.metrics.cv_rmse is not None
    assert result.assumptions.durbin_watson > 0
    assert not result.vif["vif"].isna().any()
    assert len(result.residuals) == len(result.fitted)


def test_regression_diagnostics_formula_rhs(life_expectancy_dataset) -> None:
    """Formula-based path should skip CV when k_folds is None."""
    result = fit_ols_formula(
        life_expectancy_dataset.df,
        rhs=f"{LECol.GDP} + {LECol.BMI} + {LECol.SCHOOLING}",
        cv_folds=None,
    )

    assert result.metrics.cv_rmse is None
    assert result.metrics.n_obs == len(life_expectancy_dataset.df)
    assert result.assumptions.breusch_pagan_pvalue <= 1


def test_regression_diagnostics_requires_target(life_expectancy_dataset) -> None:
    """fit_ols_design should error when the target column is missing."""
    view = life_expectancy_dataset.analyzer_view(
        columns=[LECol.GDP, LECol.BMI],
        standardized=True,
        include_target=False,
    )
    with pytest.raises(KeyError, match=LECol.TARGET):
        fit_ols_design(view.df, target_col=LECol.TARGET)
