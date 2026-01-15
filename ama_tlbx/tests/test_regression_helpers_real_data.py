"""Real-data tests for regression helper utilities."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ama_tlbx.analysis.ols_helper import fit_ols_design, fit_ols_formula
from ama_tlbx.data import LECol
from ama_tlbx.plotting.regression_plots import pred_plot


@pytest.fixture(scope="session")
def regression_rhs() -> str:
    """Common right-hand side used across helper tests."""
    return f"{LECol.GDP} + {LECol.BMI} + {LECol.SCHOOLING}"


def test_fit_ols_with_diagnostics_real_data(life_expectancy_df, regression_rhs) -> None:
    """End-to-end fit using the real life expectancy dataset."""
    result = fit_ols_formula(life_expectancy_df, rhs=regression_rhs, cv_folds=5)

    assert result.metrics.r2 > 0
    assert 0 < result.metrics.rmse < 20
    assert result.metrics.cv_scores is not None
    assert len(result.metrics.cv_scores) == 5
    assert all(np.isfinite(result.assumptions.vif))
    assert result.predictions.shape[0] == len(life_expectancy_df)
    assert set(result.design_matrix.columns).issuperset({LECol.GDP, LECol.BMI, LECol.SCHOOLING})


def test_fit_helpers_formula_vs_matrix_agree(life_expectancy_df) -> None:
    """Matrix and formula helpers should yield comparable R² on real data."""
    df = life_expectancy_df[[LECol.TARGET, LECol.GDP, LECol.BMI]].copy()

    model_matrix = fit_ols_design(df, target_col=LECol.TARGET)
    model_formula = fit_ols_formula(df, rhs=f"{LECol.GDP} + {LECol.BMI}")

    assert np.isclose(model_matrix.model.rsquared, model_formula.model.rsquared, atol=1e-6)
    assert model_matrix.model.params.index.tolist()[0] in {"const", "Intercept"}


def test_pred_plot_sets_labels_and_returns_axis(life_expectancy_df, regression_rhs) -> None:
    """pred_plot should render without error and label axes using the feature name."""
    model = fit_ols_formula(life_expectancy_df, rhs=regression_rhs)

    ax = pred_plot(model.model, LECol.GDP, life_expectancy_df, n_points=50)

    assert ax.get_xlabel() == LECol.GDP
    assert ax.get_ylabel() == LECol.TARGET
    assert "OLS prediction" in ax.get_title()
    plt.close(ax.figure)


def test_pred_plot_raises_for_unknown_feature(life_expectancy_df) -> None:
    """Unknown predictor names should raise a KeyError."""
    model = fit_ols_formula(life_expectancy_df, rhs=f"{LECol.GDP} + {LECol.BMI}")
    with pytest.raises(KeyError):
        pred_plot(model.model, "not_a_column", life_expectancy_df)


def test_residual_diagnostics_handles_misaligned_predictors(monkeypatch, life_expectancy_df) -> None:
    """residual_diagnostics should not crash when requested predictors are missing."""
    model = fit_ols_formula(life_expectancy_df, rhs=f"{LECol.GDP} + {LECol.BMI}")

    show_calls = {"count": 0}
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: show_calls.update(count=show_calls["count"] + 1))

    model.plot_residual_diags(predictors=[LECol.GDP, "nonexistent"], max_cols=1)

    assert show_calls["count"] >= 1
