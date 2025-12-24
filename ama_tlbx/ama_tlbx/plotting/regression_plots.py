"""Plotting helpers for regression diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.regressionplots import influence_plot

from ama_tlbx.analysis.ols_diagnostics import design_matrix_from_model
from ama_tlbx.data import LECol


if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ama_tlbx.analysis.ols_diagnostics import RegressionResult


def plot_residuals_vs_fitted(
    result: RegressionResult,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Residuals vs fitted values with LOESS smooth.

    Wraps [:func:`seaborn.residplot`](https://seaborn.pydata.org/generated/seaborn.residplot.html)
    on the statsmodels OLS residuals contained in ``RegressionResult``.
    """
    ax = ax or plt.gca()
    sns.residplot(x=result.fitted, y=result.residuals, lowess=True, ax=ax, scatter_kws={"alpha": 0.45})
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    return ax


def plot_scale_location(
    result: RegressionResult,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scale-location scatter plot to display sqrt(|studentized residuals|) against fitted values."""
    ax = ax or plt.gca()
    influence = result.model.get_influence()
    stud_resid = influence.resid_studentized_internal
    sns.scatterplot(x=result.fitted, y=(abs(stud_resid) ** 0.5), ax=ax, alpha=0.45, color="tab:orange")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("sqrt(|studentized residuals|)")
    ax.set_title("Scale-Location")
    return ax


def plot_qq(
    result: RegressionResult,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """QQ plot of studentized residuals to assess normality."""
    ax = ax or plt.gca()
    influence = result.model.get_influence()
    stud_resid = influence.resid_studentized_internal
    qqplot(stud_resid, line="45", fit=True, ax=ax)
    ax.set_title("QQ plot (studentized residuals)")
    return ax


def plot_influence(
    result: RegressionResult,
    *,
    ax: plt.Axes | None = None,
) -> Figure:
    """Cook's distance / leverage influence plot."""
    fig = influence_plot(result.model, criterion="cooks", ax=ax)
    fig.set_figwidth(8)
    fig.set_figheight(6)
    return fig


def pred_plot(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    feat: str,
    df: pd.DataFrame,
    *,
    n_points: int = 300,
    ax: plt.Axes | None = None,
    target_col: str | None = None,
) -> plt.Axes:
    """Plot fitted line with 95% CI for a single predictor.

    The plot holds other predictors at their sample means, so the line reflects
    the partial relationship between ``feat`` and the target under the fitted
    model. The band is a 95% confidence interval for the mean response (not a
    prediction interval for new observations).
    """
    if feat not in df.columns:
        raise KeyError(f"Feature '{feat}' not in provided DataFrame")

    ax = ax or plt.gca()
    series = df[feat]
    if pd.api.types.is_numeric_dtype(series):
        grid = np.linspace(series.min(), series.max(), n_points)
    else:
        grid = series.dropna().unique()
        if grid.size == 0:
            raise ValueError(f"Feature '{feat}' has no non-null values to plot.")

    data = getattr(model, "model", None).data if hasattr(model, "model") else None
    design_info = getattr(data, "design_info", None)
    frame = getattr(data, "frame", None)

    if design_info is not None and frame is not None:
        base = {}
        for col in frame.columns:
            if col in {data.ynames, getattr(model.model, "endog_names", "")}:
                continue
            if col == feat:
                base[col] = grid
                continue
            col_series = frame[col]
            if pd.api.types.is_numeric_dtype(col_series):
                fill_value = float(col_series.astype(float).mean())
            else:
                mode = col_series.mode(dropna=True)
                fill_value = mode.iloc[0] if not mode.empty else col_series.dropna().iloc[0]
            base[col] = np.repeat(fill_value, len(grid))
        exog_pred = pd.DataFrame(base)
    else:
        design_matrix = design_matrix_from_model(model)
        exog_names = model.model.exog_names
        if feat not in exog_names:
            raise KeyError(f"Feature '{feat}' not in model design matrix.")
        exog_pred = pd.DataFrame(index=range(len(grid)), columns=exog_names, dtype=float)
        for name in exog_names:
            if name == feat:
                exog_pred[name] = grid
            else:
                exog_pred[name] = float(design_matrix[name].mean())

    pred_summary = model.get_prediction(exog_pred).summary_frame()
    if target_col is None:
        target_col = getattr(data, "ynames", None) if data is not None else None
        target_col = target_col or getattr(model.model, "endog_names", None)
        if target_col not in df.columns:
            target_col = LECol.TARGET if LECol.TARGET in df.columns else None

    if target_col is not None and target_col in df.columns:
        sns.scatterplot(data=df, x=feat, y=target_col, alpha=0.45, ax=ax)
    ax.plot(grid, pred_summary["mean"], color="tab:red", linewidth=2.5)
    ax.fill_between(grid, pred_summary["mean_ci_lower"], pred_summary["mean_ci_upper"], color="tab:red", alpha=0.2)
    ax.set_xlabel(feat)
    if target_col is not None:
        ax.set_ylabel(target_col)
    else:
        ax.set_ylabel("Target")
    ax.set_title("OLS prediction ±95% CI")
    return ax
