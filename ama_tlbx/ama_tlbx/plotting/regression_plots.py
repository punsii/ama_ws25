"""Plotting helpers for regression diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.regressionplots import influence_plot

from ama_tlbx.analysis.ols_helper import (
    cooksd_contours,
    design_matrix_for_data,
    design_matrix_from_model,
)
from ama_tlbx.data import LECol


_MAX_DISCRETE_LEVELS = 6


if TYPE_CHECKING:
    import statsmodels.api as sm
    from matplotlib.figure import Figure

    from ama_tlbx.analysis.model_selection import SelectionPathResult
    from ama_tlbx.analysis.ols_helper import EvalMetrics, RegressionResult


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
    data = getattr(model, "model", None).data if hasattr(model, "model") else None
    design_info = getattr(data, "design_info", None)
    frame = getattr(data, "frame", None)
    categorical_levels = None
    categorical_map: dict[str, tuple] = {}
    if design_info is not None:
        for factor, finfo in design_info.factor_infos.items():
            code = getattr(factor, "code", None)
            if finfo.type == "categorical" and isinstance(code, str) and code.startswith("C(") and code.endswith(")"):
                raw_name = code[2:-1]
                categorical_map[raw_name] = finfo.categories
        categorical_levels = categorical_map.get(feat)

    if categorical_levels is not None:
        grid = np.asarray(categorical_levels)
    elif pd.api.types.is_numeric_dtype(series):
        grid = np.linspace(series.min(), series.max(), n_points)
    else:
        grid = series.dropna().unique()
        if grid.size == 0:
            raise ValueError(f"Feature '{feat}' has no non-null values to plot.")

    if design_info is not None and frame is not None:
        base = {}
        for col in frame.columns:
            if col in {data.ynames, getattr(model.model, "endog_names", "")}:
                continue
            if col == feat:
                base[col] = grid
                continue
            col_series = frame[col]
            if col in categorical_map:
                categories = categorical_map[col]
                fill_value = categories[0] if categories else col_series.dropna().iloc[0]
            elif pd.api.types.is_numeric_dtype(col_series):
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


def plot_observed_vs_fitted(
    result: RegressionResult,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Observed vs fitted values with 45-degree reference line.

    What it shows:
        Agreement between observed targets and model predictions.
    How to interpret:
        Points close to the diagonal indicate good calibration; systematic
        curvature or fanning suggests nonlinearity or heteroscedasticity.
    """
    ax = ax or plt.gca()
    sns.scatterplot(x=result.fitted, y=result.y, alpha=0.45, ax=ax)
    line_min = float(min(result.fitted.min(), result.y.min()))
    line_max = float(max(result.fitted.max(), result.y.max()))
    ax.plot([line_min, line_max], [line_min, line_max], color="tab:red", linestyle="--")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Observed values")
    ax.set_title("Observed vs Fitted")
    return ax


def plot_calibration(  # noqa: C901, PLR0913, PLR0914
    metrics: EvalMetrics,
    *,
    ax: plt.Axes | None = None,
    bins: int = 10,
    bootstrap: int = 300,
    ci: float = 0.95,
    random_state: int | None = None,
    scatter_alpha: float = 0.4,
    show_binned: bool = True,
) -> plt.Axes:
    """Plot observed vs predicted values on an evaluation set with bootstrap CIs.

    The calibration curve is shown as binned averages (by predicted value
    quantiles) with bootstrap confidence intervals for the mean observed target
    within each bin.

    Args:
        metrics: Evaluation metrics containing ``y_true`` and ``y_pred``.
        ax: Optional matplotlib axis.
        bins: Number of quantile bins for the calibration curve.
        bootstrap: Number of bootstrap resamples per bin (0 disables CIs).
        ci: Confidence level, e.g. 0.95.
        random_state: RNG seed.
        scatter_alpha: Alpha for the raw scatter points.
        show_binned: If True, add binned mean calibration curve.
    """
    if not (0 < ci < 1):
        raise ValueError("ci must be in (0, 1).")
    if bins < 2:
        raise ValueError("bins must be >= 2.")
    if bootstrap < 0:
        raise ValueError("bootstrap must be >= 0.")

    ax = ax or plt.gca()
    df_plot = (
        pd.DataFrame(
            {
                "y_pred": pd.Series(metrics.y_pred, name="y_pred").astype(float),
                "y_true": pd.Series(metrics.y_true, name="y_true").astype(float),
            },
        )
        .dropna()
        .sort_values("y_pred")
    )

    sns.scatterplot(data=df_plot, x="y_pred", y="y_true", alpha=scatter_alpha, ax=ax)
    line_min = float(min(df_plot["y_pred"].min(), df_plot["y_true"].min()))
    line_max = float(max(df_plot["y_pred"].max(), df_plot["y_true"].max()))
    ax.plot([line_min, line_max], [line_min, line_max], color="tab:red", linestyle="--", linewidth=1.5)

    if show_binned and not df_plot.empty:
        pred_vals = df_plot["y_pred"].to_numpy()
        edges = np.unique(np.quantile(pred_vals, np.linspace(0, 1, bins + 1)))
        if edges.size >= 3:
            df_plot = df_plot.assign(
                pred_bin=pd.cut(df_plot["y_pred"], bins=edges, include_lowest=True, duplicates="drop"),
            )
            grouped = df_plot.dropna(subset=["pred_bin"]).groupby("pred_bin", observed=True)
            bin_pred = grouped["y_pred"].mean()
            bin_obs = grouped["y_true"].mean()

            ax.plot(bin_pred, bin_obs, color="black", marker="o", linewidth=2.0, label="binned mean")

            if bootstrap > 0:
                rng = np.random.default_rng(random_state)
                alpha = (1.0 - ci) / 2.0
                ci_low: list[float] = []
                ci_high: list[float] = []
                for _, grp in grouped:
                    values = grp["y_true"].to_numpy()
                    n = values.size
                    if n == 0:
                        ci_low.append(float("nan"))
                        ci_high.append(float("nan"))
                        continue
                    if n == 1:
                        ci_low.append(float(values[0]))
                        ci_high.append(float(values[0]))
                        continue
                    boot_means = np.empty(bootstrap, dtype=float)
                    for b in range(bootstrap):
                        sample = rng.choice(values, size=n, replace=True)
                        boot_means[b] = float(np.mean(sample))
                    ci_low.append(float(np.quantile(boot_means, alpha)))
                    ci_high.append(float(np.quantile(boot_means, 1 - alpha)))

                ax.fill_between(bin_pred.to_numpy(), ci_low, ci_high, color="black", alpha=0.15, linewidth=0)
                ax.legend(loc="best")

    label = metrics.label or "eval"
    ax.set_title(f"Calibration ({label}) — RMSE={metrics.rmse:.2f}, R²={metrics.r2:.3f}")
    ax.set_xlabel("Predicted life expectancy")
    ax.set_ylabel("Observed life expectancy")
    return ax


def plot_residual_hist(
    result: RegressionResult,
    *,
    ax: plt.Axes | None = None,
    bins: int = 30,
) -> plt.Axes:
    """Histogram of residuals with optional KDE overlay.

    What it shows:
        The residual distribution relative to normality and symmetry.
    How to interpret:
        Skewness or heavy tails can indicate non-normal errors or outliers.
    """
    ax = ax or plt.gca()
    sns.histplot(result.residuals, bins=bins, kde=True, ax=ax, color="tab:green")
    ax.set_xlabel("Residuals")
    ax.set_title("Residual distribution")
    return ax


def plot_residuals_vs_predictors(
    result: RegressionResult,
    *,
    predictors: list[str] | None = None,
    max_cols: int = 4,
    figsize: tuple[int, int] | None = None,
) -> Figure:
    """Residuals vs predictors to check linearity and variance patterns.

    What it shows:
        Whether residuals vary with specific predictors.
    How to interpret:
        Trends or funnel shapes suggest nonlinearity or heteroscedasticity tied
        to a covariate.
    """
    design = result.design_matrix
    resid = result.residuals
    all_preds = [c for c in design.columns if c not in {"Intercept", "const"}]
    preds = all_preds if predictors is None else [p for p in predictors if p in all_preds]

    n_cols = max(1, max_cols)
    n_rows = max(1, int(np.ceil(len(preds) / n_cols)))
    if figsize is None:
        figsize = (max(8, int(3.5 * n_cols)), max(3, int(3.0 * n_rows)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_list = np.atleast_1d(axes).ravel()
    for ax, pred in zip(axes_list, preds, strict=False):
        sns.scatterplot(x=design[pred], y=resid, ax=ax, alpha=0.45)
        ax.set_title(f"Residuals vs {pred}")
        ax.set_xlabel(pred)
        ax.set_ylabel("Residuals")
    for ax in axes_list[len(preds) :]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_leverage_resid_cooks(
    result: RegressionResult,
    *,
    ax: plt.Axes | None = None,
    levels: tuple[float, ...] = (0.5, 1.0),
) -> plt.Axes:
    """Leverage vs studentized residuals with Cook's distance contours.

    What it shows:
        High leverage points with large residuals and Cook's distance overlays.
    How to interpret:
        Points outside Cook's contours can be influential and worth inspection.
    """
    ax = ax or plt.gca()
    influence = result.model.get_influence()
    leverage = influence.hat_matrix_diag
    resid = influence.resid_studentized_internal
    cooks = influence.cooks_distance[0]
    size = 50 * (cooks / cooks.max()) if cooks.max() > 0 else 20
    ax.scatter(leverage, resid, s=size, alpha=0.45, color="tab:blue")
    cooksd_contours(ax, leverage, levels=levels)
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Studentized residuals")
    ax.set_title("Leverage vs Studentized Residuals")
    return ax


def plot_selection_path(
    path: SelectionPathResult,
    *,
    metric: str = "aic",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot selection criterion across steps in a model-selection path.

    What it shows:
        How the chosen criterion evolves as terms are added/removed.
    How to interpret:
        Lower AIC/BIC/Cp or lower CV RMSE indicates a better trade-off between
        fit and complexity.
    """
    ax = ax or plt.gca()
    table = path.summary_table()
    if metric not in table.columns:
        raise KeyError(f"Metric '{metric}' not found in selection summary.")
    ax.plot(table.index, table[metric], marker="o", linestyle="-", color="tab:blue")
    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.set_title(f"Selection path ({metric})")
    return ax


def plot_interaction_effect(  # noqa: C901, PLR0912, PLR0913
    result: RegressionResult,
    *,
    x: str,
    by: str,
    df: pd.DataFrame | None = None,
    n_points: int = 100,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot marginal interaction effects by varying `x` across levels of `by`.

    What it shows:
        Predicted response curves for different levels of the interacting term.
    How to interpret:
        Diverging slopes across levels indicate effect modification (interaction).
    """
    ax = ax or plt.gca()
    data = getattr(getattr(result.model, "model", None), "data", None)
    design_info = getattr(data, "design_info", None)
    categorical_map: dict[str, tuple] = {}
    if design_info is not None:
        for factor, finfo in design_info.factor_infos.items():
            code = getattr(factor, "code", None)
            if finfo.type == "categorical" and isinstance(code, str) and code.startswith("C(") and code.endswith(")"):
                raw_name = code[2:-1]
                categorical_map[raw_name] = finfo.categories
    frame = getattr(data, "frame", None)
    if df is None:
        if isinstance(frame, pd.DataFrame):
            df = frame
        else:
            raise ValueError("A DataFrame with original features is required for interaction plots.")

    if x not in df.columns or by not in df.columns:
        raise KeyError("Both 'x' and 'by' must be present in the provided DataFrame.")

    x_series = df[x]
    if x in categorical_map:
        x_grid = np.asarray(categorical_map[x])
    elif pd.api.types.is_numeric_dtype(x_series):
        x_grid = np.linspace(float(x_series.min()), float(x_series.max()), n_points)
    else:
        x_grid = x_series.dropna().unique()
        if x_grid.size == 0:
            raise ValueError(f"Feature '{x}' has no non-null values to plot.")

    by_series = df[by]
    if by in categorical_map:
        by_levels = np.asarray(categorical_map[by])
    elif pd.api.types.is_numeric_dtype(by_series):
        unique = np.sort(pd.unique(by_series.dropna()))
        # Treat small-cardinality numeric variables (e.g., 0/1 dummies) as discrete levels.
        by_levels = (
            unique
            if unique.size <= _MAX_DISCRETE_LEVELS
            else np.unique(np.quantile(by_series.dropna(), [0.25, 0.5, 0.75]))
        )
    else:
        by_levels = by_series.dropna().unique()

    for level in by_levels:
        base: dict[str, np.ndarray | float | str] = {}
        for col in df.columns:
            if col in {data.ynames, getattr(result.model.model, "endog_names", "")}:
                continue
            if col == x:
                base[col] = x_grid
            elif col == by:
                base[col] = np.repeat(level, len(x_grid))
            else:
                series = df[col]
                if col in categorical_map:
                    categories = categorical_map[col]
                    fill_value = categories[0] if categories else series.dropna().iloc[0]
                elif pd.api.types.is_numeric_dtype(series):
                    fill_value = float(series.mean())
                else:
                    mode = series.mode(dropna=True)
                    fill_value = mode.iloc[0] if not mode.empty else series.dropna().iloc[0]
                base[col] = np.repeat(fill_value, len(x_grid))

        pred_df = pd.DataFrame(base)
        if design_info is not None:
            pred = result.model.predict(pred_df)
        else:
            exog = design_matrix_for_data(result.model, pred_df)
            pred = result.model.predict(exog)
        ax.plot(x_grid, pred, label=f"{by}={level}")

    ax.set_xlabel(x)
    ax.set_ylabel("Predicted target")
    ax.set_title(f"Interaction: {x} by {by}")
    ax.legend()
    return ax
