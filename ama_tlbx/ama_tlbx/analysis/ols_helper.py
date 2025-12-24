"""OLS diagnostics, model-fitting helpers, and plotting utilities.

The helpers in this module focus on classical linear regression with ordinary
least squares (OLS). They compute standard fit metrics (e.g., :math:`R^2`,
RMSE, AIC/BIC), generate diagnostic plots, and run common assumption checks for
linearity, independence, homoscedasticity, normality, and collinearity. Tests
are *diagnostic* rather than definitive: small p-values indicate evidence
against the null (e.g., heteroscedasticity or non-normal residuals), but results
are sensitive to sample size and should be interpreted alongside residual plots
and domain knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import itemgetter
from typing import TYPE_CHECKING, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from attr import asdict
from patsy import build_design_matrices
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_score
from statsmodels.graphics.regressionplots import influence_plot
from statsmodels.stats import diagnostic as sm_diagnostic
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from ..data import LECol


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


_INTERCEPT_COLS = ("Intercept", "const")
_SHAPIRO_MAX_N = 5000


@dataclass(frozen=True)
class MetricsResult:
    r"""Fit and generalization metrics for OLS models.

    Key equations (with :math:`n` observations and :math:`p` predictors):

    - :math:`R^2 = 1 - \frac{SS_{res}}{SS_{tot}}`
    - :math:`\bar{R}^2 = 1 - (1 - R^2)\frac{n-1}{n-p-1}`
    - :math:`\text{RMSE} = \sqrt{\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2}`
    - :math:`\text{MAE} = \frac{1}{n}\sum_i |y_i - \hat{y}_i|`
    - :math:`\text{MAPE} = \frac{1}{n}\sum_i \left|\frac{y_i - \hat{y}_i}{y_i}\right|`
    - :math:`\text{AIC} = 2k - 2\log L`, :math:`\text{BIC} = k\log n - 2\log L`

    Information criteria are most meaningful for *relative* comparisons across
    models fit on the same response and dataset (lower is better).
    """

    r2: float
    """Coefficient of determination: :math:`1 - SS_{res}/SS_{tot}`.

    Interpreted as the proportion of variance explained by the model; higher is
    better (subject to overfitting risk).
    """

    adj_r2: float | None
    r"""Adjusted :math:`R^2` penalising extra predictors.

    :math:`\bar{R}^2 = 1 - (1 - R^2)\frac{n-1}{n-p-1}`.
    """

    rmse: float
    r"""Root mean squared error (in target units).

    :math:`\sqrt{\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2}`; lower is better.
    """

    mae: float
    r"""Mean absolute error (in target units).

    :math:`\frac{1}{n}\sum_i |y_i - \hat{y}_i|`; lower is better and more robust
    to outliers than RMSE.
    """

    mape: float | None
    r"""Mean absolute percentage error (fraction); ``None`` when undefined.

    :math:`\frac{1}{n}\sum_i \left|\frac{y_i - \hat{y}_i}{y_i}\right|`; undefined
    when any :math:`y_i = 0`.
    """

    aic: float | None
    r"""Akaike Information Criterion.

    :math:`\text{AIC} = 2k - 2\log L`, where :math:`k` is the number of
    estimated parameters and :math:`L` is the maximized likelihood. Lower is
    preferred in model comparison.
    """

    bic: float | None
    r"""Bayesian Information Criterion.

    :math:`\text{BIC} = k\log n - 2\log L`; stronger penalty on complexity than
    AIC. Lower is preferred in model comparison.
    """

    loglik: float | None
    """Log-likelihood of the fitted model (at the MLE)."""

    n_obs: float | None
    """Number of observations used in the fit (:math:`n`)."""

    cv_scores: list[float] | None
    """Raw cross-validation RMSE scores (if enabled)."""

    cv_rmse: float | None
    """Mean CV RMSE when cross-validation is configured (lower is better)."""

    def __repr__(self) -> str:
        def fmt(value: float | None, decimals: int = 3) -> str:
            if value is None:
                return "nan"
            return f"{value:.{decimals}f}"

        fit_block = (
            "Fit["
            f"r2={fmt(self.r2)}, "
            f"adj_r2={fmt(self.adj_r2)}, "
            f"rmse={fmt(self.rmse)}, "
            f"mae={fmt(self.mae)}, "
            f"mape={fmt(self.mape)}, "
            f"aic={fmt(self.aic)}, "
            f"bic={fmt(self.bic)}"
            "]"
        )
        cv_block = ""
        if self.cv_rmse is not None or self.cv_scores is not None:
            folds = len(self.cv_scores) if self.cv_scores is not None else 0
            cv_block = f" CV[rmse={fmt(self.cv_rmse)}, folds={folds}]"
        n_obs = f" n={int(self.n_obs)}" if self.n_obs is not None else ""
        return f"MetricsResult({fit_block}{cv_block}{n_obs})"


@dataclass(frozen=True)
class AssumptionCheckResult:
    """Regression assumption diagnostics with canonical test references.

    The tests reported here are standard OLS diagnostics:

    - Independence (autocorrelation): Durbin-Watson.
    - Normality of residuals: Jarque-Bera, Shapiro-Wilk (or Anderson-Darling).
    - Homoscedasticity: Breusch-Pagan and White tests.
    - Collinearity: condition number and variance inflation factors (VIF).
    - Influence: leverage and Cook's distance.

    Most statistics are compared against a null hypothesis (e.g., homoscedastic
    errors). Small p-values indicate evidence against that assumption.
    """

    durbin_watson: float
    r"""Durbin-Watson statistic for residual autocorrelation.

    :math:`DW = \frac{\sum_t (e_t - e_{t-1})^2}{\sum_t e_t^2}`, ranges in
    :math:`[0, 4]`. Values near 2 indicate no autocorrelation; <2 suggests
    positive and >2 negative autocorrelation.
    """

    jarque_bera_statistic: float
    r"""Jarque-Bera test statistic for residual normality.

    Uses skewness (:math:`S`) and kurtosis (:math:`K`):
    :math:`JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)`, asymptotically
    :math:`\chi^2_2`.
    """

    jarque_bera_pvalue: float
    """Jarque-Bera p-value for residual normality (small => non-normality)."""

    shapiro_statistic: float
    """Shapiro-Wilk W statistic for residual normality (or AD fallback).

    Shapiro-Wilk assesses correlation between ordered residuals and normal
    scores. For large samples (>5k), Anderson-Darling (AD) is used; AD measures
    weighted squared distance between the empirical and normal CDFs.
    """

    shapiro_pvalue: float
    """Shapiro-Wilk (or AD) p-value for residual normality (small => non-normality)."""

    breusch_pagan_statistic: float
    r"""Breusch-Pagan LM statistic for heteroscedasticity.

    Computes an auxiliary regression of squared residuals on the design matrix;
    the LM statistic is asymptotically :math:`\chi^2`.
    """

    breusch_pagan_pvalue: float
    """Breusch-Pagan p-value (small => heteroscedasticity)."""

    white_statistic: float
    """White test LM statistic for general heteroscedasticity.

    Similar to Breusch-Pagan but includes squares and interactions to detect
    broader variance patterns.
    """

    white_pvalue: float
    """White test p-value (small => heteroscedasticity)."""

    condition_number: float
    """Condition number of the design matrix (large => collinearity risk).

    Ratio of the largest to smallest singular values of :math:`X`; large values
    indicate multicollinearity and numerical instability.
    """

    vif: pd.Series
    """Variance Inflation Factor per predictor (intercept excluded).

    :math:`VIF_j = \frac{1}{1 - R_j^2}` where :math:`R_j^2` is from regressing
    predictor :math:`j` on all others. Large values indicate collinearity.
    """

    leverage: np.ndarray
    """Diagonal of the hat matrix; high leverage implies strong influence.

    :math:`h_{ii}` from :math:`H = X(X'X)^{-1}X'`; common heuristics flag
    :math:`h_{ii} > 2p/n` or :math:`3p/n`.
    """

    cooks_distance: np.ndarray
    """Cook's distance per observation; heuristic flags often use ``4/n``.

    Measures how much fitted values change if observation ``i`` is removed.
    """

    def __repr__(self) -> str:
        alpha = 0.05

        def fmt(value: float, decimals: int = 3) -> str:
            return f"{value:.{decimals}f}"

        def decision(p_value: float) -> str:
            return "FAIL" if p_value < alpha else "OK"

        normality = (
            "Normality: "
            f"JB(stat={fmt(self.jarque_bera_statistic, 2)}, p={fmt(self.jarque_bera_pvalue)}, {decision(self.jarque_bera_pvalue)}); "
            f"Shapiro/AD(stat={fmt(self.shapiro_statistic, 2)}, p={fmt(self.shapiro_pvalue)}, {decision(self.shapiro_pvalue)})"
        )
        homoscedasticity = (
            "Homoscedasticity: "
            f"BP(stat={fmt(self.breusch_pagan_statistic, 2)}, p={fmt(self.breusch_pagan_pvalue)}, {decision(self.breusch_pagan_pvalue)}); "
            f"White(stat={fmt(self.white_statistic, 2)}, p={fmt(self.white_pvalue)}, {decision(self.white_pvalue)})"
        )
        dw_status = "OK" if 1.5 <= self.durbin_watson <= 2.5 else "WARN"
        autocorr = f"Autocorrelation: Durbin-Watson={fmt(self.durbin_watson, 2)} ({dw_status})"

        vif_series = self.vif if isinstance(self.vif, pd.Series) else pd.Series(self.vif)
        max_vif = float(vif_series.max()) if not vif_series.empty else float("nan")
        mean_vif = float(vif_series.mean()) if not vif_series.empty else float("nan")
        collinearity = (
            "Collinearity: "
            f"cond#={fmt(self.condition_number, 2)}, "
            f"max_vif={fmt(max_vif, 2)}, "
            f"mean_vif={fmt(mean_vif, 2)}"
        )

        n_obs = int(self.cooks_distance.shape[0]) if hasattr(self.cooks_distance, "shape") else len(self.cooks_distance)
        cooks_thresh = 4 / n_obs if n_obs else float("nan")
        cooks_exceed = int(np.sum(self.cooks_distance > cooks_thresh)) if n_obs else 0
        influence = (
            "Influence: "
            f"max_leverage={fmt(float(np.max(self.leverage)), 3)}, "
            f"max_cook={fmt(float(np.max(self.cooks_distance)), 3)}, "
            f"cooks>4/n={cooks_exceed}"
        )

        return (
            "AssumptionCheckResult(\n"
            f"  {normality}\n"
            f"  {homoscedasticity}\n"
            f"  {autocorr}\n"
            f"  {collinearity}\n"
            f"  {influence}\n"
            ")"
        )


@dataclass(frozen=True)
class RegressionResult:
    """Packaged OLS fit, metrics, and diagnostics for reporting.

    Encapsulates the fitted statsmodels result, the design matrix used for the
    fit, residual diagnostics, and convenience plotting helpers.
    """

    model: sm.regression.linear_model.RegressionResultsWrapper
    design_matrix: pd.DataFrame
    y: pd.Series
    metrics: MetricsResult
    assumptions: AssumptionCheckResult
    residuals: pd.Series
    predictions: pd.Series

    def print_summary(self) -> str:
        """Print the statsmodels summary to stdout (returns ``None``)."""
        return print(self.model.summary())

    @property
    def fitted(self) -> pd.Series:
        """Alias for fitted values aligned with ``residuals``."""
        return self.predictions

    @property
    def r2(self) -> float:
        """Coefficient of determination :math:`R^2` of the fitted model."""
        return self.metrics.r2

    @property
    def aic(self) -> float:
        """Akaike Information Criterion (AIC) of the fitted model."""
        return self.metrics.aic or float("nan")

    @property
    def adj_r2(self) -> float:
        """Adjusted :math:`R^2` correcting for the number of regressors."""
        return self.metrics.adj_r2 or float("nan")

    @property
    def rmse(self) -> float:
        """Root mean squared error (RMSE) of the in-sample residuals."""
        return self.metrics.rmse

    @property
    def vif(self) -> pd.DataFrame:
        """Variance-inflation factors as a tidy DataFrame."""
        return self.assumptions.vif.rename_axis("feature").reset_index(name="vif")

    @property
    def max_leverage(self) -> float:
        """Maximum leverage across observations."""
        return np.max(self.assumptions.leverage)

    @property
    def max_cooks(self) -> float:
        """Maximum Cook's distance across observations."""
        return np.max(self.assumptions.cooks_distance)

    # ------------------------------------------------------------------ plotting shortcuts
    def plot_residuals_vs_fitted(self, **kwargs: object) -> Axes:
        r"""Plot residuals against fitted values.

        What it shows:
            A scatter of residuals :math:`e_i = y_i - \hat{y}_i` versus fitted
            values :math:`\hat{y}_i`, with a LOESS/LOWESS smooth (via seaborn).

        How to interpret:
            - Desired: a random cloud centered around 0 with roughly constant
              vertical spread across the x-axis.
            - Curvature/patterns in the smooth: suggests nonlinearity, missing
              terms/interactions, or an incorrect functional form.
            - Funnel/megaphone spread: suggests heteroscedasticity (non-constant
              error variance).
            - Isolated points far from 0: potential outliers (inspect alongside
              leverage/influence).
        """
        from ama_tlbx.plotting.regression_plots import plot_residuals_vs_fitted  # noqa: PLC0415

        return plot_residuals_vs_fitted(self, **kwargs)

    def plot_scale_location(self, **kwargs: object) -> Axes:
        r"""Plot scale-location (spread) to assess homoscedasticity.

        What it shows:
            A scatter of :math:`\sqrt{|r_i|}` against fitted values
            :math:`\hat{y}_i`, where :math:`r_i` are (internally) studentized
            residuals. Studentization rescales residuals by an estimate of their
            standard deviation, accounting for leverage.

        How to interpret:
            - Desired: roughly horizontal band (no trend) and constant spread.
            - Increasing spread with fitted values: suggests heteroscedasticity.
            - Very high points: candidates for outliers in (studentized)
              residual magnitude.
        """
        from ama_tlbx.plotting.regression_plots import plot_scale_location  # noqa: PLC0415

        return plot_scale_location(self, **kwargs)

    def plot_qq(self, **kwargs: object) -> Axes:
        """Plot a normal Q-Q plot of studentized residuals.

        What it shows:
            Ordered studentized residuals plotted against theoretical quantiles
            from a normal distribution (a straight 45-degree reference line is
            added).

        How to interpret:
            - Points close to the line: residual distribution is approximately
              normal.
            - S-shape / systematic curvature: skewness or heavy/light tails.
            - Deviations in the extremes: tail issues and/or outliers.

        Notes:
            Normal residuals are mainly important for small-sample t/F-based
            inference; point estimates can still be useful under non-normality,
            especially with larger samples.
        """
        from ama_tlbx.plotting.regression_plots import plot_qq  # noqa: PLC0415

        return plot_qq(self, **kwargs)

    def plot_influence(self, **kwargs: object) -> Figure:
        """Plot influence diagnostics (leverage, residuals, Cook's distance).

        What it shows (statsmodels influence plot):
            - X-axis: leverage :math:`h_{ii}` (hat-matrix diagonal).
            - Y-axis: studentized residuals.
            - Marker size: Cook's distance (larger => more influential).

        How to interpret:
            - High leverage + large residual magnitude: potentially influential.
            - Very large Cook's distance: the fitted model may be sensitive to
              that observation (common heuristic: :math:`D_i > 4/n`).

        Guidance:
            Influential points are not automatically "bad" data. Investigate
            for data errors, consider robust alternatives, and/or run a
            sensitivity analysis with/without those points.
        """
        from ama_tlbx.plotting.regression_plots import plot_influence  # noqa: PLC0415

        return plot_influence(self, **kwargs)

    def plot_residual_diags(
        self,
        predictors: list[str] | None = None,
        *,
        max_cols: int = 4,
        figsize: tuple[int, int] = (12, 10),
        pred_figsize: tuple[int, int] | None = None,
    ) -> tuple[Figure, Figure | None, Figure]:
        """Plot a standard suite of residual diagnostics.

        The main 2x2 grid includes:
            - Residuals vs fitted (linearity & homoscedasticity).
            - Scale-location (spread of studentized residuals vs fitted).
            - Normal Q-Q plot (normality/tails/outliers).
            - Residual histogram (distribution shape).

        If ``predictors`` is provided (or defaults to all predictors), an
        additional figure shows residuals vs each predictor to help detect
        nonlinearity and variance patterns tied to specific covariates.

        Finally, an influence plot is produced to highlight high-leverage and
        high-influence observations.
        """
        resid = self.residuals
        fitted = self.fitted
        influence = self.model.get_influence()
        student_resid = influence.resid_studentized_internal

        fig_main, axes = plt.subplots(2, 2, figsize=figsize)
        sns.residplot(x=fitted, y=resid, lowess=True, ax=axes[0, 0], color="tab:blue")
        axes[0, 0].set_title("Residuals vs Fitted")
        axes[0, 0].set_xlabel("Fitted values")
        axes[0, 0].set_ylabel("Residuals")

        sns.scatterplot(
            x=fitted,
            y=np.sqrt(np.abs(student_resid)),
            ax=axes[0, 1],
            color="tab:orange",
        )
        axes[0, 1].set_title("Scale-Location (sqrt|studentized residuals|)")
        axes[0, 1].set_ylabel("sqrt(|studentized resid|)")

        sm.qqplot(student_resid, line="45", fit=True, ax=axes[1, 0])
        axes[1, 0].set_title("QQ plot (studentized residuals)")

        sns.histplot(resid, kde=True, ax=axes[1, 1], color="tab:green")
        axes[1, 1].set_title("Residual distribution")
        axes[1, 1].set_xlabel("Residuals")

        plt.tight_layout()
        plt.show()

        all_preds = [c for c in self.design_matrix.columns if c not in _INTERCEPT_COLS]
        if predictors is None:
            preds_to_plot = all_preds
        else:
            preds_to_plot = [p for p in predictors if p in self.design_matrix.columns and p not in _INTERCEPT_COLS]
        fig_pred: Figure | None = None
        if preds_to_plot:
            n_cols = max(1, max_cols)
            n_rows = int(np.ceil(len(preds_to_plot) / n_cols))
            if pred_figsize is None:
                pred_figsize = (max(8, int(3.5 * n_cols)), max(3, int(3.0 * n_rows)))
            fig_pred, axes = plt.subplots(n_rows, n_cols, figsize=pred_figsize)
            axes_list = np.atleast_1d(axes).ravel()
            for ax, pred in zip(axes_list, preds_to_plot, strict=False):
                sns.scatterplot(x=self.design_matrix[pred], y=resid, ax=ax)
                ax.set_title(f"Residuals vs {pred}")
            for ax in axes_list[len(preds_to_plot) :]:
                ax.set_visible(False)
            plt.tight_layout()
            plt.show()

        fig_influence = influence_plot(self.model, criterion="cooks")
        plt.tight_layout()
        plt.show()
        return fig_main, fig_pred, fig_influence

    def plot_predictions(
        self,
        feature: str,
        df: pd.DataFrame | None = None,
        **kwargs: object,
    ) -> Axes:
        """Plot fitted line and confidence band for a single predictor."""
        frame = getattr(getattr(self.model, "data", None), "frame", None)
        if df is None:
            if isinstance(frame, pd.DataFrame):
                df = frame
            else:
                df = self.design_matrix.copy()
                target = getattr(self.model.model, "endog_names", None)
                if target and target not in df.columns:
                    df[target] = self.y
        return pred_plot(self.model, feature, df, **kwargs)


def fit_ols_formula(
    df: pd.DataFrame,
    *,
    rhs: str,
    target_col: str = LECol.TARGET,
    cv_folds: int | None = None,
    shuffle_cv: bool = False,
    random_state: int | None = None,
) -> RegressionResult:
    """Fit OLS using a Patsy formula and return a full diagnostics bundle.

    This uses ``statsmodels.formula.api.ols`` with a formula like
    ``"{target} ~ {rhs}"``, allowing transformations and categorical encoding
    via Patsy.
    """
    model = smf.ols(f"{target_col} ~ {rhs}", data=df).fit()
    return diagnose_ols(
        model,
        cv_folds=cv_folds,
        shuffle_cv=shuffle_cv,
        random_state=random_state,
    )


def fit_ols_design(
    design_matrix_Xy: pd.DataFrame,
    *,
    target_col: str = LECol.TARGET,
    add_intercept: bool | None = None,
    cv_folds: int | None = None,
    shuffle_cv: bool = False,
    random_state: int | None = None,
) -> RegressionResult:
    """Fit OLS on a design matrix (including target) and return diagnostics.

    Args:
        design_matrix_Xy: DataFrame containing predictors and the target column.
        target_col: Name of the target column contained in ``design_matrix_Xy``.
        add_intercept: Whether to add an intercept column. If ``None``, the
            function adds one only when no intercept column is present.
    """
    x_matrix = design_matrix_Xy.drop(columns=[target_col]).copy()
    y = design_matrix_Xy[target_col].copy()
    if add_intercept is None:
        add_intercept = not any(col in x_matrix.columns for col in _INTERCEPT_COLS)
    if add_intercept:
        x_matrix = sm.add_constant(x_matrix, has_constant="add")

    model = sm.OLS(y.astype(float), x_matrix.astype(float)).fit()
    return diagnose_ols(
        model,
        cv_folds=cv_folds,
        shuffle_cv=shuffle_cv,
        random_state=random_state,
    )


def diagnose_ols(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    *,
    cv_folds: int | None = None,
    shuffle_cv: bool = False,
    random_state: int | None = None,
) -> RegressionResult:
    """Compute full diagnostics for an already-fitted OLS model.

    The returned object includes fitted values, residuals, summary metrics, and
    assumption checks. Diagnostics are descriptive; small p-values suggest
    potential issues but should be evaluated alongside plots and context.
    """
    design_matrix = design_matrix_from_model(model)
    predictions = pd.Series(model.fittedvalues, index=design_matrix.index)
    residuals = pd.Series(model.resid, index=design_matrix.index)
    y = pd.Series(model.model.endog, index=design_matrix.index, name=getattr(model.model, "endog_names", None))

    metrics = compute_metrics(
        model=model,
        y_true=y,
        y_pred=predictions,
        design_matrix=design_matrix,
        cv_folds=cv_folds,
        shuffle_cv=shuffle_cv,
        random_state=random_state,
    )
    assumptions = compute_assumptions(model=model, design_matrix=design_matrix)

    return RegressionResult(
        model=model,
        design_matrix=design_matrix,
        y=y,
        metrics=metrics,
        assumptions=assumptions,
        residuals=residuals,
        predictions=predictions,
    )


def design_matrix_from_model(
    model: sm.regression.linear_model.RegressionResultsWrapper,
) -> pd.DataFrame:
    """Return the fitted design matrix used by a statsmodels OLS result.

    This mirrors the model's exogenous matrix (including any encoded terms).
    """
    row_labels = getattr(getattr(model.model, "data", None), "row_labels", None)
    index = row_labels if row_labels is not None else None
    return pd.DataFrame(model.model.exog, columns=model.model.exog_names, index=index)


def design_matrix_for_data(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a Patsy design matrix for new data, falling back to fitted exog.

    This is mainly useful for formula-based models where transformations or
    categorical encodings are governed by Patsy ``design_info``.
    """
    design_info = getattr(getattr(model.model, "data", None), "design_info", None)
    if design_info is None:
        return design_matrix_from_model(model)
    matrices = build_design_matrices([design_info], df, return_type="dataframe")
    return matrices[0]


@dataclass
class EvalMetrics:
    """Evaluation metrics computed on a holdout dataset."""

    rmse: float
    mae: float
    r2: float
    n_obs: float
    label: str | None = None


def evaluate_model(
    diag: RegressionResult,
    df: pd.DataFrame,
    *,
    label: str | None = None,
    target_col: str = LECol.TARGET,
) -> EvalMetrics:
    """Evaluate a fitted model on a new dataset using aligned design matrices."""
    exog = design_matrix_for_data(diag.model, df)
    pred = pd.Series(diag.model.predict(exog), index=exog.index)
    y = df[target_col].astype(float)
    mask = y.notna() & pred.notna()
    y = y.loc[mask]
    pred = pred.loc[mask]
    rmse = float(np.sqrt(((y - pred) ** 2).mean()))
    mae = float((y - pred).abs().mean())
    r2 = float(1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return EvalMetrics(
        label=label,
        rmse=rmse,
        mae=mae,
        r2=r2,
        n_obs=float(len(y)),
    )


def _drop_intercept_cols(design_matrix: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [col for col in _INTERCEPT_COLS if col in design_matrix.columns]
    return design_matrix.drop(columns=cols_to_drop) if cols_to_drop else design_matrix


def _linear_regression_for_design(design_matrix: pd.DataFrame) -> LinearRegression:
    has_intercept = any(col in design_matrix.columns for col in _INTERCEPT_COLS)
    return LinearRegression(fit_intercept=not has_intercept)


def compute_vif(design_matrix: pd.DataFrame) -> pd.Series:
    r"""Compute VIF per regressor (intercept excluded).

    :math:`VIF_j = \frac{1}{1 - R_j^2}`, where :math:`R_j^2` comes from regressing
    predictor :math:`j` on all other predictors. For a single regressor (after
    excluding the intercept), VIF is defined as 1.0 because the auxiliary
    regression has :math:`R^2 = 0`.
    """
    x = _drop_intercept_cols(design_matrix)
    if x.shape[1] == 0:
        return pd.Series(dtype=float)
    if x.shape[1] == 1:
        return pd.Series({x.columns[0]: 1.0})
    return pd.Series(
        {col: float(variance_inflation_factor(x.values, idx)) for idx, col in enumerate(x.columns)},
    )


def compute_cv_scores(
    design_matrix: pd.DataFrame,
    y: pd.Series,
    *,
    cv_folds: int,
    shuffle: bool = False,
    random_state: int | None = None,
) -> list[float]:
    """Compute cross-validation RMSE scores for a linear regression baseline."""
    lr = _linear_regression_for_design(design_matrix)
    splitter = KFold(
        n_splits=cv_folds,
        shuffle=shuffle,
        random_state=(random_state if shuffle else None),
    )
    scores = list(
        cross_val_score(
            lr,
            design_matrix,
            y,
            cv=splitter,
            scoring="neg_root_mean_squared_error",
            error_score="raise",
        ),
    )
    return list(-np.asarray(scores))


def compute_metrics(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    y_true: pd.Series,
    y_pred: pd.Series,
    design_matrix: pd.DataFrame,
    *,
    cv_folds: int | None = None,
    shuffle_cv: bool = False,
    random_state: int | None = None,
) -> MetricsResult:
    """Compute fit, information criteria, and optional CV scores.

    Metrics are computed on in-sample residuals, with optional K-fold CV to
    estimate generalization (mean over folds).
    """
    # Some sklearn versions lack `squared` kwarg; compute RMSE manually for compatibility.
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    mape = None if (y_true == 0).any() else float(mean_absolute_percentage_error(y_true, y_pred))

    cv_scores: list[float] | None = None
    cv_rmse: float | None = None
    if cv_folds and cv_folds > 1:
        cv_scores = compute_cv_scores(
            design_matrix,
            y_true,
            cv_folds=cv_folds,
            shuffle=shuffle_cv,
            random_state=random_state,
        )
        cv_rmse = float(np.asarray(cv_scores).mean())

    return MetricsResult(
        r2=float(r2_score(y_true, y_pred)),
        adj_r2=float(model.rsquared_adj) if hasattr(model, "rsquared_adj") else None,
        rmse=rmse,
        mae=mae,
        mape=mape,
        aic=float(model.aic) if hasattr(model, "aic") else None,
        bic=float(model.bic) if hasattr(model, "bic") else None,
        loglik=float(model.llf) if hasattr(model, "llf") else None,
        n_obs=float(model.nobs) if hasattr(model, "nobs") else None,
        cv_scores=cv_scores,
        cv_rmse=cv_rmse,
    )


def compute_assumptions(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    design_matrix: pd.DataFrame,
) -> AssumptionCheckResult:
    """Run key regression assumption checks and return structured results.

    Includes autocorrelation (Durbin-Watson), normality (Jarque-Bera,
    Shapiro-Wilk / Anderson-Darling), heteroscedasticity (Breusch-Pagan, White),
    and collinearity/influence diagnostics (condition number, VIF, leverage,
    Cook's distance).
    """
    resid = pd.Series(model.resid, index=design_matrix.index)

    jb_stat, jb_pvalue, _, _ = jarque_bera(resid)

    if resid.shape[0] > _SHAPIRO_MAX_N:
        # Shapiro-Wilk warns above 5k; fall back to Anderson-Darling.
        shapiro_stat, shapiro_pvalue = sm_diagnostic.normal_ad(resid)
    else:
        shapiro_stat, shapiro_pvalue = stats.shapiro(resid)

    bp_stat, bp_pvalue, _, _ = sm_diagnostic.het_breuschpagan(
        resid,
        design_matrix.values,
    )
    white_stat, white_pvalue, _, _ = sm_diagnostic.het_white(
        resid,
        design_matrix.values,
    )

    vif = compute_vif(design_matrix)

    influence = model.get_influence()
    leverage = influence.hat_matrix_diag
    cooks_distance = influence.cooks_distance[0]

    return AssumptionCheckResult(
        durbin_watson=float(durbin_watson(resid)),
        jarque_bera_statistic=float(jb_stat),
        jarque_bera_pvalue=float(jb_pvalue),
        shapiro_statistic=float(shapiro_stat),
        shapiro_pvalue=float(shapiro_pvalue),
        breusch_pagan_statistic=float(bp_stat),
        breusch_pagan_pvalue=float(bp_pvalue),
        white_statistic=float(white_stat),
        white_pvalue=float(white_pvalue),
        condition_number=float(np.linalg.cond(design_matrix.values)),
        vif=vif,
        leverage=leverage,
        cooks_distance=cooks_distance,
    )


def compare_models(models: dict[str, sm.regression.linear_model.RegressionResultsWrapper]) -> pd.DataFrame:
    """Tabulate AIC/BIC, adj R², and RMSE for multiple fitted OLS models.

    Information criteria are most meaningful for comparing models fit to the
    same response on the same data; lower values indicate a better trade-off of
    fit and complexity.
    """
    rows = []
    for name, res in models.items():
        rows.append(
            {
                "model": name,
                "aic": float(res.aic),
                "bic": float(res.bic),
                "adj_r2": float(res.rsquared_adj),
                "rmse": float(np.sqrt(res.mse_resid)),
            },
        )
    return pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)


def stepwise_aic(
    data: pd.DataFrame,
    target_col: str,
    base_terms: list[str],
    candidates: list[str],
    *,
    threshold: float = 1.0,
) -> tuple[list[str], sm.regression.linear_model.RegressionResultsWrapper]:
    """Forward stepwise search that keeps adding terms while AIC improves by ``threshold``.

    This is a greedy heuristic: it evaluates candidate additions one at a time
    and does not guarantee a global optimum. Use with caution due to multiple
    testing and selection bias.
    """
    current = base_terms.copy()
    base_formula = f"{target_col} ~ " + (" + ".join(current) if current else "1")
    best_res = smf.ols(base_formula, data=data).fit()
    remaining = candidates.copy()
    improved = True
    while improved and remaining:
        improved = False
        scores: list[tuple[float, str, sm.regression.linear_model.RegressionResultsWrapper]] = []
        for cand in remaining:
            formula = f"{target_col} ~ " + " + ".join([*current, cand])
            res = smf.ols(formula=formula, data=data).fit()
            scores.append((res.aic, cand, res))
        scores.sort(key=itemgetter(0))
        best_aic, best_cand, best_candidate_res = scores[0]
        if best_aic + threshold < best_res.aic:
            current.append(best_cand)
            remaining.remove(best_cand)
            best_res = best_candidate_res
            improved = True
    return current, best_res


@overload
def collect_split_metrics(
    name: str,
    model: sm.regression.linear_model.RegressionResultsWrapper,
    train: tuple[pd.DataFrame, pd.Series],
    test: tuple[pd.DataFrame, pd.Series],
) -> dict[str, float | str]: ...


@overload
def collect_split_metrics(
    name: str,
    model: sm.regression.linear_model.RegressionResultsWrapper,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float | str]: ...


def collect_split_metrics(
    name: str,
    model: sm.regression.linear_model.RegressionResultsWrapper,
    *args: object,
) -> dict[str, float | str]:
    """Train/test R² and RMSE summary for quick model comparison.

    The test :math:`R^2` is computed as :math:`1 - SSE/SST` on the holdout set.
    """
    n_args = len(args)
    train_test_pair_args = 2
    split_arrays_args = 4

    if n_args == train_test_pair_args:
        train, test = args
        x_train, y_train = train  # type: ignore[misc]
        x_test, y_test = test  # type: ignore[misc]
    elif n_args == split_arrays_args:
        x_train, x_test, y_train, y_test = args  # type: ignore[misc]
    else:
        raise TypeError("Expected either (train, test) or (x_train, x_test, y_train, y_test).")

    tr_pred = model.predict(sm.add_constant(x_train))  # type: ignore[arg-type]
    te_pred = model.predict(sm.add_constant(x_test))  # type: ignore[arg-type]
    train_rmse = float(np.sqrt(np.mean((y_train - tr_pred) ** 2)))  # type: ignore[operator]
    test_rmse = float(np.sqrt(np.mean((y_test - te_pred) ** 2)))  # type: ignore[operator]
    test_r2 = float(  # type: ignore[arg-type, operator]
        1 - np.sum((y_test - te_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2),
    )
    return {
        "model": name,
        "train_r2": float(model.rsquared),
        "test_r2": test_r2,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "adj_r2": float(model.rsquared_adj),
        "aic": float(model.aic),
    }


def cooksd_contours(
    ax: plt.Axes,
    leverage: np.ndarray,
    *,
    levels: tuple[float, ...] = (0.5, 1.0),
) -> None:
    """Overlay Cook's distance contour lines on a leverage vs studentized residual plot.

    Contours help visualize influence levels; points outside a contour are more
    influential under that Cook's distance threshold.
    """
    x = leverage
    for level in levels:
        y = np.sqrt(level * (len(x) - 2)) * np.sqrt((1 - x) / x)
        ax.plot(x, y, ls="--", color="tab:red", linewidth=1)
        ax.plot(x, -y, ls="--", color="tab:red", linewidth=1)
        ax.annotate(f"Cook {level}", xy=(x.max(), y[-1]), fontsize=8, ha="right")
