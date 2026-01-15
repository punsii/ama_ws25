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
from typing import TYPE_CHECKING, Literal, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
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

    from ama_tlbx.analysis.outlier_detector import OutlierDetectionResult


_INTERCEPT_COLS = ("Intercept", "const")
_SHAPIRO_MAX_N = 5000
_METRICS_FIT_FIELDS = ("r2", "adj_r2", "rmse", "mae", "mape", "aic", "bic", "aicc", "mdl")


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
    - :math:`\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n-k-1}`
    - :math:`\text{MDL} = -\log L + \frac{k}{2}\log n`

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

    aicc: float | None
    r"""Small-sample corrected AIC (AICc).

    :math:`\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n-k-1}`. Returns ``None``
    when :math:`n \le k+1`.
    """

    mdl: float | None
    r"""Minimum Description Length (MDL) criterion.

    :math:`\text{MDL} = -\log L + \frac{k}{2}\log n`. Lower is preferred in
    model comparison.
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

        fit_fields = [name for name in _METRICS_FIT_FIELDS if hasattr(self, name)]
        fit_parts = [f"{name}={fmt(getattr(self, name))}" for name in fit_fields]
        fit_block = f"Fit[{', '.join(fit_parts)}]"
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


@dataclass(frozen=True)  # noqa: PLR0904
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
    def aicc(self) -> float:
        """Small-sample corrected AIC (AICc) of the fitted model."""
        return self.metrics.aicc or float("nan")

    @property
    def adj_r2(self) -> float:
        """Adjusted :math:`R^2` correcting for the number of regressors."""
        return self.metrics.adj_r2 or float("nan")

    @property
    def rmse(self) -> float:
        """Root mean squared error (RMSE) of the in-sample residuals."""
        return self.metrics.rmse

    @property
    def mdl(self) -> float:
        """Minimum Description Length (MDL) criterion of the fitted model."""
        return self.metrics.mdl or float("nan")

    @property
    def intercept(self) -> float:
        """Intercept coefficient from the fitted model.

        Uses the single intercept term present in the fitted design matrix
        (e.g., ``Intercept`` for formula-based models or ``const`` for
        design-matrix fits). Raises if no intercept or multiple intercept
        columns are found.
        """
        params = self.model.params
        intercept_cols = [col for col in _INTERCEPT_COLS if col in params.index]
        if len(intercept_cols) != 1:
            raise KeyError(
                "Intercept term not found (or ambiguous) in fitted model parameters.",
            )
        return float(params[intercept_cols[0]])

    def coef_and_intercept(self, term: str) -> tuple[float, float]:
        """Return (coefficient, intercept) for a given term in the model."""
        params = self.model.params
        if term not in params.index:
            raise KeyError(f"Unknown term '{term}' in fitted model parameters.")
        return float(params[term]), self.intercept

    def pearson_r(self, term: str) -> float:
        """Compute Pearson correlation between a term and the target.

        Uses the fitted design matrix column for ``term`` and the aligned target
        values used in the regression.
        """
        if term not in self.design_matrix.columns:
            raise KeyError(f"Unknown term '{term}' in design matrix.")
        x = pd.Series(self.design_matrix[term], index=self.design_matrix.index, name=term)
        y = pd.Series(self.y, index=self.design_matrix.index, name="y")
        return float(x.corr(y))

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

    def detect_residual_outliers(
        self,
        *,
        method: Literal["zscore", "iqr"] = "zscore",
        threshold: float = 3.0,
        use_studentized: bool = True,
    ) -> OutlierDetectionResult:
        """Detect outliers in the residuals using existing univariate detectors.

        This helper wraps the project's :class:`ZScoreOutlierDetector` and
        :class:`IQROutlierDetector` by building a temporary one-column
        :class:`DatasetView` from the residuals. By default it uses studentized
        residuals, which scale residual magnitude by an estimate of their
        variance and account for leverage.

        Args:
            method: Which detector to use ("zscore" or "iqr").
            threshold: Z-score cutoff or IQR multiplier depending on `method`.
            use_studentized: If True, use internally studentized residuals;
                otherwise use raw residuals.

        Returns:
            OutlierDetectionResult with a single-column outlier mask.

        Notes:
            - These detectors are univariate; they flag large residuals but do
              not diagnose multivariate leverage. Use Cook's distance alongside.
            - Outlier flags are diagnostic and should not be treated as
              automatic exclusions.
        """
        from ama_tlbx.analysis.outlier_detector import (  # noqa: PLC0415
            IQROutlierDetector,
            ZScoreOutlierDetector,
        )
        from ama_tlbx.data.views import DatasetView  # noqa: PLC0415

        if use_studentized:
            influence = self.model.get_influence()
            residuals = pd.Series(influence.resid_studentized_internal, index=self.residuals.index, name="residual")
            label = "Studentized residual"
        else:
            residuals = self.residuals.rename("residual")
            label = "Residual"

        view = DatasetView(
            df=pd.DataFrame({"residual": residuals}),
            pretty_by_col={"residual": label},
            numeric_cols=["residual"],
            target_col=None,
            is_standardized=None,
        )

        if method == "zscore":
            detector = ZScoreOutlierDetector(view, threshold=threshold)
        elif method == "iqr":
            detector = IQROutlierDetector(view, threshold=threshold)
        else:
            raise ValueError(f"Unknown outlier method '{method}'. Use 'zscore' or 'iqr'.")

        return detector.fit().result()

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
        from ama_tlbx.plotting.regression_plots import pred_plot

        return pred_plot(self.model, feature, df, **kwargs)


def _prepare_design_matrix(
    design_matrix_Xy: pd.DataFrame,
    *,
    target_col: str,
    add_intercept: bool | None,
) -> tuple[pd.DataFrame, pd.Series]:
    x_matrix = design_matrix_Xy.drop(columns=[target_col]).copy()
    y = design_matrix_Xy[target_col].copy()
    if add_intercept is None:
        add_intercept = not any(col in x_matrix.columns for col in _INTERCEPT_COLS)
    if add_intercept:
        x_matrix = sm.add_constant(x_matrix, has_constant="add")
    return x_matrix, y


def fit_ols(
    df: pd.DataFrame | None = None,
    *,
    rhs: str | None = None,
    design_matrix_Xy: pd.DataFrame | None = None,
    target_col: str = LECol.TARGET,
    add_intercept: bool | None = None,
    cv_folds: int | None = None,
    shuffle_cv: bool = False,
    random_state: int | None = None,
) -> RegressionResult:
    """Fit OLS via formula or design matrix and return diagnostics bundle."""
    if rhs is not None:
        if df is None:
            raise ValueError("df is required when rhs is provided.")
        model = smf.ols(f"{target_col} ~ {rhs}", data=df).fit()
    else:
        if design_matrix_Xy is None:
            raise ValueError("design_matrix_Xy is required when rhs is None.")
        x_matrix, y = _prepare_design_matrix(
            design_matrix_Xy,
            target_col=target_col,
            add_intercept=add_intercept,
        )
        model = sm.OLS(y.astype(float), x_matrix.astype(float)).fit()
    return diagnose_ols(
        model,
        cv_folds=cv_folds,
        shuffle_cv=shuffle_cv,
        random_state=random_state,
    )


def fit_ols_formula(
    df: pd.DataFrame,
    *,
    rhs: str,
    target_col: str = LECol.TARGET,
    cv_folds: int | None = None,
    shuffle_cv: bool = False,
    random_state: int | None = None,
) -> RegressionResult:
    """Fit OLS using a Patsy formula and return a full diagnostics bundle."""
    return fit_ols(
        df,
        rhs=rhs,
        target_col=target_col,
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
    """Fit OLS on a design matrix (including target) and return diagnostics."""
    return fit_ols(
        design_matrix_Xy=design_matrix_Xy,
        target_col=target_col,
        add_intercept=add_intercept,
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


def compute_mallows_cp(
    full_model: sm.regression.linear_model.RegressionResultsWrapper,
    model: sm.regression.linear_model.RegressionResultsWrapper,
) -> float:
    r"""Compute Mallows' :math:`C_p` for a candidate model.

    Uses the variance estimate from the full model to penalize model size:

    :math:`C_p = \frac{RSS}{\hat{\sigma}^2} + 2(p+1) - n`

    where :math:`p+1` is the number of parameters including the intercept,
    :math:`RSS` is the residual sum of squares of the candidate model, and
    :math:`\hat{\sigma}^2` is estimated from the full model. A well-fitting
    model often yields :math:`C_p \approx p+1` (no strong underfitting).
    """
    n = float(model.nobs)
    p = float(model.df_model) + 1.0
    rss = float(np.sum(model.resid**2))
    sigma2 = float(full_model.mse_resid)
    return rss / sigma2 + 2.0 * p - n


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
        exog_names = list(model.model.exog_names)
        exog = pd.DataFrame(index=df.index)
        for name in exog_names:
            if name in _INTERCEPT_COLS:
                exog[name] = 1.0
            else:
                if name not in df.columns:
                    raise KeyError(f"Column '{name}' missing from evaluation data.")
                exog[name] = df[name].astype(float)
        return exog
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
    design_info = getattr(getattr(diag.model.model, "data", None), "design_info", None)
    if design_info is not None:
        pred = pd.Series(diag.model.predict(df), index=df.index)
    else:
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


def _compute_cv_rmse_scores(
    design_matrix: pd.DataFrame,
    y_true: pd.Series,
    *,
    cv_folds: int | None,
    shuffle_cv: bool,
    random_state: int | None,
) -> tuple[list[float] | None, float | None]:
    if not cv_folds or cv_folds <= 1:
        return None, None
    lr = _linear_regression_for_design(design_matrix)
    splitter = KFold(
        n_splits=cv_folds,
        shuffle=shuffle_cv,
        random_state=(random_state if shuffle_cv else None),
    )
    cv_scores = cross_val_score(
        lr,
        design_matrix,
        y_true,
        cv=splitter,
        scoring="neg_root_mean_squared_error",
        error_score="raise",
    )
    cv_scores = np.asarray(cv_scores, dtype=float).tolist()
    cv_rmse = float(np.mean(cv_scores))
    return cv_scores, cv_rmse


def _extract_model_info(
    model: sm.regression.linear_model.RegressionResultsWrapper,
) -> tuple[float, float | None, float | None, float | None, float | None]:
    n_obs = float(model.nobs)
    loglik = float(model.llf)
    aic = float(model.aic)
    bic = float(model.bic)
    k_params = float(len(model.params))
    return n_obs, loglik, aic, bic, k_params


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
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    cv_scores, cv_rmse = _compute_cv_rmse_scores(
        design_matrix,
        y_true,
        cv_folds=cv_folds,
        shuffle_cv=shuffle_cv,
        random_state=random_state,
    )
    n_obs, loglik, aic, bic, k_params = _extract_model_info(model)

    aicc: float | None = None
    if aic is not None and k_params is not None:
        denom = n_obs - k_params - 1
        if denom > 0:
            aicc = float(aic + (2 * k_params * (k_params + 1)) / denom)

    mdl: float | None = None
    if loglik is not None and k_params is not None and n_obs > 0:
        mdl = float(-loglik + 0.5 * k_params * np.log(n_obs))

    return MetricsResult(
        r2=float(r2_score(y_true, y_pred)),
        adj_r2=float(model.rsquared_adj),
        rmse=rmse,
        mae=mae,
        mape=mape,
        aic=aic,
        bic=bic,
        aicc=aicc,
        mdl=mdl,
        loglik=loglik,
        n_obs=n_obs,
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
    try:
        white_stat, white_pvalue, _, _ = sm_diagnostic.het_white(
            resid,
            design_matrix.values,
        )
    except (AssertionError, np.linalg.LinAlgError, ValueError):
        white_stat, white_pvalue = float("nan"), float("nan")

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


def bootstrap_rmse(
    data: pd.DataFrame,
    *,
    rhs: str,
    target_col: str,
    n_bootstrap: int = 200,
    random_state: int | None = None,
) -> pd.Series:
    r"""Estimate RMSE variability via simple bootstrap resampling.

    The model is refit on each bootstrap sample. When out-of-bag (OOB)
    observations exist, RMSE is computed on OOB data; otherwise the bootstrap
    sample is used as a fallback. This provides a rough uncertainty estimate of
    predictive performance without requiring a separate holdout set and mirrors
    the bootstrap workflow discussed in the lecture.
    """
    rng = np.random.default_rng(random_state)
    n = data.shape[0]
    rmses: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        boot = data.iloc[idx]
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[idx] = False
        oob = data.loc[oob_mask]
        model = smf.ols(f"{target_col} ~ {rhs}", data=boot).fit()
        eval_df = oob if not oob.empty else boot
        pred = model.predict(eval_df)
        y = eval_df[target_col]
        rmse = float(np.sqrt(((y - pred) ** 2).mean()))
        rmses.append(rmse)
    return pd.Series(rmses, name="rmse")


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
