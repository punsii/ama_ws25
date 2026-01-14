"""Model selection helpers for OLS regression workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from operator import itemgetter
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .ols_helper import MetricsResult, compute_metrics, design_matrix_from_model


@dataclass(frozen=True)
class SelectionStep:
    """Single step in a model selection path.

    Stores the fitted model and metrics for a particular term set. This keeps
    the selection logic independent from plotting/reporting and enables later
    inspection of how the criterion evolves across steps.
    """

    step: int
    terms: list[str]
    rhs: str
    model: sm.regression.linear_model.RegressionResultsWrapper
    metrics: MetricsResult
    cp: float | None = None


@dataclass(frozen=True)
class SelectionPathResult:
    """Results from a greedy model-selection path.

    The path is a sequence of models produced by forward, backward, or stepwise
    selection using a chosen criterion (AIC by default). Each step trades off
    goodness of fit against model complexity. Because selection is data-adaptive,
    results should be interpreted with caution: in-sample fit metrics tend to be
    optimistic and do not account for selection uncertainty.
    """

    steps: list[SelectionStep]
    criterion: str
    direction: str
    best_index: int

    def best_step(self) -> SelectionStep:
        """Return the best step according to the selection criterion."""
        return self.steps[self.best_index]

    def summary_table(self) -> pd.DataFrame:
        """Return a tidy summary table for plotting and reporting."""
        rows: list[dict[str, float | int | str | None]] = []
        for step in self.steps:
            rows.append(
                {
                    "step": step.step,
                    "n_terms": len(step.terms),
                    "rhs": step.rhs,
                    "aic": step.metrics.aic,
                    "bic": step.metrics.bic,
                    "adj_r2": step.metrics.adj_r2,
                    "rmse": step.metrics.rmse,
                    "cv_rmse": step.metrics.cv_rmse,
                    "cp": step.cp,
                },
            )
        return pd.DataFrame(rows).set_index("step")


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


def selection_path(  # noqa: C901, PLR0912, PLR0913, PLR0915
    data: pd.DataFrame,
    *,
    target_col: str,
    base_terms: list[str] | None,
    candidates: list[str],
    direction: Literal["forward", "backward", "stepwise"] = "forward",
    criterion: Literal["aic", "bic", "cp", "adj_r2", "cv_rmse"] = "aic",
    threshold: float = 1.0,
    cv_folds: int | None = None,
    shuffle_cv: bool = False,
    random_state: int | None = None,
) -> SelectionPathResult:
    """Run a greedy selection procedure and return the full path.

    Args:
        data: DataFrame containing target and candidate predictors.
        target_col: Name of the response variable.
        base_terms: Terms that are always included (e.g., confounders).
        candidates: Candidate terms to add/remove during selection.
        direction: "forward", "backward", or "stepwise" (both directions).
        criterion: Selection criterion. Lower is better except for adj_r2.
        threshold: Minimum improvement required to accept a step.
        cv_folds: Optional K-folds for computing CV RMSE at each step.
        shuffle_cv: Whether to shuffle during CV.
        random_state: Random seed used when shuffling CV splits.

    Returns:
        SelectionPathResult containing all visited steps.

    Notes:
        Selection is a greedy heuristic and does not guarantee a global optimum.
        Information criteria (AIC/BIC/Cp) balance fit and complexity; lower is
        preferred. Treat selected models as candidates and validate on held-out
        data to avoid selection bias.
    """
    direction = direction.lower()
    criterion = criterion.lower()
    if direction not in {"forward", "backward", "stepwise"}:
        raise ValueError("direction must be one of: forward, backward, stepwise")
    if criterion not in {"aic", "bic", "cp", "adj_r2", "cv_rmse"}:
        raise ValueError("criterion must be one of: aic, bic, cp, adj_r2, cv_rmse")
    if criterion == "cv_rmse" and (cv_folds is None or cv_folds <= 1):
        raise ValueError("cv_folds must be > 1 when using criterion='cv_rmse'")

    base_terms = list(base_terms or [])
    candidates = [term for term in candidates if term not in base_terms]

    def rhs_for(terms: list[str]) -> str:
        return " + ".join(terms) if terms else "1"

    full_model = None
    if criterion == "cp":
        full_rhs = rhs_for([*base_terms, *candidates])
        full_model = smf.ols(f"{target_col} ~ {full_rhs}", data=data).fit()

    def build_step(terms: list[str], step_index: int) -> SelectionStep:
        rhs = rhs_for(terms)
        model = smf.ols(f"{target_col} ~ {rhs}", data=data).fit()
        design_matrix = design_matrix_from_model(model)
        y = pd.Series(model.model.endog, index=design_matrix.index, name=getattr(model.model, "endog_names", None))
        y_pred = pd.Series(model.fittedvalues, index=design_matrix.index)
        metrics = compute_metrics(
            model=model,
            y_true=y,
            y_pred=y_pred,
            design_matrix=design_matrix,
            cv_folds=cv_folds,
            shuffle_cv=shuffle_cv,
            random_state=random_state,
        )
        cp = compute_mallows_cp(full_model, model) if full_model is not None else None
        return SelectionStep(step=step_index, terms=list(terms), rhs=rhs, model=model, metrics=metrics, cp=cp)

    def score(step: SelectionStep) -> float:  # noqa: C901
        if criterion == "aic":
            if step.metrics.aic is None:
                raise ValueError("AIC not available for this model.")
            return float(step.metrics.aic)
        if criterion == "bic":
            if step.metrics.bic is None:
                raise ValueError("BIC not available for this model.")
            return float(step.metrics.bic)
        if criterion == "cp":
            if step.cp is None:
                raise ValueError("Cp not computed for this model.")
            return float(step.cp)
        if criterion == "adj_r2":
            if step.metrics.adj_r2 is None:
                raise ValueError("Adjusted R^2 not available for this model.")
            return float(step.metrics.adj_r2)
        if criterion == "cv_rmse":
            if step.metrics.cv_rmse is None:
                raise ValueError("CV RMSE not computed for this model.")
            return float(step.metrics.cv_rmse)
        raise ValueError(f"Unsupported criterion '{criterion}'.")

    def better(candidate: float, current: float) -> bool:
        if criterion == "adj_r2":
            return candidate > current + threshold
        return candidate < current - threshold

    steps: list[SelectionStep] = []
    current_terms = [*base_terms, *candidates] if direction == "backward" else base_terms.copy()

    steps.append(build_step(current_terms, step_index=0))

    while True:
        candidate_steps: list[SelectionStep] = []
        if direction in {"forward", "stepwise"}:
            for term in candidates:
                if term in current_terms:
                    continue
                candidate_steps.append(build_step([*current_terms, term], step_index=-1))
        if direction in {"backward", "stepwise"} and len(current_terms) > len(base_terms):
            for term in list(current_terms):
                if term in base_terms:
                    continue
                reduced_terms = [t for t in current_terms if t != term]
                candidate_steps.append(build_step(reduced_terms, step_index=-1))

        if not candidate_steps:
            break

        best_candidate = max(candidate_steps, key=score) if criterion == "adj_r2" else min(candidate_steps, key=score)

        if better(score(best_candidate), score(steps[-1])):
            current_terms = best_candidate.terms
            steps.append(replace(best_candidate, step=len(steps)))
        else:
            break

    scores = [score(step) for step in steps]
    best_index = int(np.argmax(scores)) if criterion == "adj_r2" else int(np.argmin(scores))

    return SelectionPathResult(
        steps=steps,
        criterion=criterion,
        direction=direction,
        best_index=best_index,
    )


def collect_selection_paths(
    paths: dict[str, SelectionPathResult],
    *,
    best_only: bool = False,
) -> pd.DataFrame:
    """Collect selection-path summaries into a tidy DataFrame.

    Args:
        paths: Mapping from label to selection-path result.
        best_only: If True, keep only the best step per path.

    Returns:
        DataFrame with per-step metrics and path metadata.
    """
    tables = []
    for label, path in paths.items():
        table = path.summary_table().reset_index()
        table = table.assign(
            label=label,
            direction=path.direction,
            criterion=path.criterion,
            is_best=lambda d: d["step"] == path.best_index,
        )
        if best_only:
            table = table.loc[table["is_best"]]
        tables.append(table)
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def forward_selection(  # noqa: PLR0913
    data: pd.DataFrame,
    *,
    target_col: str,
    base_terms: list[str] | None,
    candidates: list[str],
    criterion: Literal["aic", "bic", "cp", "adj_r2", "cv_rmse"] = "aic",
    threshold: float = 1.0,
    cv_folds: int | None = None,
    shuffle_cv: bool = False,
    random_state: int | None = None,
) -> SelectionPathResult:
    """Convenience wrapper for forward selection."""
    return selection_path(
        data,
        target_col=target_col,
        base_terms=base_terms,
        candidates=candidates,
        direction="forward",
        criterion=criterion,
        threshold=threshold,
        cv_folds=cv_folds,
        shuffle_cv=shuffle_cv,
        random_state=random_state,
    )


def backward_selection(  # noqa: PLR0913
    data: pd.DataFrame,
    *,
    target_col: str,
    base_terms: list[str] | None,
    candidates: list[str],
    criterion: Literal["aic", "bic", "cp", "adj_r2", "cv_rmse"] = "aic",
    threshold: float = 1.0,
    cv_folds: int | None = None,
    shuffle_cv: bool = False,
    random_state: int | None = None,
) -> SelectionPathResult:
    """Convenience wrapper for backward selection."""
    return selection_path(
        data,
        target_col=target_col,
        base_terms=base_terms,
        candidates=candidates,
        direction="backward",
        criterion=criterion,
        threshold=threshold,
        cv_folds=cv_folds,
        shuffle_cv=shuffle_cv,
        random_state=random_state,
    )


def stepwise_selection(  # noqa: PLR0913
    data: pd.DataFrame,
    *,
    target_col: str,
    base_terms: list[str] | None,
    candidates: list[str],
    criterion: Literal["aic", "bic", "cp", "adj_r2", "cv_rmse"] = "aic",
    threshold: float = 1.0,
    cv_folds: int | None = None,
    shuffle_cv: bool = False,
    random_state: int | None = None,
) -> SelectionPathResult:
    """Convenience wrapper for stepwise selection in both directions."""
    return selection_path(
        data,
        target_col=target_col,
        base_terms=base_terms,
        candidates=candidates,
        direction="stepwise",
        criterion=criterion,
        threshold=threshold,
        cv_folds=cv_folds,
        shuffle_cv=shuffle_cv,
        random_state=random_state,
    )


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


__all__ = [
    "SelectionPathResult",
    "SelectionStep",
    "backward_selection",
    "collect_selection_paths",
    "compare_models",
    "compute_mallows_cp",
    "forward_selection",
    "selection_path",
    "stepwise_aic",
    "stepwise_selection",
]
