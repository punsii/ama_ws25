from dataclasses import asdict, dataclass, field
import re
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from ..data import LECol
from .model_selection import SelectionPathResult, selection_path
from .ols_helper import (
    EvalMetrics,
    MetricsResult,
    RegressionResult,
    evaluate_model,
    fit_ols,
)


@dataclass
class ModelEntry:
    """Typed model registry entry for reporting workflows."""

    name: str
    rhs: str
    diag: RegressionResult
    metrics: MetricsResult
    eval_metrics: EvalMetrics | None = None
    eval_metrics_by_label: dict[str, EvalMetrics] = field(default_factory=dict)
    selection_path: SelectionPathResult | None = None


@dataclass
class ModelRegistry:
    """Registry to cache fitted models and compare their diagnostics."""

    eval_year: int = 2011
    models: dict[str, ModelEntry] = field(default_factory=dict)

    def add(self, entry: ModelEntry, *, overwrite: bool = False) -> None:
        """Add an entry to the registry (optionally overwriting by name)."""
        name = entry.name
        if name in self.models and not overwrite:
            raise KeyError(f"Model '{name}' already exists in registry.")
        self.models[name] = entry

    def get(self, name: str) -> ModelEntry:
        """Retrieve a model entry by name."""
        if name not in self.models:
            raise KeyError(f"Unknown model '{name}'.")
        return self.models[name]

    def fit(
        self,
        df: pd.DataFrame,
        *,
        rhs: str | None = None,
        name: str | None = None,
        target_col: str = LECol.TARGET,
        cv_folds: int | None = None,
        shuffle_cv: bool = False,
        random_state: int | None = None,
        refit: bool = False,
    ) -> RegressionResult:
        """Fit an OLS formula model and cache it by name.

        Args:
            df: DataFrame with training data including all features and target.
            rhs: RHS formula string (if None, use all columns except target).
            name: Unique name for the model in the registry.
            target_col: Name of the target column in `df`.
            cv_folds: Number of CV folds for metrics (if None, no CV).
            shuffle_cv: Whether to shuffle data before CV splitting.
            random_state: Random state for reproducibility.
            refit: If True, refit even if model with `name` exists.
        """
        name = name or f"model_{len(self.models) + 1}"
        if name in self.models and not refit:
            return self.models[name].diag
        if rhs is not None:
            diag = fit_ols(
                df,
                rhs=rhs,
                target_col=target_col,
                cv_folds=cv_folds,
                shuffle_cv=shuffle_cv,
                random_state=random_state,
            )
        else:
            diag = fit_ols(
                design_matrix_Xy=df,
                target_col=target_col,
                cv_folds=cv_folds,
                shuffle_cv=shuffle_cv,
                random_state=random_state,
            )

        self.add(
            ModelEntry(
                name=name,
                rhs=rhs,
                diag=diag,
                metrics=diag.metrics,
            ),
            overwrite=True,
        )

        return diag

    def evaluate_on(
        self,
        name: str,
        df: pd.DataFrame,
        *,
        label: str | None = None,
        target_col: str = LECol.TARGET,
    ) -> EvalMetrics:
        """Evaluate a registered model on new data and cache the results.

        Args:
            name: Name of the registered model to evaluate.
            df: DataFrame with evaluation data including all features and target.
            label: Optional label for the evaluation metrics (if None, use year).
            target_col: Name of the target column in `df`.
        """
        entry = self.get(name)
        label = label or f"year{self.eval_year}"
        metrics = evaluate_model(
            entry.diag,
            df,
            label=label,
            target_col=target_col,
        )
        entry.eval_metrics = metrics
        entry.eval_metrics_by_label[label] = metrics
        return metrics

    def fit_stepwise(  # noqa: PLR0913
        self,
        df: pd.DataFrame,
        *,
        base_terms: list[str] | None,
        candidates: list[str],
        name: str | None = None,
        target_col: str = LECol.TARGET,
        direction: Literal["forward", "backward", "stepwise"] = "forward",
        criterion: Literal["aic", "bic", "cp", "adj_r2", "cv_rmse"] = "aic",
        threshold: float = 1.0,
        cv_folds: int | None = None,
        shuffle_cv: bool = False,
        random_state: int | None = None,
        refit: bool = False,
    ) -> RegressionResult:
        """Run a stepwise selection path and cache the best model.

        Stepwise selection is a greedy variable-selection procedure that trades
        off goodness of fit against model complexity using a criterion such as
        AIC/BIC (or Cp/adjusted R^2). AIC/BIC compare models via the maximized
        log-likelihood and the number of estimated parameters (with BIC using
        an n-dependent penalty). Lower AIC/BIC/Cp indicates a preferred model;
        BIC penalizes complexity more strongly than AIC for larger n. In
        forward selection, we start from the base model and add the candidate
        term that most improves the criterion; in backward selection, we start
        from the full model and remove the term that most improves the
        criterion; stepwise selection allows both add and drop moves each
        iteration. The search stops when no move clears the improvement
        threshold (or a prespecified test/criterion cutoff).

        Notes:
        - Greedy stepwise searches do not guarantee the globally optimal subset
          among all 2^K candidate models.
        - Global criteria can yield models with non-significant coefficients, so
          treat results as exploratory and validate with holdout/CV when
          possible.

        References:
        - [Wikipedia :: Stepwise regression](https://en.wikipedia.org/wiki/Stepwise_regression)
        - [Wikipedia :: Akaike information criterion](https://en.wikipedia.org/wiki/Akaike_information_criterion)
        - [Wikipedia :: Bayesian information criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion)
        - [Wikipedia :: Mallows's Cp](https://en.wikipedia.org/wiki/Mallows%27s_Cp)
        """
        name = name or f"model_{len(self.models) + 1}"
        if name in self.models and not refit:
            return self.models[name].diag

        path = selection_path(
            data=df,
            target_col=target_col,
            base_terms=base_terms,
            candidates=candidates,
            direction=direction,
            criterion=criterion,
            threshold=threshold,
            cv_folds=cv_folds,
            shuffle_cv=shuffle_cv,
            random_state=random_state,
        )
        best_rhs = path.best_step().rhs
        diag = fit_ols(
            df,
            rhs=best_rhs,
            target_col=target_col,
            cv_folds=cv_folds,
            shuffle_cv=shuffle_cv,
            random_state=random_state,
        )

        self.add(
            ModelEntry(
                name=name,
                rhs=best_rhs,
                diag=diag,
                metrics=diag.metrics,
                selection_path=path,
            ),
            overwrite=True,
        )

        return diag

    def compare(self, *, sort_by: str = "aic") -> pd.DataFrame:
        """Return a comparison table for all cached models."""
        rows = []
        for entry in self.models.values():
            row = {"model": entry.name, "rhs": entry.rhs}
            row.update(asdict(entry.metrics))
            if entry.eval_metrics_by_label:
                for label, metrics in entry.eval_metrics_by_label.items():
                    prefix = label or f"year{self.eval_year}"
                    row.update(
                        {
                            f"{prefix}_rmse": metrics.rmse,
                            f"{prefix}_mae": metrics.mae,
                            f"{prefix}_r2": metrics.r2,
                            f"{prefix}_n_obs": metrics.n_obs,
                        },
                    )
            elif entry.eval_metrics is not None:
                prefix = entry.eval_metrics.label or f"year{self.eval_year}"
                row.update(
                    {
                        f"{prefix}_rmse": entry.eval_metrics.rmse,
                        f"{prefix}_mae": entry.eval_metrics.mae,
                        f"{prefix}_r2": entry.eval_metrics.r2,
                        f"{prefix}_n_obs": entry.eval_metrics.n_obs,
                    },
                )
            rows.append(row)
        df = pd.DataFrame(rows).set_index("model")
        if sort_by in df.columns:
            return df.sort_values(sort_by)
        return df

    def register_best_paths(
        self,
        df: pd.DataFrame,
        *,
        paths: dict[str, SelectionPathResult],
        target_col: str = LECol.TARGET,
        name_prefix: str = "best",
        cv_folds: int | None = None,
        shuffle_cv: bool = False,
        random_state: int | None = None,
        refit: bool = True,
    ) -> pd.DataFrame:
        """Fit and register the best step from each selection path.

        Returns a tidy mapping table that links each registry model name to the
        originating selection path metadata.
        """

        def _slugify(label: str) -> str:
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower()).strip("_")
            return slug or "model"

        rows: list[dict[str, object]] = []
        for label, path in paths.items():
            slug = _slugify(label)
            name = f"{name_prefix}_{slug}" if name_prefix else slug
            step = path.best_step()
            diag = self.fit(
                df,
                rhs=step.rhs,
                name=name,
                target_col=target_col,
                cv_folds=cv_folds,
                shuffle_cv=shuffle_cv,
                random_state=random_state,
                refit=refit,
            )
            entry = self.get(name)
            entry.selection_path = path
            rows.append(
                {
                    "model": name,
                    "label": label,
                    "direction": path.direction,
                    "criterion": path.criterion,
                    "step": int(step.step),
                    "n_terms": int(len(step.terms)),
                    "rhs": step.rhs,
                    "aic": diag.metrics.aic,
                    "bic": diag.metrics.bic,
                    "adj_r2": diag.metrics.adj_r2,
                    "rmse": diag.metrics.rmse,
                    "cv_rmse": diag.metrics.cv_rmse,
                    "cp": step.cp,
                },
            )
        return pd.DataFrame(rows)

    def evaluate_all(
        self,
        df: pd.DataFrame,
        *,
        label: str | None = None,
        target_col: str = LECol.TARGET,
    ) -> dict[str, EvalMetrics]:
        """Evaluate every registered model on a new dataset."""
        results: dict[str, EvalMetrics] = {}
        for name in self.models:
            results[name] = self.evaluate_on(
                name,
                df,
                label=label,
                target_col=target_col,
            )
        return results

    def assumptions_table(self, *, names: Iterable[str] | None = None) -> pd.DataFrame:
        """Return a tidy table of assumption-test statistics for registered models."""
        entries = (
            [self.get(name) for name in names] if names is not None else list(self.models.values())
        )
        rows: list[dict[str, float | str]] = []
        for entry in entries:
            assumptions = entry.diag.assumptions
            vif = assumptions.vif
            max_vif = float(vif.max()) if getattr(vif, "size", 0) else float("nan")
            rows.append(
                {
                    "model": entry.name,
                    "durbin_watson": float(assumptions.durbin_watson),
                    "jarque_bera_pvalue": float(assumptions.jarque_bera_pvalue),
                    "shapiro_pvalue": float(assumptions.shapiro_pvalue),
                    "breusch_pagan_pvalue": float(assumptions.breusch_pagan_pvalue),
                    "white_pvalue": float(assumptions.white_pvalue),
                    "condition_number": float(assumptions.condition_number),
                    "max_vif": max_vif,
                    "max_leverage": float(np.max(assumptions.leverage)),
                    "max_cooks": float(np.max(assumptions.cooks_distance)),
                },
            )
        return pd.DataFrame(rows).set_index("model")

    def __iter__(self):
        return iter(self.models.values())

    def __len__(self) -> int:
        return len(self.models)
