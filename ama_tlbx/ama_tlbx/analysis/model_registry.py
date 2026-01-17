import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ..data import LECol
from .model_selection import SelectionPathResult, selection_path
from .ols_helper import (
    EvalMetrics,
    MetricsResult,
    RegressionResult,
    diagnose_ols,
    evaluate_model,
    fit_ols,
)


@dataclass
class ModelEntry:
    """Typed model registry entry for reporting workflows."""

    name: str
    """Registry key identifying the model."""
    rhs: str
    """RHS of the fitted model formula."""
    diag: RegressionResult
    """Full diagnostics bundle returned by OLS fitting."""
    metrics: MetricsResult
    """Primary fit metrics extracted from the diagnostics."""
    eval_metrics_by_label: dict[str, EvalMetrics] = field(default_factory=dict)
    """Evaluation metrics keyed by a label (e.g., year2011)."""
    selection_path: SelectionPathResult | None = None
    """Selection-path metadata if the model originates from stepwise search."""
    data: pd.DataFrame | None = None
    """Training dataframe used to fit the model (if available)."""


@dataclass
class ModelRegistry:
    """Registry to cache fitted models and compare their diagnostics."""

    eval_year: int = 2011
    models: dict[str, ModelEntry] = field(default_factory=dict)

    def _default_label(self, label: str | None) -> str:
        return label or f"year{self.eval_year}"

    def _ensure_name(self, name: str | None) -> str:
        return name or f"model_{len(self.models) + 1}"

    def _register(
        self,
        *,
        name: str,
        rhs: str,
        diag: RegressionResult,
        selection_path: SelectionPathResult | None = None,
        data: pd.DataFrame | None = None,
    ) -> ModelEntry:
        entry = ModelEntry(
            name=name,
            rhs=rhs,
            diag=diag,
            metrics=diag.metrics,
            eval_metrics_by_label={},
            selection_path=selection_path,
            data=data,
        )
        self.add(entry, overwrite=True)
        return entry

    @staticmethod
    def _frame_from_model(model: object) -> pd.DataFrame | None:
        """Best-effort extraction of the training DataFrame from a statsmodels result."""
        data = getattr(getattr(model, "model", None), "data", None)
        return getattr(data, "frame", None) if data is not None else None

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
        name = self._ensure_name(name)
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

        self._register(name=name, rhs=rhs, diag=diag, data=df)

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
        label = self._default_label(label)
        metrics = evaluate_model(
            entry.diag,
            df,
            label=label,
            target_col=target_col,
        )
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
        criterion: Literal["aic", "aicc", "bic", "mdl", "cp", "adj_r2"] = "aic",
        threshold: float = 1.0,
        cv_folds: int | None = None,
        shuffle_cv: bool = False,
        random_state: int | None = None,
        max_models: int | None = None,
        reuse_path_model: bool = True,
        refit: bool = False,
    ) -> RegressionResult:
        """Run a stepwise selection path and cache the best model.

        Stepwise selection is a greedy variable-selection procedure that trades
        off goodness of fit against model complexity using a criterion such as
        AIC/BIC (or Cp/adjusted R^2). AIC/BIC compare models via the maximized
        log-likelihood and the number of estimated parameters.
        Lower AIC/BIC/Cp indicates a preferred model;
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
        name = self._ensure_name(name)
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
            max_models=max_models,
        )

        return self.add_from_path(
            path,
            name=name,
            df=df,
            target_col=target_col,
            cv_folds=cv_folds,
            shuffle_cv=shuffle_cv,
            random_state=random_state,
            refit=refit,
            reuse_path_model=reuse_path_model,
        )

    def add_from_path(
        self,
        path: SelectionPathResult,
        *,
        name: str,
        df: pd.DataFrame | None = None,
        target_col: str = LECol.TARGET,
        cv_folds: int | None = None,
        shuffle_cv: bool = False,
        random_state: int | None = None,
        refit: bool = False,
        reuse_path_model: bool = True,
    ) -> RegressionResult:
        """Register the best step from a selection path.

        By default, reuse the fitted model stored in the path to avoid a refit.
        Set ``reuse_path_model=False`` to refit from ``df``.
        """
        if name in self.models and not refit:
            return self.models[name].diag

        step = path.best_step()
        if reuse_path_model:
            diag = diagnose_ols(
                step.model,
                cv_folds=cv_folds,
                shuffle_cv=shuffle_cv,
                random_state=random_state,
            )
            entry_data = df or self._frame_from_model(step.model)
        else:
            if df is None:
                raise ValueError("df is required when reuse_path_model is False.")
            diag = fit_ols(
                df,
                rhs=step.rhs,
                target_col=target_col,
                cv_folds=cv_folds,
                shuffle_cv=shuffle_cv,
                random_state=random_state,
            )
            entry_data = df

        self._register(
            name=name,
            rhs=step.rhs,
            diag=diag,
            selection_path=path,
            data=entry_data,
        )
        return diag

    def compare(self, *, sort_by: str = "aic") -> pd.DataFrame:
        """Return a comparison table for all cached models."""
        rows = []
        for entry in self.models.values():
            row = {"model": entry.name, "rhs": entry.rhs}
            row.update(asdict(entry.metrics))
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
        reuse_path_model: bool = True,
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
        seen: dict[str, int] = {}
        for label, path in paths.items():
            slug = _slugify(label)
            if slug in seen:
                seen[slug] += 1
            else:
                seen[slug] = 1
            suffix = f"_{seen[slug]}" if seen[slug] > 1 else ""
            name = f"{name_prefix}_{slug}{suffix}" if name_prefix else f"{slug}{suffix}"
            step = path.best_step()
            diag = self.add_from_path(
                path,
                name=name,
                df=df,
                target_col=target_col,
                cv_folds=cv_folds,
                shuffle_cv=shuffle_cv,
                random_state=random_state,
                refit=refit,
                reuse_path_model=reuse_path_model,
            )
            rows.append(
                {
                    "model": name,
                    "label": label,
                    "direction": path.direction,
                    "criterion": path.criterion,
                    "step": int(step.step),
                    "n_terms": len(step.terms),
                    "rhs": step.rhs,
                    "aic": diag.metrics.aic,
                    "aicc": diag.metrics.aicc,
                    "bic": diag.metrics.bic,
                    "mdl": diag.metrics.mdl,
                    "adj_r2": diag.metrics.adj_r2,
                    "rmse": diag.metrics.rmse,
                    "cv_rmse": diag.metrics.cv_rmse,
                    "cp": step.cp,
                },
            )
        return pd.DataFrame(rows)

    def run_selection_grid(  # noqa: PLR0913
        self,
        df: pd.DataFrame,
        *,
        base_terms: list[str] | None,
        candidates: list[str],
        target_col: str = LECol.TARGET,
        directions: Iterable[str] | None = None,
        criteria: Iterable[str] | None = None,
        thresholds: dict[str, float] | None = None,
        cv_folds: int | None = None,
        shuffle_cv: bool = False,
        random_state: int | None = None,
        max_models: int | None = None,
        name_prefix: str = "best",
        reuse_path_model: bool = True,
        refit: bool = True,
        eval_df: pd.DataFrame | None = None,
        eval_label: str | None = None,
        sort_by: str = "aic",
    ) -> tuple[dict[str, SelectionPathResult], pd.DataFrame, pd.DataFrame]:
        """Run a grid of selection paths and register the best model per path.

        Returns (paths, best_map, model_compare).
        """
        if directions is None:
            directions = ("forward", "backward", "stepwise")
        if criteria is None:
            criteria = ("aic", "aicc", "bic", "mdl", "cp", "adj_r2")
        if thresholds is None:
            thresholds = {
                "aic": 1.0,
                "aicc": 1.0,
                "bic": 1.0,
                "mdl": 1.0,
                "cp": 1.0,
                "adj_r2": 0.0,
            }

        paths: dict[str, SelectionPathResult] = {}
        for direction in directions:
            for criterion in criteria:
                label = f"{direction.upper()}-{criterion.upper()}"
                paths[label] = selection_path(
                    data=df,
                    target_col=target_col,
                    base_terms=base_terms,
                    candidates=candidates,
                    direction=direction,
                    criterion=criterion,
                    threshold=thresholds.get(criterion, 1.0),
                    cv_folds=cv_folds,
                    shuffle_cv=shuffle_cv,
                    random_state=random_state,
                    max_models=max_models,
                )

        best_map = self.register_best_paths(
            df,
            paths=paths,
            target_col=target_col,
            name_prefix=name_prefix,
            cv_folds=cv_folds,
            shuffle_cv=shuffle_cv,
            random_state=random_state,
            reuse_path_model=reuse_path_model,
            refit=refit,
        )

        if eval_df is not None:
            self.evaluate_all(eval_df, label=eval_label, target_col=target_col)

        model_compare = (
            self.compare(sort_by=sort_by)
            .reset_index()
            .merge(
                best_map.drop(
                    columns=[
                        col
                        for col in ("aic", "aicc", "bic", "mdl", "adj_r2", "rmse", "cv_rmse", "cp")
                        if col in best_map.columns
                    ],
                ),
                on="model",
                how="left",
            )
            .sort_values(["direction", "criterion", sort_by])
        )

        return paths, best_map, model_compare

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
        entries = [self.get(name) for name in names] if names is not None else list(self.models.values())
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
