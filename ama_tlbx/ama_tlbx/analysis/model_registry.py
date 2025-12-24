from dataclasses import asdict, dataclass, field

import pandas as pd

from ..data import LECol
from .ols_helper import (
    EvalMetrics,
    MetricsResult,
    RegressionResult,
    evaluate_model,
    fit_ols_design,
    fit_ols_formula,
)


@dataclass
class ModelEntry:
    """Typed model registry entry for reporting workflows."""

    name: str
    rhs: str
    diag: RegressionResult
    metrics: MetricsResult
    eval_metrics: EvalMetrics | None = None


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
            diag = fit_ols_formula(
                df,
                rhs=rhs,
                target_col=target_col,
                cv_folds=cv_folds,
                shuffle_cv=shuffle_cv,
                random_state=random_state,
            )
        else:
            diag = fit_ols_design(
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
        return metrics

    def compare(self, *, sort_by: str = "aic") -> pd.DataFrame:
        """Return a comparison table for all cached models."""
        rows = []
        for entry in self.models.values():
            row = {"model": entry.name, "rhs": entry.rhs}
            row.update(asdict(entry.metrics))
            if entry.eval_metrics is not None:
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

    def __iter__(self):
        return iter(self.models.values())

    def __len__(self) -> int:
        return len(self.models)
