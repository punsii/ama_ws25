"""Outlier detection strategies operating."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class OutlierDetector(ABC):
    """Abstract base class for vectorized outlier detection strategies."""

    @abstractmethod
    def detect(self, data: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
        """Return boolean mask highlighting outliers."""


@dataclass
class IQROutlierDetector(OutlierDetector):
    r"""Detect outliers via the classical interquartile range rule.

    Points outside :math:`[Q_1 - k\cdot IQR,\, Q_3 + k\cdot IQR]` are flagged as
    anomalies. See [Wikipedia :: Interquartile range](https://en.wikipedia.org/wiki/Interquartile_range)
    for additional background.

    Attributes:
        threshold: Multiplier ``k`` applied to the IQR when computing the fences.
    """

    threshold: float = 1.5

    def detect(self, data: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
        selected = data.loc[:, list(columns) if columns is not None else data.columns]
        q1 = selected.quantile(0.25)
        q3 = selected.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.threshold * iqr
        upper = q3 + self.threshold * iqr
        return selected.lt(lower) | selected.gt(upper)


@dataclass
class ZScoreOutlierDetector(OutlierDetector):
    """Detect outliers using Z-scores.

    Values whose standardized score exceeds ``threshold`` are considered outliers.
    See [Wikipedia :: Standard score](https://en.wikipedia.org/wiki/Standard_score) for details.

    Attributes:
        threshold: Absolute z-score limit used to flag observations.
        ddof: Delta degrees of freedom used when estimating the standard deviation.
    """

    threshold: float = 3.0
    ddof: int = 0

    def detect(self, data: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
        selected = data.loc[:, list(columns) if columns is not None else data.columns]
        standardized = selected.sub(selected.mean()).div(selected.std(ddof=self.ddof))
        return standardized.abs().gt(self.threshold)


@dataclass
class IsolationForestOutlierDetector(OutlierDetector):
    """Detect row-wise outliers using scikit-learn's IsolationForest.

    The algorithm isolates anomalies by randomly partitioning feature space.
    See [sklearn :: IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
    for implementation details.

    Attributes:
        contamination: Expected proportion of outliers in the dataset.
        random_state: Optional random seed for reproducible tree ensembles.
        n_estimators: Number of isolation trees fitted in the ensemble.
    """

    contamination: float | str = "auto"
    random_state: int | None = None
    n_estimators: int = 100

    def detect(self, data: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
        selected = data.loc[:, list(columns) if columns is not None else data.columns]
        model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
        )
        model.fit(selected)
        row_outliers = pd.Series(model.predict(selected) == -1, index=selected.index)
        mask = np.repeat(row_outliers.to_numpy()[:, None], selected.shape[1], axis=1)
        return pd.DataFrame(mask, index=selected.index, columns=selected.columns)
