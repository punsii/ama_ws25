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


# TODO: all outlier detector datasets must return multi index dfs that contain the naked boolean values per entrie and also relevant statistics that have been computed along the way!
@dataclass
class IQROutlierDetector(OutlierDetector):
    r"""Detect outliers via the interquartile range rule.

    Points outside :math:`[Q_1 - k\cdot IQR,\, Q_3 + k\cdot IQR]` are considered outliers. :math:`IQR=Q3.-Q1`

    Theory and Assumptions:
        - Does not assume any specific data distribution (non-parametric)
        - Robust to skewed data with heavy tails
        - Best suited for *univariate* datasets [[2]](https://procogia.com/interquartile-range-method-for-reliable-data-analysis/#:~:text=While%20the%20IQR%20method%20is,ML%29%20methods)
        See [Wikipedia :: Interquartile range](https://en.wikipedia.org/wiki/Interquartile_range) for additional background.

    Attributes:
        threshold: Multiplier ``k`` applied to the IQR when computing the fences.
    """

    threshold: float = 1.5
    """default is 1.5 according to Tukey's rule. 5-10% of data points are typically flagged as outliers [[Wiki]](https://en.wikipedia.org/wiki/Outlier)."""

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
    r"""Detect outliers using Z-scores.

    Values whose standardized score exceeds ``threshold`` are considered outliers.

    Theory and Assumptions:
        - Assumes normality; extreme z-scores may reflect skewness or heavy tails rather than true anomalis [[1]](https://www.statology.org/top-5-statistical-techniques-detect-handle-outliers-data/#:~:text=Pros%20and%20Cons).
        - Suited for univairate / feature-wise outlier detection.
        - Alternative for better robustness uses median and MAD instead of mean and stddev.
        See [Wikipedia :: Standard score](https://en.wikipedia.org/wiki/Standard_score) for details.

    Attributes:
        threshold: Absolute z-score limit used to flag observations.
    """

    threshold: float = 3.0

    def detect(self, data: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
        r"""Detect outliers in the specified columns of the input DataFrame. Resulting dataframe will contain boolean values indicating outliers. A data point is considered an outlier if its absolute z-score exceeds the defined threshold.

        Returns:
            pd.DataFrame(dtypes: bool): df with same shape as data[selected], where df.loc[...] = True if :math`\mathbb{1}(df |X - E[X]| / SD(X) > k)`
        """
        selected = data.loc[:, list(columns) if columns is not None else data.columns]

        return selected.sub(selected.mean()).div(selected.std()).abs().gt(self.threshold)


@dataclass
class IsolationForestOutlierDetector(OutlierDetector):
    """Detect row-wise outliers (=: anomalies) using scikit-learn's IsolationForest.

    Theory:
        Isolates anomalies by building an ensemble of random binary trees. At each node, a random feature and a random split value are chosen until every sample is isolated or a maximum tree depth is reached [[1]](https://seppe.net/aa/papers/iforest.pdf#:~:text=2%20Isolation%20and%20Isolation%20Trees,forest%20of%20random%20trees%20collectively).
        - Internal nodes are randomly choosen split-values on a random feature.
        - Leaf nodes store the number of samples per partition.
        - The length of any sample path h(x) is the number is the number of edges traversed from root to leaf.
        -  For a dataset of n points, the expected path length for random partitioning is c(n) = 2H(n-1). The anomaly score is computed as 2^{-h(x)/c(n)}; values close to
        1 indicate anomalies, whereas scores near 0.5 correspond to normal observations [[2]](https://seppe.net/aa/papers/iforest.pdf).
        See [sklearn :: IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) for implementation details.

    Assumptions:
        -  Assumes anomalies are rare and differ markedly from normal data; such points can be isolated with fewer random splits.
        - Effective for large, multivariate datasets without strong priors.


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
