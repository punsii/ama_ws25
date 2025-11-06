"""Outlier detection strategies following the analyzer pattern."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ama_tlbx.data.views import DatasetView

from .base_analyser import BaseAnalyser


@dataclass(frozen=True)
class OutlierDetectionResult:
    """Container for outlier detection results.

    TODO(@jd): Expand with relevant statistics (for each outlier detection strategy).
        https://github.com/punsii/ama_ws25/issues/TBD

    Attributes:
        outlier_mask: DataFrame of boolean values indicating outliers per column
        n_outliers_per_column: Series with count of outliers per column
        n_outliers_per_row: Series with count of outliers per row
        total_outliers: Total number of outlier flags across all entries
        outlier_percentage: Percentage of values flagged as outliers
        pretty_names: Mapping of column names to pretty display names
    """

    outlier_mask: pd.DataFrame
    n_outliers_per_column: pd.Series
    n_outliers_per_row: pd.Series
    pretty_names: dict[str, str] | None = None


class IQROutlierDetector(BaseAnalyser):
    r"""Detect outliers via the interquartile range rule.

    Points outside :math:`[Q_1 - k\cdot IQR,\, Q_3 + k\cdot IQR]` are considered outliers. :math:`IQR=Q3.-Q1`

    Theory and Assumptions:
        - Does not assume any specific data distribution (non-parametric)
        - Robust to skewed data with heavy tails
        - Best suited for *univariate* datasets [[2]](https://procogia.com/interquartile-range-method-for-reliable-data-analysis/#:~:text=While%20the%20IQR%20method%20is,ML%29%20methods)
        See [Wikipedia :: Interquartile range](https://en.wikipedia.org/wiki/Interquartile_range) for additional background.

    Attributes:
        threshold: Multiplier ``k`` applied to the IQR when computing the fences.
            Default is 1.5 according to Tukey's rule. 5-10% of data points are typically flagged as outliers.
    """

    def __init__(self, view: DatasetView, threshold: float = 1.5) -> None:
        """Initialize IQR outlier detector.

        Args:
            view: Immutable dataset view to analyze
            threshold: IQR multiplier for fence calculation (default: 1.5)
        """
        self._view = view
        self.threshold = threshold
        self._fitted = False
        self._outlier_mask: pd.DataFrame | None = None

    def fit(self) -> "IQROutlierDetector":
        """Detect outliers using IQR method.

        Returns:
            Self for method chaining.
        """
        selected = self._view.df.loc[:, self._view.numeric_cols]
        q1 = selected.quantile(0.25)
        q3 = selected.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.threshold * iqr
        upper = q3 + self.threshold * iqr
        self._outlier_mask = selected.lt(lower) | selected.gt(upper)
        self._fitted = True
        return self

    def result(self) -> OutlierDetectionResult:
        """Return outlier detection results.

        Returns:
            OutlierDetectionResult containing outlier mask and statistics.

        Raises:
            ValueError: If fit() has not been called yet.
        """
        if not self._fitted or self._outlier_mask is None:
            raise ValueError("Must call fit() before result()")

        return OutlierDetectionResult(
            outlier_mask=self._outlier_mask,
            n_outliers_per_column=self._outlier_mask.sum(),
            n_outliers_per_row=self._outlier_mask.sum(axis=1),
            pretty_names=dict(self._view.pretty_by_col),
        )


class ZScoreOutlierDetector(BaseAnalyser):
    r"""Detect outliers using Z-scores.

    Values whose standardized score exceeds ``threshold`` are considered outliers.

    Theory and Assumptions:
        - Assumes normality; extreme z-scores may reflect skewness or heavy tails rather than true anomalis [[1]](https://www.statology.org/top-5-statistical-techniques-detect-handle-outliers-data/#:~:text=Pros%20and%20Cons).
        - Suited for univairate / feature-wise outlier detection.
        - Alternative for better robustness uses median and MAD instead of mean and stddev.
        See [Wikipedia :: Standard score](https://en.wikipedia.org/wiki/Standard_score) for details.

    Attributes:
        threshold: Absolute z-score limit used to flag observations (default: 3.0).
    """

    def __init__(self, view: DatasetView, threshold: float = 3.0) -> None:
        """Initialize Z-score outlier detector.

        Args:
            view: Immutable dataset view to analyze
            threshold: Z-score threshold for outlier detection (default: 3.0)
        """
        self._view = view
        self.threshold = threshold
        self._fitted = False
        self._outlier_mask: pd.DataFrame | None = None

    def fit(self) -> "ZScoreOutlierDetector":
        r"""Detect outliers using Z-score method.

        A data point is considered an outlier if its absolute z-score exceeds the threshold:
        :math:`\mathbb{1}(|X - \mu| / \sigma > k)`

        Returns:
            Self for method chaining.
        """
        selected = self._view.df.loc[:, self._view.numeric_cols]
        self._outlier_mask = selected.sub(selected.mean()).div(selected.std()).abs().gt(self.threshold)
        self._fitted = True
        return self

    def result(self) -> OutlierDetectionResult:
        """Return outlier detection results.

        Returns:
            OutlierDetectionResult containing outlier mask and statistics.

        Raises:
            ValueError: If fit() has not been called yet.
        """
        if not self._fitted or self._outlier_mask is None:
            raise ValueError("Must call fit() before result()")

        return OutlierDetectionResult(
            outlier_mask=self._outlier_mask,
            n_outliers_per_column=self._outlier_mask.sum(),
            n_outliers_per_row=self._outlier_mask.sum(axis=1),
            pretty_names=dict(self._view.pretty_by_col),
        )


class IsolationForestOutlierDetector(BaseAnalyser):
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
        contamination: Expected proportion of outliers in the dataset (default: "auto").
        random_state: Optional random seed for reproducible tree ensembles.
        n_estimators: Number of isolation trees fitted in the ensemble (default: 100).
    """

    def __init__(
        self,
        view: DatasetView,
        contamination: float | str = "auto",
        random_state: int | None = None,
        n_estimators: int = 100,
    ) -> None:
        """Initialize Isolation Forest outlier detector.

        Args:
            view: Immutable dataset view to analyze
            contamination: Expected proportion of outliers (default: "auto")
            random_state: Random seed for reproducibility (default: None)
            n_estimators: Number of trees in the forest (default: 100)
        """
        self._view = view
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self._fitted = False
        self._outlier_mask: pd.DataFrame | None = None

    def fit(self) -> "IsolationForestOutlierDetector":
        """Detect outliers using Isolation Forest method.

        Returns:
            Self for method chaining.
        """
        selected = self._view.df.loc[:, self._view.numeric_cols]
        model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
        )
        model.fit(selected)
        row_outliers = pd.Series(model.predict(selected) == -1, index=selected.index)
        mask = np.repeat(row_outliers.to_numpy()[:, None], selected.shape[1], axis=1)
        self._outlier_mask = pd.DataFrame(mask, index=selected.index, columns=selected.columns)
        self._fitted = True
        return self

    def result(self) -> OutlierDetectionResult:
        """Return outlier detection results.

        Returns:
            OutlierDetectionResult containing outlier mask and statistics.

        Raises:
            ValueError: If fit() has not been called yet.
        """
        if not self._fitted or self._outlier_mask is None:
            raise ValueError("Must call fit() before result()")

        return OutlierDetectionResult(
            outlier_mask=self._outlier_mask,
            n_outliers_per_column=self._outlier_mask.sum(),
            n_outliers_per_row=self._outlier_mask.sum(axis=1),
            pretty_names=dict(self._view.pretty_by_col),
        )
