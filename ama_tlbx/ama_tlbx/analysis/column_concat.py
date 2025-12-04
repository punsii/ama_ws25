"""Column Concatenation."""

from dataclasses import dataclass

import pandas as pd
from sklearn.decomposition import PCA

from ama_tlbx.data.base_dataset import BaseDataset
from ama_tlbx.data.views import DatasetView


@dataclass
class ColumnConcatenator:
    """Utility for concatenating multiple columns into a single column.

    Example:
        >>> from ama_tlbx.data import LifeExpectancyDataset
        >>> from ama_tlbx.analysis.column_concat import ColumnConcatenator
        >>> dataset = LifeExpectancyDataset.from_csv()
        >>> view = dataset.analyzer_view(standardized=False)
        >>> cc = ColumnConcatenator(view)
        >>> combined_view = cc.concatenate(["polio", "diphtheria"], new_column_name="immunization_mean")
    """

    dataset: BaseDataset
    explained_variance: float = 0.0
    loadings: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __init__(self, dataset: BaseDataset):
        """Initialize the Column Concatenator."""
        self.dataset = dataset

    def print_results(self) -> None:
        """Print the explained variance of the last PCA operation."""
        print(
            f"The first principal component  explains {self.explained_variance:.2f}% of the variance within these columns."
        )

        print("PCA loadings, weighted (squared for component contribution):")
        print(self.loadings)

    def concatenate(
        self,
        columns: list[str],
        new_column_name: str,
    ) -> BaseDataset:
        # copy the underlying DataFrame from the provided view and return after
        # concatenation. Use `self.view` (the dataclass field) rather than
        # an underscore-prefixed attribute which doesn't exist.
        df = self.dataset.df.copy()

        # Separate the columns to be concatenated and those to remain
        only_columns = df[columns]
        only_not_columns = df.drop(columns=columns)

        # Perform PCA to reduce to a single column
        pca = PCA(n_components=1)
        pc = pca.fit(only_columns)

        # build a tidy loadings DataFrame with one row per feature and a
        # single numeric 'loading' column (shape: n_features x 1), then
        # square the loading values as requested.
        loadings = (
            pd.DataFrame(pc.components_.T, index=only_columns.columns, columns=["loading"])  # (n_features, 1)
            .reset_index()
            .rename(columns={"index": "feature"})
        )

        # raise loadings to the power of 2 to get normalised importance
        loadings["loading"] **= 2
        self.loadings = loadings

        # Construct a weighted sum using the squared loadings.
        # - `loadings` has columns ['feature', 'loading']
        weights = loadings.set_index("feature")["loading"]

        self.explained_variance = pc.explained_variance_ratio_[0] * 100  # Convert to percentage

        available = [f for f in weights.index if f in only_columns.columns]
        if len(available) == 0:
            raise ValueError("None of the PCA features are present in the DataFrame columns")

        # Select the columns in the same order as weights, fill missing values
        # with 0 so they don't contribute to the weighted sum.
        selected = only_columns[available].fillna(0)

        # Perform weighted sum
        weighted_series = selected.mul(weights.loc[available], axis=1).sum(axis=1)
        only_not_columns[new_column_name] = weighted_series

        # Return a new dataset instance containing the modified DataFrame. We avoid
        # mutating the original dataset and instead use BaseDataset.with_df which by
        # default constructs a new instance of the same concrete class.
        return self.dataset.with_df(only_not_columns)
