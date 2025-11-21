from collections.abc import Iterable
from typing import Union

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .pca_dim_reduction import FeatureGroup


def suggest_groups_from_correlation(
    corr_mat: pd.DataFrame,
    threshold: float = 0.7,
    min_group_size: int = 2,
    return_summary: bool = False,
) -> list[FeatureGroup] | tuple[list[FeatureGroup], pd.DataFrame]:
    """Suggest feature groups based on hierarchical clustering of correlations. Heavily inspired by this
    [Stack Overflow discussion](https://stackoverflow.com/questions/38070478/how-to-do-clustering-using-the-matrix-of-correlation-coefficients).

    This method automatically identifies groups of highly correlated features using
    **hierarchical agglomerative clustering** on the correlation matrix.
    Features with high pairwise correlations will be clustered together.

    Args:
        corr_mat: Pearson correlation matrix from `df.corr()`
        threshold: Minimum absolute correlation to group features together
            (default: 0.7). Features with |r| ≥ threshold will be grouped.
            Higher values (e.g., 0.85) create smaller, tighter groups.
            Lower values (e.g., 0.6) create larger, looser groups.
        min_group_size: Minimum number of features required to form a group
            (default: 2). Groups with fewer features are discarded to prevent singleton groups.
        return_summary: If True, also return a summary DataFrame with group statistics.

    Returns:
        - If ``return_summary`` is False (default): list of FeatureGroup objects
          (2+ highly correlated features), named "Group_1", "Group_2", ...
        - If ``return_summary`` is True: tuple of (
              groups: list[FeatureGroup],
              summary: DataFrame with columns
                  `group`, `size`, `features` (comma-separated),
                  `mean_abs_corr` (average |r| within group),
                  `min_abs_corr` (minimum |r| within group)
          )

    **Theory: Hierarchical Clustering on Correlation Distance**

    1. **Distance Matrix Construction:**
       Transform the correlation matrix into a distance matrix using:

       .. math::
           d(i,j) = 1 - |r_{ij}|

       where :math:`r_{ij}` is the Pearson correlation between features i and j.

    2. **Hierarchical Clustering:**
       Build a **dendrogram** by iteratively merging the closest feature pairs:

       - Start: Each feature is its own cluster
       - Iterate: Find the two closest clusters and merge them
       - Stop: When all features are in one cluster

       We use **average linkage** together with the **L2 distance metric**. The average linkage expresses the
       distance between clusters A and B as the average of all pariwise distances between all elements in A and B
       respectively.

    3. **Cutting the Dendrogram:**
       Cutting the tree at a specific height to obtain final clusters. The height of the cut is parametrized by
       :arg:`treshold`:

       - threshold = 0.7 -> cut at distance = 1 - 0.7 = 0.3
       - Features separated by distance < 0.3 end up in the same group
       - This corresponds to features with |corr| > 0.7

    **Utilized Scipy Functions:**

    - [**scipy.spatial.distance.squareform()**](https://docs.scipy.org/doc//scipy-1.16.2/reference/generated/scipy.spatial.distance.squareform.html): Converts the symmetric distance matrix to "compressed" form (1D array of upper triangle).

    - [**scipy.cluster.hierarchy.linkage()**](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html):
    Performs hierarchical clustering and returns the linkage matrix encoding the dendrogram structure. Each row
    shows which clusters were merged at each step.

    - [**scipy.cluster.hierarchy.fcluster()**](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster): Cuts the dendrogram at the specified distance :arg:`threshold` and assigns cluster
    labels to each feature.

    Example:
        >>> from ama_tlbx.data import LifeExpectancyDataset
        >>> groups = (
        ...     LifeExpectancyDataset.from_csv()
        ...     .make_correlation_analyzer(standardized=True)
        ...     .get_correlation_matrix()
        ...     .pipe(lambda corr_mat:
        ...         PCADimReductionAnalyzer.suggest_groups_from_correlation(corr_mat, threshold=0.7, min_group_size=2)
        ...     )
        ... )
        >>> for group in groups:
        ...     print(f"{group.name}: {', '.join(group.features)}")
        Group_10: life_expectancy, income_composition_of_resources, schooling
        Group_1: infant_deaths, under_five_deaths, population
        Group_8: percentage_expenditure, gdp
        Group_9: thinness_1_19_years, thinness_5_9_years

    """  # noqa: D205
    # Convert to distance matrix
    dist_mat = 1 - corr_mat.abs()

    # Hierarchical clustering
    linkage_matrix = linkage(squareform(dist_mat.values, checks=False), method="average")

    # Cut dendrogram at threshold
    cluster_labels = fcluster(linkage_matrix, t=1 - threshold, criterion="distance")

    # Group features by cluster and convert to FeatureGroups (filter by min_group_size)
    groups = [
        FeatureGroup(name=f"Group_{item[0]}", features=item[1])
        for item in filter(
            lambda item: len(item[1]) >= min_group_size,
            {
                label: [feature for feature, lbl in zip(corr_mat.columns, cluster_labels, strict=False) if lbl == label]
                for label in set(cluster_labels)
            }.items(),
        )
    ]

    if not return_summary:
        return groups

    def _within_group_stats(features: Iterable[str]) -> tuple[float, float]:
        if len(features) < min_group_size:
            return (float("nan"), float("nan"))
        sub = corr_mat.loc[features, features].abs()
        mask = np.tril(np.ones(sub.shape, dtype=bool), k=-1)  # lower triangle excl diag
        vals = sub.where(mask).stack()
        if vals.empty:
            return (float("nan"), float("nan"))
        return float(vals.mean()), float(vals.min())

    summary_rows = []
    for g in groups:
        mean_abs, min_abs = _within_group_stats(g.features)
        summary_rows.append(
            {
                "group": g.name,
                "size": len(g.features),
                "features": ", ".join(map(str, g.features)),
                "mean_abs_corr": mean_abs,
                "min_abs_corr": min_abs,
            },
        )

    return groups, pd.DataFrame(summary_rows).sort_values("group").reset_index(drop=True)
