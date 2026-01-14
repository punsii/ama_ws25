"""Clustering diagnostics: elbow, silhouette, and dendrogram plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def plot_elbow_curve(
    data: np.ndarray,
    k_range: range | list[int] = range(2, 11),
    *,
    random_state: int | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot within-cluster inertia across candidate k to find the elbow.

    Fits [:class:`sklearn.cluster.KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
    for each ``k`` and plots the resulting inertia.
    """
    ax = ax or plt.gca()
    inertias = []
    ks = list(k_range)
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(data)
        inertias.append(km.inertia_)
    ax.plot(ks, inertias, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia (within-cluster SSE)")
    ax.set_title("Elbow Plot")
    ax.grid(alpha=0.2)
    return ax


def plot_silhouette_scores(
    data: np.ndarray,
    k_range: range | list[int] = range(2, 11),
    *,
    random_state: int | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Average silhouette score per k.

    Uses [:func:`sklearn.metrics.silhouette_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
    to summarize cluster separation.
    """
    ax = ax or plt.gca()
    scores = []
    ks = list(k_range)
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(data)
        scores.append(silhouette_score(data, labels))
    ax.plot(ks, scores, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("Average silhouette score")
    ax.set_title("Silhouette Scores by k")
    ax.grid(alpha=0.2)
    return ax


def plot_silhouette_bars(
    data: np.ndarray,
    labels: np.ndarray,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Per-sample silhouette values as bars, grouped by cluster.

    Computes per-point values with [:func:`sklearn.metrics.silhouette_samples`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html)
    and draws stacked bars with Matplotlib.
    """
    ax = ax or plt.gca()
    n_clusters = len(np.unique(labels))
    sil_vals = silhouette_samples(data, labels)
    y_lower = 10
    for cluster in np.unique(labels):
        cluster_vals = sil_vals[labels == cluster]
        cluster_vals.sort()
        size = cluster_vals.shape[0]
        y_upper = y_lower + size
        color = sns.color_palette("husl", n_clusters)[cluster % n_clusters]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals, facecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, str(cluster))
        y_lower = y_upper + 10
    ax.axvline(np.mean(sil_vals), color="red", linestyle="--", label="Average silhouette")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Samples (grouped by cluster)")
    ax.set_title("Silhouette Plot per Cluster")
    ax.legend()
    return ax


def plot_dendrogram(
    data: np.ndarray,
    *,
    method: str = "ward",
    metric: str = "euclidean",
    ax: plt.Axes | None = None,
) -> Figure:
    """Hierarchical clustering dendrogram.

    Calls [:func:`scipy.cluster.hierarchy.dendrogram`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html)
    on linkage produced by [:func:`scipy.cluster.hierarchy.linkage`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html).
    """
    ax = ax or plt.gca()
    Z = linkage(data, method=method, metric=metric)
    dendrogram(Z, ax=ax)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Distance")
    ax.set_title(f"Dendrogram ({method}, {metric})")
    return ax.figure
