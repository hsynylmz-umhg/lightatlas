"""Latent space clustering module with HDBSCAN and Agglomerative fallback."""

import logging

import numpy as np
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)


def cluster_latent(
    Z: np.ndarray,
    min_cluster_size: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """Cluster latent representations into discrete morphologic clusters.

    Attempts to use HDBSCAN, falling back to AgglomerativeClustering if HDBSCAN
    is unavailable or fails.

    Parameters
    ----------
    Z : np.ndarray
        Latent vectors or feature matrix of shape (n_samples, n_features).
    min_cluster_size : int, default=10
        Minimum cluster size.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        1D integer array of cluster assignments of shape (n_samples,).
        Outliers are assigned label -1 where supported.
    """
    Z_arr = np.asarray(Z, dtype=np.float64)
    if Z_arr.ndim != 2:
        raise ValueError(f"Z must be 2D array, got shape {Z_arr.shape}")

    n_samples = len(Z_arr)
    if n_samples == 0:
        return np.array([], dtype=np.int64)
    if n_samples < min_cluster_size:
        return np.zeros(n_samples, dtype=np.int64)

    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=max(1, min_cluster_size // 2),
        )
        labels = clusterer.fit_predict(Z_arr)
        return labels.astype(np.int64)
    except ImportError:
        logger.warning("hdbscan is not installed; falling back to AgglomerativeClustering.")
    except (ValueError, TypeError, RuntimeError) as exc:
        logger.warning(
            "hdbscan failed (%s); falling back to AgglomerativeClustering.",
            exc,
        )

    n_clusters = max(1, min(8, n_samples // min_cluster_size))
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="average",
    )
    labels = clusterer.fit_predict(Z_arr)
    return labels.astype(np.int64)
