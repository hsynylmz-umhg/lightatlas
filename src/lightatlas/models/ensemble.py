"""Ensemble anomaly scoring methods."""

from collections.abc import Mapping

import numpy as np
from scipy.stats import rankdata


def rank_average(scores: Mapping[str, np.ndarray]) -> np.ndarray:
    """Compute consensus percentile rank average across multiple scoring models.

    Parameters
    ----------
    scores : Mapping[str, np.ndarray]
        Mapping from model identifier to 1D array of anomaly scores.

    Returns
    -------
    np.ndarray
        Consensus percentile rank scores in [0.0, 1.0].

    Raises
    ------
    ValueError
        If scores mapping is empty, arrays have mismatched lengths, or inputs are not 1D.
    """
    if not scores:
        raise ValueError("scores mapping cannot be empty")

    arrays: list[np.ndarray] = []
    expected_len: int | None = None

    for name, arr in scores.items():
        arr_np = np.asarray(arr, dtype=np.float64)
        if arr_np.ndim != 1 or arr_np.size == 0:
            raise ValueError(
                f"Score array for '{name}' must be non-empty 1D array, got shape {arr_np.shape}"
            )
        if expected_len is None:
            expected_len = len(arr_np)
        elif len(arr_np) != expected_len:
            msg = f"Score array length mismatch: expected {expected_len}, got {len(arr_np)}"
            raise ValueError(msg)

        n = len(arr_np)
        if n == 1:
            arrays.append(np.array([0.5], dtype=np.float64))
        else:
            ranks = rankdata(arr_np, method="average")
            arrays.append((ranks - 1.0) / (n - 1.0))

    consensus_mean = np.mean(arrays, axis=0)
    n = len(consensus_mean)
    if n == 1:
        return np.array([0.5], dtype=np.float64)

    final_ranks = rankdata(consensus_mean, method="average")
    return (final_ranks - 1.0) / (n - 1.0)
