"""Rarity scoring for clustered light curve atlas."""

import numpy as np


def rarity_score(labels: np.ndarray) -> dict[int, float]:
    """Compute rarity metric per cluster label.

    Rarity is inversely proportional to the square root of cluster population:
    rarity[c] = 1.0 / sqrt(cluster_size).
    Outliers (label -1) are assigned the maximum rarity score (max(rarity) + 1.0).

    Parameters
    ----------
    labels : np.ndarray
        Array of integer cluster labels.

    Returns
    -------
    dict[int, float]
        Dictionary mapping cluster ID to rarity score.
    """
    labels_arr = np.asarray(labels, dtype=np.int64)
    if len(labels_arr) == 0:
        return {}

    unique, counts = np.unique(labels_arr, return_counts=True)
    count_map = dict(zip(unique, counts, strict=False))

    scores: dict[int, float] = {}
    valid_rarities: list[float] = []

    for c, cnt in count_map.items():
        if c != -1:
            val = float(1.0 / np.sqrt(cnt))
            scores[int(c)] = val
            valid_rarities.append(val)

    if -1 in count_map:
        max_rarity = max(valid_rarities) if valid_rarities else 0.0
        scores[-1] = float(max_rarity + 1.0)

    return scores
