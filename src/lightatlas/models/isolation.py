"""Isolation Forest baseline anomaly scorer."""

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest


class IsolationScorer:
    """Isolation Forest anomaly scorer wrapping scikit-learn.

    Parameters
    ----------
    n_estimators : int, default=300
        Number of trees in the forest.
    contamination : float or str, default=0.01
        Expected proportion of outliers in the dataset.
    random_state : int, default=42
        Random seed for determinism.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        contamination: float | str = 0.01,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "IsolationScorer":
        """Fit the Isolation Forest model on feature matrix X.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        IsolationScorer
            Fitted instance.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[0] == 0:
            raise ValueError(f"X must be a non-empty 2D array, got shape {X_arr.shape}")

        self.model.fit(X_arr)
        self._is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute percentile rank normalized anomaly scores in [0.0, 1.0].

        Higher values indicate higher degree of anomaly.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Array of scores in range [0.0, 1.0] with shape (n_samples,).
        """
        if not self._is_fitted:
            raise RuntimeError("IsolationScorer must be fitted before calling score.")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[0] == 0:
            raise ValueError(f"X must be a non-empty 2D array, got shape {X_arr.shape}")

        # In scikit-learn IsolationForest, lower score_samples means more anomalous.
        raw_anomaly = -self.model.score_samples(X_arr)
        n = len(raw_anomaly)
        if n == 1:
            return np.array([0.5], dtype=np.float64)

        ranks = rankdata(raw_anomaly, method="average")
        return (ranks - 1.0) / (n - 1.0)
