"""Tests for IsolationScorer and rank-averaging consensus ensemble."""

import numpy as np
import pytest

from lightatlas.core.features import feature_matrix
from lightatlas.models.ensemble import rank_average
from lightatlas.models.isolation import IsolationScorer
from lightatlas.synth import random_mixture


def test_isolation_recovers_injected_anomalies():
    # 400-curve dataset with injected anomalies (5% non-quiet: 20 anomalies)
    synth_set = random_mixture(
        n=400,
        seed=42,
        n_points=1024,
        quiet_ratio=0.95,
        flare_ratio=0.025,
        transit_ratio=0.0125,
        eclipse_ratio=0.0125,
    )

    X = feature_matrix(synth_set.t, synth_set.f)
    scorer = IsolationScorer(contamination=0.05, random_state=42)
    scorer.fit(X)
    scores = scorer.score(X)

    # Verify score normalization
    assert len(scores) == 400
    assert np.min(scores) >= 0.0
    assert np.max(scores) <= 1.0

    # Top 10% anomaly scores (top 40 items)
    top_10_count = int(len(scores) * 0.10)
    top_10_indices = np.argsort(scores)[-top_10_count:]

    anom_indices = np.where(synth_set.labels != "quiet")[0]
    assert len(anom_indices) > 0

    recovered = np.intersect1d(top_10_indices, anom_indices)
    retrieval_rate = len(recovered) / len(anom_indices)

    assert retrieval_rate >= 0.80


def test_rank_average_consensus():
    rng = np.random.default_rng(42)
    n = 100

    score1 = rng.uniform(0.0, 1.0, size=n)
    score2 = score1 + rng.normal(0.0, 0.05, size=n)

    models_dict = {"iso_1": score1, "iso_2": score2}
    consensus = rank_average(models_dict)

    # Check shape, bounds, and monotonicity
    assert consensus.shape == (n,)
    assert 0.0 <= np.min(consensus) <= 0.05
    assert 0.95 <= np.max(consensus) <= 1.0

    # Strong correlation with inputs
    corr = np.corrcoef(consensus, score1)[0, 1]
    assert corr > 0.90

    # Single-element edge case
    single_res = rank_average({"m1": np.array([0.8]), "m2": np.array([0.2])})
    assert single_res.shape == (1,)
    assert single_res[0] == 0.5

    # Invalid input handling
    with pytest.raises(ValueError, match="cannot be empty"):
        rank_average({})

    with pytest.raises(ValueError, match="mismatch"):
        rank_average({"m1": np.zeros(10), "m2": np.zeros(12)})

    with pytest.raises(ValueError, match="non-empty 1D array"):
        rank_average({"m1": np.zeros((10, 2))})
