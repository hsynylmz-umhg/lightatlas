# Tutorial 1: Synthetic Light Curve Generation & Anomaly Scoring

In this tutorial, we explore how to generate synthetic time-series photometry and score anomalies using the unified pipeline.

## 1. Generating Deterministic Mixtures

The `random_mixture` utility creates realistic mixtures of astronomical variable stars and transients:

```python
from lightatlas.synth import random_mixture

# Generate 100 light curves
data = random_mixture(n=100, seed=42, n_points=1024)

print("IDs:", data.ids[:5])
print("Labels:", data.labels[:5])
print("Flux shape:", data.f.shape)
```

## 2. Feature Extraction & Anomaly Ranking

```python
from lightatlas.core.features import feature_matrix
from lightatlas.models.isolation import IsolationScorer

# Extract 40 features
X = feature_matrix(data.t, data.f)

# Fit Isolation Forest baseline
scorer = IsolationScorer(contamination=0.05, random_state=42)
scorer.fit(X)
scores = scorer.score(X)

print("Highest anomaly score:", scores.max())
```
