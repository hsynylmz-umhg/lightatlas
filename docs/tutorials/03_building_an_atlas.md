# Tutorial 3: Building a Morphological Light Curve Atlas

Learn how to group detected anomalies into morphological clusters and generate an interactive, zero-network SVG gallery.

## 1. Latent Space Clustering & Rarity

```python
import numpy as np
from lightatlas.atlas.cluster import cluster_latent
from lightatlas.atlas.rank import rarity_score

# Latent embeddings Z of shape (N, D)
Z = np.random.randn(200, 16)
clusters = cluster_latent(Z, min_cluster_size=10)
rarity = rarity_score(clusters)

print("Cluster assignments:", np.unique(clusters))
print("Rarity scores:", rarity)
```

## 2. Generating the Offline HTML Gallery

The gallery generator writes fully standalone HTML embedding inline SVG curves without external CDN links:

```python
import pandas as pd
from lightatlas.atlas.report import render_gallery

df = pd.DataFrame(
    {
        "id": ["STAR_01", "STAR_02"],
        "score": [0.95, 0.88],
        "cluster": [1, -1],
        "f": [np.random.randn(1024), np.random.randn(1024)],
    }
)

render_gallery(df, "atlas_gallery.html")
```
