# Tutorial 2: Processing High-Cadence Survey Photometry (TESS)

Learn how to process 2-minute cadence time series from surveys like TESS using the LightAtlas pipeline.

## 1. Setting Up the Pipeline for TESS

```python
from pathlib import Path
from lightatlas.pipeline import PipelineConfig, run_pipeline

config = PipelineConfig(
    source="tess",
    n_curves=300,
    n_points=1024,
    seed=123,
    ae_epochs=4,
    vqvae_epochs=8,
    top_fraction=0.03,
    out_dir=Path("runs/tess_sector"),
)

artifacts = run_pipeline(config)
print("Pipeline complete:", artifacts)
```

## 2. Inspecting Detected Anomalies

Read the resulting Parquet database with Pandas or PyArrow:

```python
import pandas as pd

df = pd.read_parquet("runs/tess_sector/scores.parquet")
print(df[["id", "score", "cluster", "rarity"]].head(10))
```
