# Quick Start

Get started with LightAtlas in minutes.

---

## Installation

### Standard Installation

```bash
pip install lightatlas
```

### With PyTorch and Development Extras

```bash
git clone https://github.com/hsynylmz-umhg/lightatlas.git
cd lightatlas
uv venv
uv pip install -e ".[dev,torch]"
```

---

## Command Line Interface (CLI)

LightAtlas provides a Typer-powered CLI out of the box:

### Generate Synthetic Light Curves

```bash
lightatlas synthesize --n 500 --seed 42 --out synthetic.parquet
```

### Run End-to-End Offline Demo

```bash
lightatlas demo --n 200 --seed 42 --out-dir runs/demo
```

This generates:
- `runs/demo/scores.parquet`: Parquet table with anomaly scores and clusters.
- `runs/demo/gallery.html`: Standalone offline visual gallery.

---

## Python API Usage

```python
from pathlib import Path
from lightatlas.pipeline import PipelineConfig, run_pipeline

# 1. Define configuration
cfg = PipelineConfig(
    source="synthetic",
    n_curves=250,
    seed=42,
    ae_epochs=5,
    vqvae_epochs=10,
    top_fraction=0.05,
    out_dir=Path("runs/my_experiment"),
)

# 2. Execute pipeline
artifacts = run_pipeline(cfg)

print(f"Scores Parquet: {artifacts['scores']}")
print(f"Visual Gallery: {artifacts['gallery']}")
```
