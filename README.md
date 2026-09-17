# LightAtlas

> **Autonomous Anomaly Discovery Engine & Morphological Atlas for Astronomical Time-Series**

[![Documentation](https://img.shields.io/badge/docs-gh--pages-blue)](https://hsynylmz-umhg.github.io/lightatlas/)
[![CI](https://github.com/hsynylmz-umhg/lightatlas/actions/workflows/ci.yml/badge.svg)](https://github.com/hsynylmz-umhg/lightatlas/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LightAtlas is a high-throughput, unsupervised discovery framework designed to surface rare, uncataloged photometric phenomena across massive astronomical surveys (such as TESS, Kepler, and Vera C. Rubin LSST). By combining domain-informed physical feature extraction, deep generative autoencoders (1D-CNN and VQ-VAE), and topological latent clustering, LightAtlas constructs self-organizing morphological atlases without requiring labeled training sets.

---

## ⚡ 3-Command Quickstart

Get up and running with a complete end-to-end discovery run in seconds:

```bash
# 1. Clone repository
git clone https://github.com/hsynylmz-umhg/lightatlas.git && cd lightatlas

# 2. Install with PyTorch support
pip install -e ".[torch]"

# 3. Run end-to-end anomaly discovery demo (200 curves processed in ~3 seconds)
lightatlas demo --n 200
```

This generates:
- `runs/demo/scores.parquet`: High-performance columnar dataset with composite anomaly ranks, cluster IDs, and rarity indices.
- `runs/demo/gallery.html`: Standalone, dark-themed visual gallery containing inline vectorized SVG light curves with zero network dependencies.

---

## 🔭 Why LightAtlas?

Modern time-domain astronomy is undergoing a data avalanche. Traditional machine learning pipelines predominantly rely on **supervised classifiers** (Random Forests, standard CNNs) trained on known variable star catalogs (Cepheids, RR Lyrae, Eclipsing Binaries).

While supervised methods excel at recognizing known variability classes, they suffer from severe confirmation bias:

| Feature | Traditional Supervised Classifiers | LightAtlas Autonomous Engine |
| :--- | :--- | :--- |
| **Discovery Paradigm** | Target-seeking (classifies into pre-defined buckets) | Open-ended anomaly discovery (unsupervised) |
| **Handling of the Unknown** | Misclassifies or discards novel signals as noise | Prioritizes rare morphologies by information divergence |
| **Training Requirement** | Requires tens of thousands of human labels | Zero labels required; self-supervised & geometric |
| **Rare Event Detection** | Struggles on extreme class imbalances (< 0.1%) | Explicitly ranks by morphological rarity ($1 / \sqrt{N_c}$) |
| **Phenomenon Coverage** | Known catalogs only | Exoplanetary transits, stellar superflares, dippers, exotic transients |

**LightAtlas inverts the discovery paradigm**: instead of asking *"Which known class does this curve belong to?"*, it asks:
1. *"How uncharacteristic is this curve relative to the survey baseline?"*
2. *"How structurally rare is its morphology in generative latent space?"*

---

## 🏗️ 3-Stage Discovery Pipeline

```text
               Raw Photometric Time-Series  (TESS, Kepler, Synthetic)
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │   STAGE 1: Physics-Informed Feature Space & Baseline   │
       │   • 40 Morphological & Frequency Features              │
       │     (FFT Power/SNR, ACF Decay, Moments, Spikes/Dips)   │
       │   • Isolation Forest Anomaly Scoring                   │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │   STAGE 2: Deep Generative Latent Representation       │
       │   • 1D-CNN AutoEncoder: Sequence Reconstruction Error  │
       │   • VQ-VAE: Discrete Codebook Quantization & Loss      │
       │   • Consensus Percentile Rank Ensembling               │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │   STAGE 3: Topological Clustering & Morphological Atlas│
       │   • HDBSCAN / Agglomerative Latent Space Clustering    │
       │   • Morphological Rarity Index: R(c) = 1 / sqrt(N_c)   │
       │   • Offline Dark-Themed SVG Visual Atlas & Parquet     │
       └────────────────────────────────────────────────────────┘
```

### Stage 1: Physics-Informed Feature Vector (40 Dimensions)
Extracts a robust mathematical representation capturing:
- **Frequency Domain**: Dominant period, peak power, harmonic power ratio, FFT SNR, spectral band power ratios.
- **Autocorrelation & Memory**: ACF lag-1, autocorrelation FWHM, zero-crossing interval.
- **Statistical Moments**: Robust skewness, kurtosis, IQR-to-median ratio, coefficient of variation.
- **Extreme Value Statistics**: Spike count, maximum flare amplitude, flare index, rise-to-fall ratio, dip depth, dip duration fraction, quantiles ($q_{01}, q_{99}$).

### Stage 2: Deep Generative Anomaly Detection
- **1D-CNN Autoencoder**: 3-layer symmetrical convolutional autoencoder capturing continuous sequence-level morphological anomalies via mean squared reconstruction error.
- **VQ-VAE (Vector Quantized Variational Autoencoder)**: Maps curves into a discrete learned codebook space. Curves exhibiting rare structural features yield high codebook commitment and reconstruction penalties.
- **Consensus Ensembling**: Robust percentile rank averaging across all active models:
  $$\text{Score}_{\text{composite}} = \frac{1}{M} \sum_{m=1}^{M} \text{Rank}_{\%}(S_m)$$

### Stage 3: Topological Clustering & Rarity Atlas
- Clusters compressed latent vectors using **HDBSCAN** (with graceful fallback to agglomerative clustering).
- Computes morphological rarity scores per cluster:
  $$\text{Rarity}(c) = \frac{1}{\sqrt{|C_c|}}$$
- Assigns maximum rarity to topological outliers (label `-1`).

---

## 💻 CLI Usage

LightAtlas provides an ergonomic, self-documenting CLI:

```bash
# Display help and available commands
lightatlas --help

# Generate a synthetic benchmark dataset
lightatlas synthesize --n 1000 --out data/survey.parquet

# Run end-to-end discovery demo with custom parameters
lightatlas demo --n 500 --out-dir runs/experiment_01
```

---

## 🐍 Python API

LightAtlas is modular and can be integrated into custom scientific pipelines:

```python
from pathlib import Path
from lightatlas.pipeline import PipelineConfig, run_pipeline

# Configure the discovery pipeline
cfg = PipelineConfig(
    source="synthetic",
    n_curves=1000,
    seed=42,
    n_points=1024,
    ae_epochs=5,
    vqvae_epochs=10,
    min_cluster_size=10,
    top_fraction=0.05,
    out_dir=Path("runs/discovery_run"),
)

# Execute the pipeline
artifacts = run_pipeline(cfg)
print(f"Scores exported to: {artifacts['scores']}")
print(f"Interactive atlas: {artifacts['gallery']}")
```

### Direct Modular Usage

```python
import numpy as np
from lightatlas.core.features import compute_features
from lightatlas.core.preproc import robust_normalize
from lightatlas.synth import random_mixture

# 1. Generate synthetic light curves
dataset = random_mixture(n=100, seed=42)

# 2. Vectorized preprocessing
f_norm = robust_normalize(dataset.f)

# 3. Extract 40-feature physical vector for curve 0
t, f = dataset.t, f_norm[0]
feat_vec = compute_features(t, f)
print(f"Feature vector shape: {feat_vec.shape}")  # (40,)
```

---

## 🎨 Zero-Network SVG Gallery

The generated `gallery.html` provides a zero-dependency, self-contained visual interface:
- **100% Offline & Isolated**: Zero external fonts, zero CDN links, zero tracking scripts.
- **Embedded Scalable Vector Graphics**: Light curves rendered as native inline `<svg>` polyline elements with responsive styling.
- **Information Rich**: Displays object IDs, true anomaly labels, composite percentile ranks, cluster assignments, and morphological rarity metrics.

---

## 🧪 Rigorous Quality & Verification

Every pull request and release is validated across multiple operating systems and Python versions:

```bash
# Code style and linting
ruff check .

# Code formatting
ruff format --check .

# Comprehensive test suite with coverage
pytest -q --cov-fail-under=80

# Build documentation in strict mode
mkdocs build --strict
```

---

## 📄 License & Citation

LightAtlas is distributed under the open-source [MIT License](LICENSE).

If you use LightAtlas in your astronomical research or survey analysis, please cite:

```bibtex
@software{yilmaz2026lightatlas,
  author       = {Huseyin Yilmaz},
  title        = {LightAtlas: Autonomous Anomaly Discovery Engine & Morphological Atlas for Astronomical Time-Series},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/hsynylmz-umhg/lightatlas}
}
```
