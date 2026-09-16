# LightAtlas

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LightAtlas** is an autonomous, high-throughput discovery engine and morphological atlas generator for astronomical time-series photometry.

---

## Scientific Architecture

LightAtlas implements a multi-layer consensus anomaly detection and morphological taxonomy pipeline designed for large sky surveys (e.g. Kepler, TESS, Rubin/LSST).

### ASCII Pipeline Diagram

```text
  +-------------------------------------------------------------+
  |              Raw Survey Photometry / Synthetic              |
  +-------------------------------------------------------------+
                                 |
                                 v
  +-------------------------------------------------------------+
  |     Preprocessing & Normalization (Median / 1.4826 * MAD)    |
  +-------------------------------------------------------------+
                                 |
         +-----------------------+-----------------------+
         |                                               |
         v                                               v
  +-----------------------------+               +-----------------------------+
  |  Layer 1: 40-Feature Vector |               |  Layer 2: Deep Anomaly      |
  |  (FFT, ACF, Moments, Spikes)|               |  (1D-CNN AE + VQ-VAE)       |
  +-----------------------------+               +-----------------------------+
                 |                                               |
                 v                                               v
  +-----------------------------+               +-----------------------------+
  |   Isolation Forest Scorer   |               |   Recon + Codebook Distance |
  +-----------------------------+               +-----------------------------+
                 |                                               |
                 +-----------------------+-----------------------+
                                         |
                                         v
  +-------------------------------------------------------------+
  |            Consensus Percentile Rank Ensembling             |
  +-------------------------------------------------------------+
                                 |
                                 v
  +-------------------------------------------------------------+
  |      Morphological Clustering & Rarity (HDBSCAN / Agglomer) |
  +-------------------------------------------------------------+
                                 |
         +-----------------------+-----------------------+
         |                                               |
         v                                               v
  +-----------------------------+               +-----------------------------+
  |   scores.parquet Database   |               |   Zero-Network SVG Gallery  |
  +-----------------------------+               +-----------------------------+
```

---

## Core Capabilities

- **Deterministic Synthetic Factory**: Reproducible stellar light curve generation with quiet, flare, exoplanet transit, and eclipsing binary profiles.
- **Robust Feature Engineering**: 40 specialized time-series features spanning frequency harmonics, autocorrelation width, statistical moments, and physical dip fractions.
- **Deep Learning Anomaly Detection**: CPU-optimized 1D convolutional autoencoders and vector-quantized latent representations.
- **Offline HTML Gallery**: Fully standalone, zero-network interactive reports with embedded inline SVG curves.
