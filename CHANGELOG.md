# Changelog

All notable changes to LightAtlas will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-16

### Added
- **PEP 621 Skeleton & CI**: Initial modern package layout with setuptools-scm dynamic versioning and GitHub Actions matrix.
- **Deterministic Synthetic Factory**: Parameterized astrophysical light curve generators (`_sine`, `_flare`, `_transit`, `_eclipse`, `_quiet`) and deterministic `random_mixture`.
- **Core Preprocessing**: Uniform linear interpolation resampling (`resample_curve`) and robust median/MAD scaling (`robust_normalize`).
- **40-Feature Extraction**: Exhaustive feature engineering including FFT metrics, autocorrelation, statistical moments, quantiles, spike/flare metrics, dip/transit metrics, robust z-scores, and roughness.
- **Baseline Anomaly Scoring**: Percentile-rank normalized `IsolationScorer` wrapping scikit-learn's IsolationForest with >= 80% injected anomaly retrieval.
- **Consensus Rank Ensembling**: Multi-model percentile-rank fusion algorithm (`rank_average`).
- **Deep Learning Scorers**: CPU-optimized 1D-CNN AutoEncoder and discrete codebook Vector Quantized Variational AutoEncoder (VQ-VAE) with checkpoint serialization.
- **Latent Space Atlas & Rarity**: Morphological clustering (`cluster_latent`) with HDBSCAN and Agglomerative fallback alongside cluster rarity scoring (`rarity_score`).
- **Zero-Network HTML Gallery**: Self-contained SVG curve visualizer (`render_gallery`) with dark-theme styling and zero external CDN/font calls.
- **Typer CLI**: Command line interface with `synthesize` and `demo` pipelines.
- **Pipeline Orchestrator**: Unified end-to-end `PipelineConfig` and `run_pipeline` API.
- **Documentation**: Complete MkDocs Material documentation site and tutorials.
