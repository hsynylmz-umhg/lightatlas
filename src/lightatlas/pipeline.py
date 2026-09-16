"""Unified pipeline orchestration for LightAtlas anomaly discovery and atlas creation."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from lightatlas.atlas.cluster import cluster_latent
from lightatlas.atlas.rank import rarity_score
from lightatlas.atlas.report import render_gallery
from lightatlas.core.features import feature_matrix
from lightatlas.core.preproc import robust_normalize
from lightatlas.models.ensemble import rank_average
from lightatlas.models.isolation import IsolationScorer
from lightatlas.synth import random_mixture


class PipelineConfig(BaseModel):
    """Configuration specification for the LightAtlas anomaly discovery pipeline."""

    source: Literal["synthetic", "tess"] = "synthetic"
    n_curves: int = Field(default=1000, gt=0, description="Number of curves to process")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    n_points: int = Field(default=1024, gt=1, description="Number of points per curve")
    ae_epochs: int = Field(default=5, ge=1, description="Autoencoder training epochs")
    vqvae_epochs: int = Field(default=10, ge=1, description="VQ-VAE training epochs")
    min_cluster_size: int = Field(default=10, ge=2, description="Minimum cluster size")
    top_fraction: float = Field(
        default=0.05, gt=0.0, le=1.0, description="Top anomaly fraction for gallery"
    )
    out_dir: Path = Field(
        default=Path("runs/default"), description="Destination directory for output artifacts"
    )


def run_pipeline(cfg: PipelineConfig) -> dict[str, Path]:
    """Execute the end-to-end LightAtlas anomaly pipeline.

    Sequence:
    1. Data generation (synthetic mixture or TESS stub).
    2. Robust normalization.
    3. Layer 1 feature extraction & Isolation Forest scoring.
    4. Layer 2 deep anomaly scorers (1D-CNN AutoEncoder & VQ-VAE with PyTorch).
    5. Consensus percentile rank ensembling.
    6. Latent extraction and morphological clustering.
    7. Rarity ranking.
    8. Parquet serialization and zero-network HTML gallery generation.

    Parameters
    ----------
    cfg : PipelineConfig
        Execution configuration parameters.

    Returns
    -------
    dict[str, Path]
        Dictionary with paths to generated 'scores' Parquet and 'gallery' HTML.
    """
    out_path = Path(cfg.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Data generation
    if cfg.source == "tess":
        # TESS 2-minute cadence simulation stub
        synth_set = random_mixture(
            n=cfg.n_curves,
            seed=cfg.seed,
            n_points=cfg.n_points,
            dt=0.00138,
        )
        ids = np.array([f"TIC_{10000000 + i}" for i in range(cfg.n_curves)], dtype=object)
        t = synth_set.t
        f_raw = synth_set.f
        labels = synth_set.labels
    else:
        synth_set = random_mixture(
            n=cfg.n_curves,
            seed=cfg.seed,
            n_points=cfg.n_points,
        )
        ids = synth_set.ids
        t = synth_set.t
        f_raw = synth_set.f
        labels = synth_set.labels

    # 2. Normalization
    f_norm = robust_normalize(f_raw)

    # 3. Layer 1: 40-feature extraction + Isolation Forest
    X_features = feature_matrix(t, f_norm)
    iso_scorer = IsolationScorer(contamination=cfg.top_fraction, random_state=cfg.seed)
    iso_scorer.fit(X_features)
    iso_scores = iso_scorer.score(X_features)

    # 4. Layer 2: Deep Learning Models (PyTorch)
    torch_available = False
    try:
        import torch

        from lightatlas.models.autoencoder import ae_anomaly_score, train_autoencoder
        from lightatlas.models.vqvae import train_vqvae, vqvae_anomaly_score

        torch_available = True
    except ImportError:
        pass

    if torch_available:
        ae_model, _ = train_autoencoder(
            f_norm,
            epochs=cfg.ae_epochs,
            seed=cfg.seed,
        )
        ae_scores = ae_anomaly_score(ae_model, f_norm)

        vq_model, _ = train_vqvae(
            f_norm,
            epochs=cfg.vqvae_epochs,
            seed=cfg.seed,
        )
        vq_scores = vqvae_anomaly_score(vq_model, f_norm)

        # 5. Composite Consensus Score
        scores_map = {
            "iso": iso_scores,
            "autoencoder": ae_scores,
            "vqvae": vq_scores,
        }
        composite_scores = rank_average(scores_map)

        # Latent extraction from VQ-VAE
        with torch.no_grad():
            tensor_input = torch.as_tensor(f_norm, dtype=torch.float32)
            Z_latent = vq_model.encode(tensor_input).cpu().numpy()
    else:
        composite_scores = iso_scores
        from sklearn.decomposition import PCA

        pca = PCA(n_components=min(8, X_features.shape[1]), random_state=cfg.seed)
        Z_latent = pca.fit_transform(X_features)

    # 6. Latent Clustering
    clusters = cluster_latent(
        Z_latent,
        min_cluster_size=cfg.min_cluster_size,
        random_state=cfg.seed,
    )

    # 7. Morphological Rarity
    rarities = rarity_score(clusters)
    sample_rarities = [rarities.get(int(c), 0.0) for c in clusters]

    # Build and sort DataFrame
    df = pd.DataFrame(
        {
            "id": ids,
            "label": labels,
            "score": composite_scores,
            "cluster": clusters,
            "rarity": sample_rarities,
            "f": list(f_raw),
        }
    )
    df = df.sort_values(by="score", ascending=False).reset_index(drop=True)

    # 8. Output Artifacts
    parquet_path = out_path / "scores.parquet"
    df.to_parquet(parquet_path)

    gallery_count = max(1, int(len(df) * cfg.top_fraction))
    df_gallery = df.head(gallery_count).reset_index(drop=True)
    gallery_path = out_path / "gallery.html"
    render_gallery(df_gallery, gallery_path)

    return {
        "scores": parquet_path,
        "gallery": gallery_path,
    }
