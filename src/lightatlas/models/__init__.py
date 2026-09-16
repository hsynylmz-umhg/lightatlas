"""Model implementations, deep learning anomaly scorers, and ensembling routines."""

from lightatlas.models.ensemble import rank_average
from lightatlas.models.isolation import IsolationScorer

try:
    import torch  # noqa: F401

    from lightatlas.models.autoencoder import Conv1DAutoEncoder, ae_anomaly_score, train_autoencoder
    from lightatlas.models.io import load_model, save_model
    from lightatlas.models.vqvae import VQVAE, codebook_usage, train_vqvae, vqvae_anomaly_score

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Conv1DAutoEncoder = None  # type: ignore[assignment,misc]
    ae_anomaly_score = None  # type: ignore[assignment]
    train_autoencoder = None  # type: ignore[assignment]
    load_model = None  # type: ignore[assignment]
    save_model = None  # type: ignore[assignment]
    VQVAE = None  # type: ignore[assignment,misc]
    codebook_usage = None  # type: ignore[assignment]
    train_vqvae = None  # type: ignore[assignment]
    vqvae_anomaly_score = None  # type: ignore[assignment]

__all__ = [
    "HAS_TORCH",
    "IsolationScorer",
    "rank_average",
]
if HAS_TORCH:
    __all__.extend(
        [
            "Conv1DAutoEncoder",
            "VQVAE",
            "ae_anomaly_score",
            "codebook_usage",
            "load_model",
            "save_model",
            "train_autoencoder",
            "train_vqvae",
            "vqvae_anomaly_score",
        ]
    )
