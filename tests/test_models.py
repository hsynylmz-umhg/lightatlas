"""Tests for PyTorch 1D-CNN AutoEncoder and VQ-VAE anomaly detection models."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lightatlas.models.autoencoder import (  # noqa: E402
    Conv1DAutoEncoder,
    ae_anomaly_score,
    train_autoencoder,
)
from lightatlas.models.io import load_model, save_model  # noqa: E402
from lightatlas.models.vqvae import (  # noqa: E402
    VQVAE,
    codebook_usage,
    train_vqvae,
    vqvae_anomaly_score,
)
from lightatlas.synth import random_mixture  # noqa: E402


def test_autoencoder_loss_decreases():
    synth = random_mixture(n=128, seed=42, n_points=1024)
    model, losses = train_autoencoder(synth.f, epochs=5, lr=2e-3, seed=42)

    assert len(losses) == 5
    # Verify loss drops by >= 10% after 4 epochs (comparing epoch 5 to epoch 1)
    assert losses[-1] <= 0.90 * losses[0]


def test_autoencoder_score_shape():
    synth = random_mixture(n=64, seed=42, n_points=1024)
    model = Conv1DAutoEncoder(n_points=1024, width=64)
    scores = ae_anomaly_score(model, synth.f)

    assert scores.shape == (64,)
    assert np.isfinite(scores).all()


def test_vqvae_trains_and_uses_codes():
    synth = random_mixture(n=128, seed=42, n_points=1024)
    model, losses = train_vqvae(synth.f, epochs=5, lr=2e-3, seed=42)

    usage = codebook_usage(model, synth.f)
    assert usage > 0.1


def test_vqvae_score_shape():
    synth = random_mixture(n=64, seed=42, n_points=1024)
    model = VQVAE(n_points=1024, latent_dim=16, codebook_size=64, width=64)
    scores = vqvae_anomaly_score(model, synth.f)

    assert scores.shape == (64,)
    assert np.isfinite(scores).all()


def test_save_load_roundtrip(tmp_path):
    synth = random_mixture(n=32, seed=42, n_points=1024)
    model, _ = train_autoencoder(synth.f, epochs=2, seed=42)

    meta = {"n_points": 1024, "width": 64}
    ckpt_path = tmp_path / "ae_model.pt"

    save_model(model, ckpt_path, meta=meta)
    loaded_model = load_model(Conv1DAutoEncoder, ckpt_path)

    orig_scores = ae_anomaly_score(model, synth.f)
    loaded_scores = ae_anomaly_score(loaded_model, synth.f)

    np.testing.assert_allclose(orig_scores, loaded_scores, atol=1e-5)
