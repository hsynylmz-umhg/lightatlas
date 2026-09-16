"""Vector Quantized Variational AutoEncoder (VQ-VAE) for time-series anomaly detection."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, TensorDataset


class VQVAE(nn.Module):
    """VQ-VAE model with 1D-CNN encoder, discrete codebook, and decoder.

    Parameters
    ----------
    n_points : int, default=1024
        Sequence length.
    latent_dim : int, default=16
        Dimensionality of the continuous latent and codebook vectors.
    codebook_size : int, default=64
        Number of discrete codebook entries.
    width : int, default=64
        Base channel width for convolutional operations.
    """

    def __init__(
        self,
        n_points: int = 1024,
        latent_dim: int = 16,
        codebook_size: int = 64,
        width: int = 64,
    ) -> None:
        super().__init__()
        self.n_points = n_points
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.width = width

        # Conv encoder: (B, 1, 1024) -> (B, latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, width // 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(width // 2, width, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(width, width, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(width * 4, latent_dim),
        )

        # Codebook embedding: (codebook_size, latent_dim)
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        nn.init.normal_(self.codebook.weight, 0.0, 0.1)

        # Conv decoder mapping back to (B, 1, 1024)
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, width * 16),
            nn.ReLU(),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(width, width, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(width, width // 2, kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(width // 2, width // 4, kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(width // 4, 1, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input light curves into continuous latent space.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, n_points) or (B, n_points).

        Returns
        -------
        torch.Tensor
            Latent representation tensor of shape (B, latent_dim).
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.encoder(x)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through encoder, quantization, and decoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, n_points) or (B, n_points).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            (recon, vq_loss, active_indices, min_distances)
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)

        z_e = self.encoder(x)  # (B, latent_dim)

        # Compute squared Euclidean distances to all codebook vectors
        d = (
            torch.sum(z_e**2, dim=-1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=-1)
            - 2.0 * torch.matmul(z_e, self.codebook.weight.t())
        )

        indices = torch.argmin(d, dim=-1)  # (B,)
        min_d = torch.min(d, dim=-1).values  # (B,)

        z_q = self.codebook(indices)  # (B, latent_dim)

        # VQ loss with commitment cost = 0.25
        codebook_loss = torch.mean((z_q - z_e.detach()) ** 2)
        commitment_loss = 0.25 * torch.mean((z_q.detach() - z_e) ** 2)
        vq_loss = codebook_loss + commitment_loss

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        h = self.decoder_fc(z_q_st).view(-1, self.width, 16)
        recon = self.decoder_conv(h)

        return recon, vq_loss, indices, min_d


def train_vqvae(
    data: np.ndarray,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
    n_points: int = 1024,
    latent_dim: int = 16,
    codebook_size: int = 64,
    width: int = 64,
) -> tuple[VQVAE, list[float]]:
    """Train VQ-VAE model with deterministic setup and cluster centroid tracking.

    Parameters
    ----------
    data : np.ndarray
        Input light curves of shape (n_samples, n_points).
    epochs : int, default=10
        Number of training epochs.
    batch_size : int, default=64
        Batch size.
    lr : float, default=1e-3
        Adam learning rate.
    seed : int, default=42
        Random seed for reproducibility.
    device : str, default="cpu"
        Device.
    n_points : int, default=1024
        Curve length.
    latent_dim : int, default=16
        Latent dimension.
    codebook_size : int, default=64
        Number of codebook entries.
    width : int, default=64
        Base conv channel width.

    Returns
    -------
    tuple[VQVAE, list[float]]
        Trained model instance and per-epoch average losses.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_arr = np.asarray(data, dtype=np.float32)
    if data_arr.ndim != 2:
        raise ValueError(f"data must be 2D array, got shape {data_arr.shape}")

    seq_len = data_arr.shape[1]
    model = VQVAE(
        n_points=seq_len,
        latent_dim=latent_dim,
        codebook_size=codebook_size,
        width=width,
    ).to(device)
    model.train()

    tensor_data = torch.from_numpy(data_arr).unsqueeze(1)
    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon, vq_l, _, _ = model(batch_x)
            recon_loss = F.mse_loss(recon, batch_x)
            loss = recon_loss + vq_l
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        # Periodic codebook centroid update via MiniBatchKMeans
        with torch.no_grad():
            sub = tensor_data[: min(512, len(tensor_data))].to(device)
            cur_ze = model.encoder(sub).cpu().numpy()
            n_clusters = min(codebook_size, len(cur_ze))
            km = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=seed,
                n_init="auto",
                batch_size=batch_size,
            ).fit(cur_ze)
            model.codebook.weight.data[:n_clusters].copy_(torch.from_numpy(km.cluster_centers_))

        avg_loss = epoch_loss / max(1, n_batches)
        loss_history.append(avg_loss)

    model.eval()
    return model, loss_history


def codebook_usage(model: VQVAE, data: np.ndarray, device: str = "cpu") -> float:
    """Compute fraction of unique codebook vectors activated on data.

    Parameters
    ----------
    model : VQVAE
        Trained model.
    data : np.ndarray
        Array of shape (n_samples, n_points) or (n_points,).
    device : str, default="cpu"
        Device.

    Returns
    -------
    float
        Fraction of activated codebook vectors in [0.0, 1.0].
    """
    model.eval()
    model.to(device)

    data_arr = np.asarray(data, dtype=np.float32)
    if data_arr.ndim == 1:
        data_arr = data_arr[np.newaxis, :]

    tensor_x = torch.from_numpy(data_arr).unsqueeze(1).to(device)
    with torch.no_grad():
        _, _, indices, _ = model(tensor_x)
        unique_active = len(torch.unique(indices))

    return float(unique_active / model.codebook_size)


def vqvae_anomaly_score(
    model: VQVAE,
    data: np.ndarray,
    device: str = "cpu",
    batch_size: int = 128,
) -> np.ndarray:
    """Compute composite anomaly score: std-norm recon error + 0.5 * std-norm quant distance.

    Parameters
    ----------
    model : VQVAE
        Trained model.
    data : np.ndarray
        Array of shape (n_samples, n_points) or (n_points,).
    device : str, default="cpu"
        Device.
    batch_size : int, default=128
        Batch size.

    Returns
    -------
    np.ndarray
        1D float64 array of composite anomaly scores.
    """
    model.eval()
    model.to(device)

    data_arr = np.asarray(data, dtype=np.float32)
    single_curve = data_arr.ndim == 1
    if single_curve:
        data_arr = data_arr[np.newaxis, :]

    n_samples = data_arr.shape[0]
    recon_errs = np.empty(n_samples, dtype=np.float64)
    quant_dists = np.empty(n_samples, dtype=np.float64)

    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = torch.from_numpy(data_arr[start_idx:end_idx]).unsqueeze(1).to(device)
            recon, _, _, min_d = model(batch)
            mse = torch.mean((batch - recon) ** 2, dim=(-1, -2))
            recon_errs[start_idx:end_idx] = mse.cpu().numpy().astype(np.float64)
            quant_dists[start_idx:end_idx] = min_d.cpu().numpy().astype(np.float64)

    # Standard-normalize across the dataset
    norm_recon = (recon_errs - np.mean(recon_errs)) / (np.std(recon_errs) + 1e-12)
    norm_quant = (quant_dists - np.mean(quant_dists)) / (np.std(quant_dists) + 1e-12)

    composite = norm_recon + 0.5 * norm_quant
    return composite
