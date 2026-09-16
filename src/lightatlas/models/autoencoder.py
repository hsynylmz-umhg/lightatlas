"""1D-CNN AutoEncoder anomaly scorer with deterministic training."""

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    HAS_TORCH = False

_BaseModule = nn.Module if nn is not None else object


class Conv1DAutoEncoder(_BaseModule):
    """3-layer 1D-CNN encoder and symmetric decoder preserving sequence length.

    Parameters
    ----------
    n_points : int, default=1024
        Sequence length of the light curve.
    width : int, default=64
        Base channel width of convolutional layers.
    """

    def __init__(self, n_points: int = 1024, width: int = 64) -> None:
        super().__init__()
        self.n_points = n_points
        self.width = width

        # 3-layer 1D-CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, width // 2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(width // 2, width, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(width, width, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )

        # Symmetric 3-layer 1D-CNN decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(width, width // 2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(width // 2, 1, kernel_size=5, stride=1, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass reconstructing the 1D sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, n_points) or (B, n_points).

        Returns
        -------
        torch.Tensor
            Reconstructed tensor of shape (B, 1, n_points).
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        latent = self.encoder(x)
        return self.decoder(latent)


def train_autoencoder(
    data: np.ndarray,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
    n_points: int = 1024,
    width: int = 64,
) -> tuple[Conv1DAutoEncoder, list[float]]:
    """Train Conv1DAutoEncoder reproducibly on light curve data.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_samples, n_points) containing light curves.
    epochs : int, default=5
        Number of training epochs.
    batch_size : int, default=64
        Batch size.
    lr : float, default=1e-3
        Learning rate for Adam optimizer.
    seed : int, default=42
        Seed for reproducibility.
    device : str, default="cpu"
        Torch device ("cpu" or "cuda").
    n_points : int, default=1024
        Sequence length.
    width : int, default=64
        Base channel width.

    Returns
    -------
    tuple[Conv1DAutoEncoder, list[float]]
        Trained model instance and per-epoch average MSE losses.
    """
    if not HAS_TORCH or torch is None:
        raise ImportError(
            "PyTorch is required to train the autoencoder. Install lightatlas[torch]."
        )
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_arr = np.asarray(data, dtype=np.float32)
    if data_arr.ndim != 2:
        raise ValueError(f"data must be 2D array, got shape {data_arr.shape}")

    seq_len = data_arr.shape[1]
    model = Conv1DAutoEncoder(n_points=seq_len, width=width).to(device)
    model.train()

    tensor_data = torch.from_numpy(data_arr).unsqueeze(1)
    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        loss_history.append(avg_loss)

    model.eval()
    return model, loss_history


def ae_anomaly_score(
    model: Conv1DAutoEncoder,
    data: np.ndarray,
    device: str = "cpu",
    batch_size: int = 128,
) -> np.ndarray:
    """Compute per-curve point-wise MSE reconstruction anomaly score.

    Parameters
    ----------
    model : Conv1DAutoEncoder
        Trained autoencoder.
    data : np.ndarray
        Array of shape (n_samples, n_points) or (n_points,).
    device : str, default="cpu"
        Device to run inference on.
    batch_size : int, default=128
        Batch size for inference.

    Returns
    -------
    np.ndarray
        1D float64 array of MSE reconstruction scores.
    """
    if not HAS_TORCH or torch is None:
        raise ImportError(
            "PyTorch is required to run autoencoder scoring. Install lightatlas[torch]."
        )
    model.eval()
    model.to(device)

    data_arr = np.asarray(data, dtype=np.float32)
    single_curve = data_arr.ndim == 1
    if single_curve:
        data_arr = data_arr[np.newaxis, :]

    n_samples = data_arr.shape[0]
    scores = np.empty(n_samples, dtype=np.float64)

    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = torch.from_numpy(data_arr[start_idx:end_idx]).unsqueeze(1).to(device)
            recon = model(batch)
            mse = torch.mean((batch - recon) ** 2, dim=(-1, -2))
            scores[start_idx:end_idx] = mse.cpu().numpy().astype(np.float64)

    return scores
