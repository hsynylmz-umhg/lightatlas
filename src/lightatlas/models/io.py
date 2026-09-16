"""Serialization and deserialization utilities for LightAtlas PyTorch models."""

from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    HAS_TORCH = False


def save_model(model: Any, path: str | Path, meta: dict[str, Any] | None = None) -> None:
    """Serialize model state dictionary alongside constructor metadata.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to serialize.
    path : str or Path
        Target destination path.
    meta : dict, optional
        Constructor parameters for instantiating the class.
    """
    if not HAS_TORCH or torch is None:
        raise ImportError("PyTorch is required to save models. Install lightatlas[torch].")
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": meta or {},
        "state_dict": model.state_dict(),
        "class_name": model.__class__.__name__,
    }
    torch.save(payload, target_path)


def load_model(cls: type, path: str | Path, device: str = "cpu") -> Any:
    """Instantiate model class with stored metadata and restore weights.

    Parameters
    ----------
    cls : type
        Model class to construct.
    path : str or Path
        Path to saved checkpoint.
    device : str, default="cpu"
        Target device for restored model.

    Returns
    -------
    Any
        Loaded model instance set to eval mode.
    """
    if not HAS_TORCH or torch is None:
        raise ImportError("PyTorch is required to load models. Install lightatlas[torch].")
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    meta = payload.get("meta", {})
    model = cls(**meta)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()

    return model
