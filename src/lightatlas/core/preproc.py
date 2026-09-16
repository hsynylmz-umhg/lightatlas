"""Light curve preprocessing and normalization utilities."""

import numpy as np


def resample_curve(
    t: np.ndarray,
    f: np.ndarray,
    n: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a light curve onto a uniform time grid using linear interpolation.

    Parameters
    ----------
    t : np.ndarray
        Observed time stamps (1D array). Must be strictly monotonically increasing.
    f : np.ndarray
        Observed flux values (1D array) with matching length to t.
    n : int, default=1024
        Target number of evenly spaced points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (t_new, f_new) uniform time grid and linearly interpolated flux values.

    Raises
    ------
    ValueError
        If inputs have invalid shapes, unequal lengths, fewer than 2 points,
        non-positive n, or non-monotonic time values.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    f_arr = np.asarray(f, dtype=np.float64)

    if t_arr.ndim != 1 or f_arr.ndim != 1:
        raise ValueError(
            f"t and f must be 1-dimensional arrays, got t.ndim={t_arr.ndim}, f.ndim={f_arr.ndim}"
        )
    if len(t_arr) != len(f_arr):
        raise ValueError(
            f"t and f must have identical length, got len(t)={len(t_arr)}, len(f)={len(f_arr)}"
        )
    if len(t_arr) < 2:
        raise ValueError(f"At least 2 points are required for interpolation, got {len(t_arr)}")
    if n <= 0:
        raise ValueError(f"Target number of points n must be positive, got {n}")
    if np.any(np.diff(t_arr) <= 0):
        raise ValueError("Time array t must be strictly monotonically increasing")

    t_new = np.linspace(t_arr[0], t_arr[-1], n)
    f_new = np.interp(t_new, t_arr, f_arr)
    return t_new, f_new


def robust_normalize(
    f: np.ndarray,
    n_sigma: float = 3.0,
) -> np.ndarray:
    """Normalize flux using median and MAD (1.4826 scale), clipped to [-n_sigma, n_sigma].

    Parameters
    ----------
    f : np.ndarray
        Flux values (1D or 2D array).
    n_sigma : float, default=3.0
        Clipping threshold in units of robust standard deviations.

    Returns
    -------
    np.ndarray
        Normalized and clipped flux array with the same shape as input.

    Raises
    ------
    ValueError
        If input array is empty or n_sigma <= 0.
    """
    f_arr = np.asarray(f, dtype=np.float64)

    if f_arr.size == 0:
        raise ValueError("Input array f cannot be empty")
    if n_sigma <= 0:
        raise ValueError(f"n_sigma must be positive, got {n_sigma}")

    med = np.median(f_arr, axis=-1, keepdims=True)
    mad = np.median(np.abs(f_arr - med), axis=-1, keepdims=True)
    scale = 1.4826 * mad

    # Handle constant signal where MAD is 0
    scale = np.where(scale < 1e-12, 1.0, scale)

    normalized = (f_arr - med) / scale
    clipped = np.clip(normalized, -n_sigma, n_sigma)

    return clipped.squeeze() if f_arr.ndim == 1 else clipped
