"""Deterministic synthetic light curve factory for astrophysical time-series."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticSet:
    """Container for synthetic light curve dataset.

    Attributes
    ----------
    ids : np.ndarray
        Array of curve identifiers (shape (n,)).
    t : np.ndarray
        Array of uniform timestamps (shape (n_points,)).
    f : np.ndarray
        Array of flux values (shape (n, n_points)).
    labels : np.ndarray
        Array of category labels (shape (n,)).
    """

    ids: np.ndarray
    t: np.ndarray
    f: np.ndarray
    labels: np.ndarray


def make_grid(n: int = 1024, dt: float = 0.0304) -> np.ndarray:
    """Generate a uniform 1D time grid.

    Parameters
    ----------
    n : int, default=1024
        Number of time points.
    dt : float, default=0.0304
        Sampling cadence in days.

    Returns
    -------
    np.ndarray
        Uniformly spaced time array of length n.

    Raises
    ------
    ValueError
        If n or dt is not positive.
    """
    if n <= 0:
        raise ValueError(f"Number of points n must be positive, got {n}")
    if dt <= 0:
        raise ValueError(f"Sampling cadence dt must be positive, got {dt}")
    return np.arange(n, dtype=np.float64) * dt


def _sine(
    t: np.ndarray,
    rng: np.random.Generator | None = None,
    period: float | None = None,
    amplitude: float | None = None,
    phase: float | None = None,
    noise_level: float = 0.005,
) -> np.ndarray:
    """Generate a periodic sinusoidal variable star light curve."""
    if rng is None:
        rng = np.random.default_rng(42)
    p = float(rng.uniform(0.5, 5.0)) if period is None else float(period)
    amp = float(rng.uniform(0.02, 0.10)) if amplitude is None else float(amplitude)
    phi = float(rng.uniform(0.0, 2.0 * np.pi)) if phase is None else float(phase)
    signal = 1.0 + amp * np.sin(2.0 * np.pi * t / p + phi)
    noise = rng.normal(0.0, noise_level, size=len(t))
    return signal + noise


def _flare(
    t: np.ndarray,
    rng: np.random.Generator | None = None,
    noise_level: float = 0.005,
) -> np.ndarray:
    """Generate a stellar flare light curve with rapid rise and exponential decay."""
    if rng is None:
        rng = np.random.default_rng(42)
    n_pts = len(t)
    t_peak = float(rng.uniform(t[n_pts // 4], t[3 * n_pts // 4]))
    amplitude = float(rng.uniform(0.1, 0.8))
    decay = float(rng.uniform(0.05, 0.3))
    rise = decay / 5.0

    flare = np.zeros_like(t)
    mask_rise = t < t_peak
    mask_decay = ~mask_rise
    flare[mask_rise] = amplitude * np.exp((t[mask_rise] - t_peak) / rise)
    flare[mask_decay] = amplitude * np.exp(-(t[mask_decay] - t_peak) / decay)

    signal = 1.0 + flare
    noise = rng.normal(0.0, noise_level, size=n_pts)
    return signal + noise


def _transit(
    t: np.ndarray,
    rng: np.random.Generator | None = None,
    noise_level: float = 0.005,
) -> np.ndarray:
    """Generate an exoplanetary transit light curve with a U-shaped dip."""
    if rng is None:
        rng = np.random.default_rng(42)
    n_pts = len(t)
    t_center = float(rng.uniform(t[n_pts // 4], t[3 * n_pts // 4]))
    depth = float(rng.uniform(0.01, 0.05))
    duration = float(rng.uniform(0.1, 0.4))

    sigma = duration / 2.355
    dip = depth * np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    signal = 1.0 - dip
    noise = rng.normal(0.0, noise_level, size=n_pts)
    return signal + noise


def _eclipse(
    t: np.ndarray,
    rng: np.random.Generator | None = None,
    noise_level: float = 0.005,
) -> np.ndarray:
    """Generate an eclipsing binary light curve with primary and secondary dips."""
    if rng is None:
        rng = np.random.default_rng(42)
    period = float(rng.uniform(1.0, 4.0))
    phase = (t % period) / period

    depth1 = float(rng.uniform(0.08, 0.25))
    depth2 = depth1 * float(rng.uniform(0.3, 0.7))
    width = float(rng.uniform(0.04, 0.08))

    dip1 = depth1 * np.exp(-0.5 * ((phase - 0.25) / width) ** 2)
    dip2 = depth2 * np.exp(-0.5 * ((phase - 0.75) / width) ** 2)

    signal = 1.0 - dip1 - dip2
    noise = rng.normal(0.0, noise_level, size=len(t))
    return signal + noise


def _quiet(
    t: np.ndarray,
    rng: np.random.Generator | None = None,
    noise_level: float = 0.005,
) -> np.ndarray:
    """Generate a quiet non-variable star light curve with photometric noise."""
    if rng is None:
        rng = np.random.default_rng(42)
    return 1.0 + rng.normal(0.0, noise_level, size=len(t))


def random_mixture(
    n: int = 100,
    seed: int = 42,
    n_points: int = 1024,
    dt: float = 0.0304,
    quiet_ratio: float = 0.80,
    flare_ratio: float = 0.10,
    transit_ratio: float = 0.05,
    eclipse_ratio: float = 0.05,
) -> SyntheticSet:
    """Generate a reproducible synthetic light curve mixture.

    Generates approximately 80% quiet, 10% flare, 5% transit, and 5% eclipse curves.

    Parameters
    ----------
    n : int, default=100
        Number of light curves to generate.
    seed : int, default=42
        Seed for random number generator to guarantee reproducibility.
    n_points : int, default=1024
        Number of time points per curve.
    dt : float, default=0.0304
        Sampling cadence in days.

    Returns
    -------
    SyntheticSet
        Container with ids, t, f, and labels.

    Raises
    ------
    ValueError
        If n or n_points is not positive.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n_points <= 0:
        raise ValueError(f"n_points must be positive, got {n_points}")

    rng = np.random.default_rng(seed)
    t = make_grid(n=n_points, dt=dt)

    n_flare = int(round(n * flare_ratio))
    n_transit = int(round(n * transit_ratio))
    n_eclipse = int(round(n * eclipse_ratio))
    n_quiet = n - (n_flare + n_transit + n_eclipse)

    category_counts = [
        ("quiet", n_quiet, _quiet),
        ("flare", n_flare, _flare),
        ("transit", n_transit, _transit),
        ("eclipse", n_eclipse, _eclipse),
    ]

    flux_list: list[np.ndarray] = []
    labels_list: list[str] = []

    for label, count, gen_fn in category_counts:
        for _ in range(count):
            flux_list.append(gen_fn(t, rng))
            labels_list.append(label)

    f_arr = np.vstack(flux_list) if flux_list else np.empty((0, n_points), dtype=np.float64)
    labels_arr = np.array(labels_list, dtype=object)

    indices = rng.permutation(n)
    f_shuffled = f_arr[indices]
    labels_shuffled = labels_arr[indices]

    ids_arr = np.array([f"synth_{i:06d}" for i in range(n)], dtype=object)

    return SyntheticSet(ids=ids_arr, t=t, f=f_shuffled, labels=labels_shuffled)
