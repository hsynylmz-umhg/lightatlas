"""Tests for curve resampling and robust normalization."""

import numpy as np
import pytest

from lightatlas.core.preproc import resample_curve, robust_normalize


def test_resample_curve():
    # Test valid resampling and length preservation
    t = np.linspace(0.0, 10.0, 100)
    f = np.sin(t)
    t_res, f_res = resample_curve(t, f, n=1024)

    assert len(t_res) == 1024
    assert len(f_res) == 1024
    assert np.isclose(t_res[0], 0.0)
    assert np.isclose(t_res[-1], 10.0)
    assert np.isclose(f_res[0], f[0], atol=1e-3)
    assert np.isclose(f_res[-1], f[-1], atol=1e-3)

    # Test invalid input rejections
    with pytest.raises(ValueError, match="1-dimensional"):
        resample_curve(np.zeros((10, 2)), np.zeros(10))

    with pytest.raises(ValueError, match="identical length"):
        resample_curve(np.linspace(0, 1, 10), np.linspace(0, 1, 5))

    with pytest.raises(ValueError, match="At least 2 points"):
        resample_curve([1.0], [1.0])

    with pytest.raises(ValueError, match="positive"):
        resample_curve(t, f, n=0)

    with pytest.raises(ValueError, match="monotonically increasing"):
        resample_curve([2.0, 1.0], [1.0, 2.0])


def test_robust_normalize():
    rng = np.random.default_rng(42)

    # Test 1D normalizer statistics on Gaussian distribution
    data = rng.normal(loc=10.0, scale=2.0, size=5000)
    norm = robust_normalize(data, n_sigma=3.0)

    # Median should be ~ 0 and MAD-derived scale ~ 1
    assert np.isclose(np.median(norm), 0.0, atol=0.05)
    norm_mad = np.median(np.abs(norm - np.median(norm)))
    assert np.isclose(1.4826 * norm_mad, 1.0, atol=0.1)

    # Check clipping bounds
    assert np.all(norm >= -3.0)
    assert np.all(norm <= 3.0)

    # Test 2D array support
    data_2d = rng.normal(loc=5.0, scale=1.5, size=(10, 1024))
    norm_2d = robust_normalize(data_2d, n_sigma=3.0)
    assert norm_2d.shape == (10, 1024)
    assert np.all(norm_2d >= -3.0) and np.all(norm_2d <= 3.0)

    # Test constant array fallback
    const_arr = np.full(100, 5.0)
    norm_const = robust_normalize(const_arr, n_sigma=3.0)
    assert np.all(norm_const == 0.0)

    # Test invalid input rejections
    with pytest.raises(ValueError, match="cannot be empty"):
        robust_normalize(np.array([]))

    with pytest.raises(ValueError, match="positive"):
        robust_normalize(data, n_sigma=-1.0)
