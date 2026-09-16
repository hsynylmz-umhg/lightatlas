"""Tests for 40-feature extraction and physical property verification."""

import numpy as np
from numpy.testing import assert_array_equal

from lightatlas.core.features import FEATURE_NAMES, compute_features, feature_matrix
from lightatlas.synth import _quiet, _transit, make_grid


def test_feature_shape_and_names():
    assert len(FEATURE_NAMES) == 40

    t = make_grid(n=512, dt=0.0304)
    f = 1.0 + 0.05 * np.sin(2.0 * np.pi * t / 3.0)
    feat = compute_features(t, f)

    assert feat.shape == (40,)
    assert feat.dtype == np.float64
    assert not np.isnan(feat).any()
    assert not np.isinf(feat).any()


def test_feature_matrix_shape():
    t = make_grid(n=256, dt=0.0304)
    rng = np.random.default_rng(42)
    f_batch = rng.normal(1.0, 0.01, size=(30, 256))

    mat = feature_matrix(t, f_batch)
    assert mat.shape == (30, 40)
    assert mat.dtype == np.float64

    # Single curve 1D input shape
    single_mat = feature_matrix(t, f_batch[0])
    assert single_mat.shape == (1, 40)


def test_feature_physical_property():
    t = make_grid(n=1024, dt=0.0304)
    rng = np.random.default_rng(42)

    # Transit curve with pronounced dips vs quiet curve with white noise
    curve_quiet = _quiet(t, rng=rng, noise_level=0.002)
    curve_transit = _transit(t, rng=rng, noise_level=0.002)

    feat_quiet = compute_features(t, curve_quiet)
    feat_transit = compute_features(t, curve_transit)

    dip_fraction_idx = FEATURE_NAMES.index("dip_fraction")
    dip_depth_idx = FEATURE_NAMES.index("dip_depth")

    # Transit must exhibit significantly higher dip fraction and dip depth
    assert feat_transit[dip_fraction_idx] > feat_quiet[dip_fraction_idx]
    assert feat_transit[dip_depth_idx] > feat_quiet[dip_depth_idx]


def test_feature_determinism():
    t = make_grid(n=512, dt=0.0304)
    rng = np.random.default_rng(99)
    f = rng.normal(1.0, 0.05, size=512)

    feat1 = compute_features(t, f)
    feat2 = compute_features(t, f)

    assert_array_equal(feat1, feat2)
