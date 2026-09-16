"""Tests for synthetic light curve generation and determinism."""

import time

import numpy as np
from numpy.testing import assert_array_equal
from scipy.signal import periodogram

from lightatlas.synth import _sine, make_grid, random_mixture


def test_synth_determinism():
    set1 = random_mixture(n=50, seed=42, n_points=1024)
    set2 = random_mixture(n=50, seed=42, n_points=1024)

    assert_array_equal(set1.ids, set2.ids)
    assert_array_equal(set1.t, set2.t)
    assert_array_equal(set1.f, set2.f)
    assert_array_equal(set1.labels, set2.labels)


def test_synth_shape():
    synth_set = random_mixture(n=50, seed=42, n_points=1024)
    assert synth_set.f.shape == (50, 1024)
    assert synth_set.t.shape == (1024,)
    assert len(synth_set.ids) == 50
    assert len(synth_set.labels) == 50

    # Verify proportions for canonical n=100 mixture (80% quiet, 10% flare, 5% transit, 5% eclipse)
    set_100 = random_mixture(n=100, seed=42, n_points=1024)
    assert set_100.f.shape == (100, 1024)
    labels_100 = list(set_100.labels)
    assert labels_100.count("quiet") == 80
    assert labels_100.count("flare") == 10
    assert labels_100.count("transit") == 5
    assert labels_100.count("eclipse") == 5


def test_period_recovery():
    t = make_grid(n=1024, dt=0.0304)
    target_period = 2.5
    rng = np.random.default_rng(123)
    curve = _sine(t, rng=rng, period=target_period, amplitude=0.1, noise_level=0.001)

    freqs, psd = periodogram(curve - np.mean(curve), fs=1.0 / 0.0304, nfft=16384)
    best_freq = freqs[np.argmax(psd)]
    recovered_period = 1.0 / best_freq
    relative_error = abs(recovered_period - target_period) / target_period
    assert relative_error < 0.02


def test_generation_speed():
    start = time.perf_counter()
    synth_set = random_mixture(n=1000, seed=42, n_points=1024)
    elapsed = time.perf_counter() - start

    assert synth_set.f.shape == (1000, 1024)
    assert elapsed < 10.0
