"""Shared pytest fixtures for the prg_toolbox test suite."""

import numpy as np
import pytest

from prg_toolbox import CGVariables


@pytest.fixture
def rng():
    """A fixed-seed random number generator for reproducible test data."""
    return np.random.default_rng(0)


@pytest.fixture
def random_binary_array(rng):
    """
    A (32, 500) i.i.d. Bernoulli(0.3) binary array with no constant rows.

    N=32 (rg_steps=5) is large enough that covariance_spectrum's default
    tail fit (spectrum_fit_length=1/5) has enough points for
    np.polyfit(..., cov=True), which requires more data points than the
    polynomial order.
    """
    N, T = 32, 500
    while True:
        arr = (rng.random((N, T)) < 0.3).astype(int)
        # Guarantee no zero-variance rows so shapes stay predictable.
        if np.all(arr.sum(axis=1) > 0) and np.all(arr.sum(axis=1) < T):
            return arr


@pytest.fixture
def duplicated_pairs_binary_array(rng):
    """
    An (8, 300) binary array built from 4 independent base rows, each
    duplicated once. Pearson correlation trivially pairs each row with
    its exact duplicate, giving deterministic, hand-verifiable lineage.
    """
    T = 300
    base = (rng.random((4, T)) < 0.3).astype(int)
    return np.repeat(base, repeats=2, axis=0)


@pytest.fixture
def binary_array_with_constant_rows(rng):
    """A (10, 200) binary array with 2 all-zero and 1 all-one constant rows."""
    T = 200
    arr = (rng.random((10, T)) < 0.3).astype(int)
    arr[2] = 0
    arr[5] = 0
    arr[7] = 1
    return arr


@pytest.fixture
def binary_array_with_nan_row(rng):
    """A (10, 200) binary array (as float) with one row entirely NaN."""
    T = 200
    arr = (rng.random((10, T)) < 0.3).astype(float)
    arr[4, :] = np.nan
    return arr


@pytest.fixture
def odd_binary_array(rng):
    """A (7, 200) binary array (odd N) with no constant rows."""
    N, T = 7, 200
    while True:
        arr = (rng.random((N, T)) < 0.3).astype(int)
        if np.all(arr.sum(axis=1) > 0) and np.all(arr.sum(axis=1) < T):
            return arr


@pytest.fixture
def cgvars(random_binary_array):
    """A populated CGVariables instance over random_binary_array, 5 RG steps."""
    return CGVariables(random_binary_array, cluster_method="pearson", rg_steps=5)
