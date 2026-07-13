"""Tests for prg_toolbox.utils: math primitives and power-law fitting."""

import numpy as np
import pytest

from prg_toolbox.utils import (
    covariance_evals_and_evectors,
    powerLaw_function,
    get_scaling_exponent,
)


class TestCovarianceEvalsAndEvectors:
    def test_eigenvalues_match_known_diagonal_covariance(self, rng):
        # Independent Gaussian variables with known variances -> the sample
        # covariance's eigenvalues should approximate those variances
        # (loose tolerance since this is finite-sample estimation).
        n_samples = 200_000
        variances = np.array([9.0, 4.0, 1.0])
        x = rng.normal(scale=np.sqrt(variances)[:, None], size=(3, n_samples))
        evals, evecs = covariance_evals_and_evectors(x)
        assert evals[0] > evals[1] > evals[2]  # sorted descending
        np.testing.assert_allclose(evals, variances, rtol=0.05)

    def test_eigenvectors_are_orthonormal(self, rng):
        x = rng.normal(size=(5, 500))
        _, evecs = covariance_evals_and_evectors(x)
        np.testing.assert_allclose(evecs.T @ evecs, np.eye(5), atol=1e-8)

    def test_is_invariant_to_a_constant_shift(self, rng):
        x = rng.normal(size=(4, 300))
        evals_a, _ = covariance_evals_and_evectors(x)
        evals_b, _ = covariance_evals_and_evectors(x + 100.0)
        np.testing.assert_allclose(evals_a, evals_b)


class TestPowerLawFunction:
    def test_basic_value(self):
        assert powerLaw_function(2.0, a=3.0, b=2.0) == pytest.approx(12.0)

    def test_exponent_zero_is_constant(self):
        assert powerLaw_function(np.array([1.0, 5.0, 100.0]), a=7.0, b=0.0) == pytest.approx([7.0, 7.0, 7.0])


class TestGetScalingExponent:
    def test_recovers_exact_power_law_exponent(self):
        b_true, a_true = -1.5, 3.0
        k = np.arange(6)
        x = 2 ** k
        y = a_true * x.astype(float) ** b_true

        exponent, exponent_error, r2 = get_scaling_exponent(y)

        assert exponent == pytest.approx(b_true, abs=1e-8)
        assert r2 == pytest.approx(1.0, abs=1e-8)
        assert exponent_error == pytest.approx(0.0, abs=1e-6)

    def test_skip_first_value_uses_shifted_x_axis(self):
        # skip_first_value=True builds t = 2**(i+1); feed values already
        # trimmed of a trivial first point and confirm the fit still
        # recovers the exact exponent against that shifted axis.
        b_true, a_true = 2.0, 0.5
        k = np.arange(1, 6)
        x = 2.0 ** k
        y = a_true * x ** b_true

        exponent, _, r2 = get_scaling_exponent(y, skip_first_value=True)

        assert exponent == pytest.approx(b_true, abs=1e-8)
        assert r2 == pytest.approx(1.0, abs=1e-8)

    def test_spectrum_uses_linear_rank_x_axis(self):
        b_true, a_true = -0.8, 2.0
        rank = np.arange(1, 11)  # spectrum: t = arange(len)+1
        y = a_true * rank.astype(float) ** b_true

        exponent, _, r2 = get_scaling_exponent(y, spectrum=True)

        assert exponent == pytest.approx(b_true, abs=1e-8)
        assert r2 == pytest.approx(1.0, abs=1e-8)

    def test_noisy_data_gives_imperfect_but_reasonable_fit(self, rng):
        b_true, a_true = 1.2, 1.0
        k = np.arange(8)
        x = 2.0 ** k
        y = a_true * x ** b_true
        noisy_y = y * rng.lognormal(sigma=0.05, size=y.shape)

        exponent, _, r2 = get_scaling_exponent(noisy_y)

        assert exponent == pytest.approx(b_true, abs=0.1)
        assert 0.9 < r2 <= 1.0
