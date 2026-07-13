"""Tests for prg_toolbox.observables."""

import numpy as np
import pytest

from prg_toolbox.coarse_graining import CGVariables
from prg_toolbox import observables as obs


ALL_OBSERVABLE_CLASSES = [
    obs.mean_variance,
    obs.log_silence_probability,
    obs.max_covariance_eigenvalue,
    obs.covariance_spectrum,
    obs.autocorrelation_function,
    obs.decay_time,
    obs.activity_distribution,
]


@pytest.mark.parametrize("observable_cls", ALL_OBSERVABLE_CLASSES)
def test_rejects_non_cgvariables_input(observable_cls):
    with pytest.raises(ValueError, match="CG_variables"):
        observable_cls(np.zeros((4, 10)))


@pytest.mark.parametrize("observable_cls", ALL_OBSERVABLE_CLASSES)
def test_values_length_matches_rg_steps(observable_cls, cgvars):
    result = observable_cls(cgvars)
    assert len(result.values) == cgvars.rg_steps + 1
    assert result.rg_steps == cgvars.rg_steps


class TestMeanVariance:
    def test_variance_roughly_doubles_each_step_for_independent_units(self, rng):
        # Use cluster_method='random' so pairing carries no correlation bias:
        # for independent Bernoulli units, Var(X+Y) = Var(X) + Var(Y), so mean
        # variance should approximately double at each RG step.
        N, T = 64, 4000
        binary = (rng.random((N, T)) < 0.3).astype(int)
        cg = CGVariables(binary, cluster_method="random", rg_steps=4)
        result = obs.mean_variance(cg)
        ratios = result.values[1:] / result.values[:-1]
        np.testing.assert_allclose(ratios, 2.0, rtol=0.15)
        assert result.exponent == pytest.approx(1.0, abs=0.15)


class TestLogSilenceProbability:
    def test_warns_when_never_silent(self, rng):
        # High per-unit activity (rarely silent) but not constant, so rows
        # survive variance filtering while still almost never producing a
        # fully-silent coarse-grained block at later RG steps.
        N, T = 16, 3000
        binary = (rng.random((N, T)) < 0.95).astype(int)
        cg = CGVariables(binary, cluster_method="random", rg_steps=3)
        with pytest.warns(UserWarning, match="no silence"):
            obs.log_silence_probability(cg)

    def test_positive_values_for_sparse_activity(self, cgvars):
        result = obs.log_silence_probability(cgvars)
        assert np.all(result.values >= 0)


class TestMaxCovarianceEigenvalue:
    def test_first_step_value_is_trivial_zero(self, cgvars):
        result = obs.max_covariance_eigenvalue(cgvars)
        assert result.values[0] == 0.0

    def test_eigenvalues_are_non_negative(self, cgvars):
        result = obs.max_covariance_eigenvalue(cgvars)
        assert np.all(result.values >= 0)


class TestCovarianceSpectrum:
    def test_spectrum_length_matches_cluster_size_at_each_step(self, cgvars):
        result = obs.covariance_spectrum(cgvars)
        for k, spectrum in enumerate(result.values):
            assert len(spectrum) == 2 ** k

    def test_marchenko_pastur_fields_are_populated(self, cgvars):
        result = obs.covariance_spectrum(cgvars)
        assert result.mp_lambda_plus > 0
        assert result.mp_x_fit.shape == result.mp_y_fit.shape

    def test_distribution_power_law_returns_nan_for_too_few_tail_points(self):
        # Construct a bare instance and set only the attributes
        # get_distribution_power_law reads, so the < 3 tail points fallback
        # (documented in the source) can be tested directly and reliably.
        instance = obs.covariance_spectrum.__new__(obs.covariance_spectrum)
        instance.avg_across_windows = [np.array([5.0, 4.0, 3.0, 2.0, 1.0])]
        instance.mp_lambda_plus = 10.0  # above every eigenvalue -> empty tail

        exponent, normalization = instance.get_distribution_power_law()

        assert np.isnan(exponent)
        assert np.isnan(normalization)

    def test_distribution_power_law_fits_when_enough_tail_points(self):
        instance = obs.covariance_spectrum.__new__(obs.covariance_spectrum)
        instance.avg_across_windows = [np.array([50.0, 20.0, 15.0, 12.0, 10.0, 1.0])]
        instance.mp_lambda_plus = 5.0  # tail: 50, 20, 15, 12, 10 (5 points)

        exponent, normalization = instance.get_distribution_power_law()

        assert np.isfinite(exponent)
        assert np.isfinite(normalization)


class TestAutocorrelationFunction:
    def test_zero_lag_is_normalized_to_one(self, cgvars):
        result = obs.autocorrelation_function(cgvars)
        for ac in result.values:
            assert np.max(ac) == pytest.approx(1.0)


class TestDecayTime:
    def test_decaying_autocorrelation_gives_positive_decay_time(self):
        instance = obs.decay_time.__new__(obs.decay_time)
        T = 50
        x = np.arange(T)
        decaying = np.exp(-x / 5.0)
        ac_values = [np.concatenate([decaying[::-1], decaying[1:]])]
        result = instance.get_decay_time(ac_values, nbins=10, rg_steps=1)
        assert result[0] > 0

    def test_flat_autocorrelation_gives_zero_decay_time(self):
        instance = obs.decay_time.__new__(obs.decay_time)
        T = 50
        flat = np.ones(2 * T - 1)
        result = instance.get_decay_time([flat], nbins=10, rg_steps=1)
        assert result[0] == 0

    def test_growing_autocorrelation_gives_zero_decay_time(self):
        instance = obs.decay_time.__new__(obs.decay_time)
        x = np.arange(50)
        growing = np.exp(x / 5.0)
        ac_values = [np.concatenate([growing[::-1], growing[1:]])]
        result = instance.get_decay_time(ac_values, nbins=10, rg_steps=1)
        assert result[0] == 0

    def test_end_to_end_smoke(self, cgvars):
        result = obs.decay_time(cgvars)
        assert len(result.values) == cgvars.rg_steps + 1


class TestActivityDistribution:
    def test_density_integrates_to_one(self, cgvars):
        result = obs.activity_distribution(cgvars)
        for k, density in enumerate(result.values):
            dx = 1 / (2 ** k)
            assert np.sum(density) * dx == pytest.approx(1.0)


class TestAvalancheCovarianceEigenvalue:
    """Regression tests for the fixed CGVariables-instance-as-constructor bug."""

    def test_runs_without_error(self, cgvars):
        result = obs._avalanche_covariance_eigenvalue(cgvars)
        assert len(result.values) == cgvars.rg_steps + 1
        assert np.isfinite(result.exponent)

    def test_uses_same_cluster_method_and_rg_steps_as_input(self, cgvars, monkeypatch):
        captured = {}
        original_init = CGVariables.__init__

        def spy_init(self, binary_array, cluster_method="pearson", rg_steps=6):
            captured["cluster_method"] = cluster_method
            captured["rg_steps"] = rg_steps
            return original_init(self, binary_array, cluster_method=cluster_method, rg_steps=rg_steps)

        monkeypatch.setattr(CGVariables, "__init__", spy_init)
        obs._avalanche_covariance_eigenvalue(cgvars)

        assert captured["cluster_method"] == cgvars.cluster_method
        assert captured["rg_steps"] == cgvars.rg_steps
