"""Tests for prg_toolbox.coarse_graining: similarity metrics and CGVariables."""

import numpy as np
import pytest

from prg_toolbox.coarse_graining import (
    CGVariables,
    MSVariables,
    _compute_cosine,
    _compute_hamming,
    _compute_mutual_information,
    _compute_pearson,
    _compute_random,
    _compute_spearman,
)

# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------


class TestSimilarityMetrics:
    def test_pearson_identical_rows_give_correlation_one(self):
        row = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=float)
        X = np.vstack([row, row])
        corr = _compute_pearson(X)
        assert corr.shape == (2, 2)
        np.testing.assert_allclose(np.diag(corr), 1.0)
        assert corr[0, 1] == pytest.approx(1.0)

    def test_pearson_anticorrelated_rows_give_correlation_minus_one(self):
        row = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        X = np.vstack([row, -row])
        corr = _compute_pearson(X)
        assert corr[0, 1] == pytest.approx(-1.0)

    def test_pearson_is_symmetric(self, rng):
        X = rng.random((6, 50))
        corr = _compute_pearson(X)
        np.testing.assert_allclose(corr, corr.T)

    def test_spearman_monotonic_relationship_gives_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = x**3  # any strictly increasing transform preserves rank
        X = np.vstack([x, y])
        corr = _compute_spearman(X)
        assert corr[0, 1] == pytest.approx(1.0)

    def test_cosine_orthogonal_vectors_give_zero(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = _compute_cosine(X)
        assert sim[0, 1] == pytest.approx(0.0, abs=1e-10)

    def test_cosine_identical_vectors_give_one(self):
        row = np.array([1.0, 2.0, 3.0, 4.0])
        X = np.vstack([row, row])
        sim = _compute_cosine(X)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-10)

    def test_hamming_identical_binary_rows_match_every_timestep(self):
        row = np.array([0, 1, 1, 0, 1])
        X = np.vstack([row, row])
        match_counts = _compute_hamming(X)
        assert match_counts[0, 1] == len(row)

    def test_hamming_fully_opposite_rows_match_nowhere(self):
        row = np.array([0, 1, 1, 0, 1])
        X = np.vstack([row, 1 - row])
        match_counts = _compute_hamming(X)
        assert match_counts[0, 1] == 0

    def test_mutual_information_identical_rows_exceeds_independent_rows(self, rng):
        # Identical (fully dependent) variables should show strictly higher MI
        # than two independently drawn binary variables.
        row = (rng.random(400) < 0.4).astype(int)
        independent = (rng.random(400) < 0.4).astype(int)
        X_dependent = np.vstack([row, row])
        X_independent = np.vstack([row, independent])
        mi_dependent = _compute_mutual_information(X_dependent)[0, 1]
        mi_independent = _compute_mutual_information(X_independent)[0, 1]
        assert mi_dependent > mi_independent

    def test_mutual_information_is_symmetric_and_zero_diagonal(self, rng):
        X = (rng.random((4, 100)) < 0.5).astype(int)
        mi = _compute_mutual_information(X)
        np.testing.assert_allclose(mi, mi.T)
        np.testing.assert_allclose(np.diag(mi), 0.0)

    def test_random_matrix_is_symmetric_and_bounded(self, rng):
        X = rng.random((5, 20))
        sim = _compute_random(X)
        np.testing.assert_allclose(sim, sim.T)
        assert np.all(sim >= 0.0) and np.all(sim < 1.0)


# ---------------------------------------------------------------------------
# CGVariables validation
# ---------------------------------------------------------------------------


class TestCGVariablesValidation:
    def test_rejects_non_binary_input(self, rng):
        continuous = rng.random((8, 100))
        with pytest.raises(ValueError, match="binarized"):
            CGVariables(continuous, rg_steps=1)

    def test_rejects_too_few_variables_for_requested_steps(self, random_binary_array):
        with pytest.raises(ValueError, match="less than required size"):
            CGVariables(random_binary_array, rg_steps=10)  # needs >= 1024 vars

    def test_rejects_unknown_cluster_method(self, random_binary_array):
        with pytest.raises(ValueError, match="Unknown cluster method"):
            CGVariables(
                random_binary_array, cluster_method="not_a_real_method", rg_steps=1
            )

    def test_warns_when_more_variables_than_samples(self, rng):
        wide = (rng.random((20, 10)) < 0.3).astype(int)
        # guard against zero-variance rows tripping a different warning first
        wide[:, 0] = 1 - wide[:, 1]
        with pytest.warns(UserWarning, match="unreliable"):
            CGVariables(wide, rg_steps=1)


# ---------------------------------------------------------------------------
# CGVariables zero-variance filtering (regression: commit ba4e8bf)
# ---------------------------------------------------------------------------


class TestZeroVarianceFiltering:
    def test_constant_rows_are_dropped_and_warned(
        self, binary_array_with_constant_rows
    ):
        with pytest.warns(UserWarning, match="constant"):
            cg = CGVariables(binary_array_with_constant_rows, rg_steps=1)
        # 10 rows, 3 constant -> 7 survive
        assert cg.CG_timeseries[0].shape[0] == 7

    def test_dropped_indices_never_appear_in_lineage(
        self, binary_array_with_constant_rows
    ):
        with pytest.warns(UserWarning):
            cg = CGVariables(binary_array_with_constant_rows, rg_steps=2)
        dropped = {2, 5, 7}
        for step_idx in cg.CG_cluster_idx:
            all_lineage = set(
                np.concatenate([np.atleast_1d(x) for x in step_idx]).tolist()
            )
            assert dropped.isdisjoint(all_lineage)


# ---------------------------------------------------------------------------
# CGVariables NaN row filtering (e.g. z-score binarizing a zero-variance row)
# ---------------------------------------------------------------------------


class TestNaNRowFiltering:
    def test_nan_rows_are_dropped_and_warned_with_source_explanation(
        self, binary_array_with_nan_row
    ):
        with pytest.warns(UserWarning, match="NaN"):
            cg = CGVariables(binary_array_with_nan_row, rg_steps=1)
        # 10 rows, 1 all-NaN -> 9 survive
        assert cg.CG_timeseries[0].shape[0] == 9

    def test_nan_row_index_named_in_warning(self, binary_array_with_nan_row):
        with pytest.warns(UserWarning, match=r"Row\(s\) \[4\]"):
            CGVariables(binary_array_with_nan_row, rg_steps=1)

    def test_dropped_nan_index_never_appears_in_lineage(
        self, binary_array_with_nan_row
    ):
        with pytest.warns(UserWarning):
            cg = CGVariables(binary_array_with_nan_row, rg_steps=2)
        for step_idx in cg.CG_cluster_idx:
            all_lineage = set(
                np.concatenate([np.atleast_1d(x) for x in step_idx]).tolist()
            )
            assert 4 not in all_lineage

    def test_remaining_data_is_still_processed_normally(
        self, binary_array_with_nan_row
    ):
        with pytest.warns(UserWarning):
            cg = CGVariables(
                binary_array_with_nan_row, cluster_method="random", rg_steps=1
            )
        assert not np.any(np.isnan(cg.CG_timeseries[0]))
        assert not np.any(np.isnan(cg.CG_timeseries[1]))

    def test_genuinely_non_binary_non_nan_data_still_raises(self, rng):
        continuous = rng.normal(size=(8, 100))  # no NaN, but not binarized
        with pytest.raises(ValueError, match="binarized"):
            CGVariables(continuous, rg_steps=1)

    def test_no_warning_when_no_constant_rows(self, random_binary_array, recwarn):
        CGVariables(random_binary_array, rg_steps=1)
        messages = [str(w.message) for w in recwarn.list]
        assert not any("constant" in m for m in messages)


# ---------------------------------------------------------------------------
# CGVariables odd-N handling (regression: leftover unpaired variable)
# ---------------------------------------------------------------------------


class TestOddNumberOfVariables:
    def test_prints_and_drops_leftover_variable(self, odd_binary_array, capsys):
        # The odd-N notice is a print_if_full (verbose='full' only), not a
        # real warnings.warn -- so it must be checked via captured stdout,
        # with verbose explicitly set to 'full' to make it fire at all.
        cg = CGVariables(odd_binary_array, rg_steps=1, verbose="full")
        out = capsys.readouterr().out
        assert "Odd number of variables" in out
        assert cg.CG_timeseries[1].shape[0] == 3  # 7 -> 3 pairs, 1 dropped

    def test_dropped_leftover_not_in_lineage(self, odd_binary_array):
        cg = CGVariables(odd_binary_array, rg_steps=1, verbose="full")
        lineage_step1 = set(
            np.concatenate([np.atleast_1d(x) for x in cg.CG_cluster_idx[1]]).tolist()
        )
        assert len(lineage_step1) == 6  # one of the 7 original indices is missing


# ---------------------------------------------------------------------------
# CGVariables structural invariants
# ---------------------------------------------------------------------------


class TestCGVariablesInvariants:
    def test_shape_halves_at_each_step(self, cgvars, random_binary_array):
        N0 = random_binary_array.shape[0]
        for k, ts in enumerate(cgvars.CG_timeseries):
            assert ts.shape == (N0 // (2**k), random_binary_array.shape[1])

    def test_activity_is_conserved_across_scales(self, cgvars):
        # Coarse-graining only sums paired rows, so with even N and no drops,
        # total activity (sum over all variables and time) must be identical
        # at every RG scale.
        total_step0 = cgvars.CG_timeseries[0].sum()
        for ts in cgvars.CG_timeseries[1:]:
            assert ts.sum() == pytest.approx(total_step0)

    def test_lineage_partitions_surviving_variables_at_each_step(
        self, cgvars, random_binary_array
    ):
        surviving = set(cgvars.CG_cluster_idx[0].tolist())
        for step_idx in cgvars.CG_cluster_idx:
            flat = np.concatenate([np.atleast_1d(x) for x in step_idx]).tolist()
            # no repeats within a scale
            assert len(flat) == len(set(flat))
            # every lineage index traces back to a surviving original variable
            assert set(flat).issubset(surviving)

    def test_duplicated_pairs_are_matched_by_pearson(
        self, duplicated_pairs_binary_array
    ):
        # 4 base rows each duplicated -> pearson correlation should pair each
        # row with its exact duplicate at the first coarse-graining step.
        cg = CGVariables(
            duplicated_pairs_binary_array, cluster_method="pearson", rg_steps=1
        )
        step1 = cg.CG_timeseries[1]
        step0 = cg.CG_timeseries[0]
        for j, lineage in enumerate(cg.CG_cluster_idx[1]):
            a, b = lineage
            np.testing.assert_array_equal(step1[j], step0[a] + step0[b])
            # duplicated rows are identical, so the merged pair is just 2x the original
            np.testing.assert_array_equal(step1[j], 2 * step0[a])

    def test_new_variables_equal_sum_of_paired_originals(self, cgvars):
        step0 = cgvars.CG_timeseries[0]
        step1 = cgvars.CG_timeseries[1]
        for j, lineage in enumerate(cgvars.CG_cluster_idx[1]):
            expected = step0[lineage].sum(axis=0)
            np.testing.assert_array_equal(step1[j], expected)


# ---------------------------------------------------------------------------
# Momentum-space coarse graining (MSVariables)
# ---------------------------------------------------------------------------


def _legacy_momentum_space_variables(data, rg_steps):
    """
    Reference implementation transcribed from the original (pre-toolbox)
    momentum-space script: builds the projection matrix P_k entry by entry
    and projects each variable row by row.
    """
    N = data.shape[0]
    x = data - np.mean(data, 1, keepdims=True)
    evals, evectors = np.linalg.eigh(np.cov(x))
    evectors = evectors[:, np.flip(np.argsort(evals))]
    result = []
    for k in range(rg_steps + 1):
        cutoff = int(N / (2**k))
        P_k = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                P_k[i, j] = np.dot(evectors[i, :cutoff], evectors[j, :cutoff])
        phi = np.zeros_like(x)
        for i in range(N):
            phi[i] = np.sum(np.einsum("ij,i->ij", x, P_k[i, :]), axis=0)
            phi[i] = phi[i] / np.std(phi[i])
        result.append(phi)
    return result


class TestMSVariables:
    def test_matches_legacy_implementation(self, random_binary_array):
        rg_steps = 4
        msvars = MSVariables(random_binary_array, rg_steps=rg_steps)
        legacy = _legacy_momentum_space_variables(
            random_binary_array.astype(float), rg_steps
        )
        for k in range(rg_steps + 1):
            np.testing.assert_allclose(
                msvars.get_MS_variables(k), legacy[k], atol=1e-10
            )

    def test_cutoffs_halve_at_each_step(self, random_binary_array):
        msvars = MSVariables(random_binary_array, rg_steps=5)
        np.testing.assert_array_equal(msvars.cutoffs, [32, 16, 8, 4, 2, 1])

    def test_step_zero_is_zscored_data(self, random_binary_array):
        msvars = MSVariables(random_binary_array, rg_steps=3)
        x = random_binary_array.astype(float)
        zscored = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
        np.testing.assert_allclose(msvars.get_MS_variables(0), zscored, atol=1e-10)

    def test_every_step_has_zero_mean_and_unit_variance(self, random_binary_array):
        msvars = MSVariables(random_binary_array, rg_steps=5)
        for k in range(msvars.rg_steps + 1):
            phi = msvars.get_MS_variables(k)
            assert phi.shape == random_binary_array.shape
            np.testing.assert_allclose(phi.mean(axis=1), 0.0, atol=1e-10)
            np.testing.assert_allclose(phi.std(axis=1), 1.0)

    def test_single_mode_variables_are_perfectly_correlated(self, random_binary_array):
        # With K = 1, every variable is a rescaled copy of the first principal
        # component, so all pairwise correlations are +-1.
        msvars = MSVariables(random_binary_array, rg_steps=5)
        corr = np.corrcoef(msvars.get_MS_variables(5))
        np.testing.assert_allclose(np.abs(corr), 1.0)

    def test_accepts_continuous_data(self, rng):
        continuous = rng.normal(size=(16, 300))
        msvars = MSVariables(continuous, rg_steps=3)
        assert np.all(np.isfinite(msvars.get_MS_variables(3)))

    def test_eigenvalues_sorted_descending(self, random_binary_array):
        msvars = MSVariables(random_binary_array, rg_steps=3)
        assert np.all(np.diff(msvars.eigenvalues) <= 0)

    def test_raises_when_too_few_variables(self, random_binary_array):
        with pytest.raises(ValueError, match="less than required size"):
            MSVariables(random_binary_array, rg_steps=6)

    def test_raises_on_out_of_range_step(self, random_binary_array):
        msvars = MSVariables(random_binary_array, rg_steps=3)
        with pytest.raises(ValueError, match="between 0 and 3"):
            msvars.get_MS_variables(4)

    def test_constant_and_nan_rows_are_dropped(self, random_binary_array):
        data = random_binary_array.astype(float)
        data[3] = 0.0
        data[7, 10] = np.nan
        with pytest.warns(UserWarning) as record:
            msvars = MSVariables(data, rg_steps=4)
        messages = " ".join(str(w.message) for w in record)
        assert "Row(s) [7]" in messages and "NaN" in messages
        assert "Row(s) [3]" in messages and "constant" in messages
        assert 3 not in msvars.original_idx and 7 not in msvars.original_idx
        assert msvars.get_MS_variables(0).shape == (30, data.shape[1])

    def test_rows_in_discarded_modes_become_nan(self, rng):
        # Two independent groups: variables 0-3 share one strong common
        # signal, variables 4-7 are pure, much weaker noise. Keeping only the
        # leading mode leaves the noise variables (almost) unrepresented;
        # the exactly-zero case is built by making the weak block orthogonal.
        common = rng.normal(size=500)
        strong = np.vstack([10 * common + rng.normal(size=500) for _ in range(4)])
        weak = 0.01 * rng.normal(size=(4, 500))
        msvars = MSVariables(np.vstack([strong, weak]), rg_steps=3)
        # Force an exactly-zero projection by zeroing the weak variables'
        # loading on the leading mode.
        msvars.eigenvectors[4:, 0] = 0.0
        with pytest.warns(UserWarning, match="zero variance after projection"):
            phi = msvars.get_MS_variables(3)
        assert np.all(np.isnan(phi[4:]))
        assert np.all(np.isfinite(phi[:4]))


class TestMSVariablesInputValidation:
    @pytest.mark.parametrize("rg_steps", [-1, 2.0, "2"])
    def test_rejects_invalid_rg_steps(self, random_binary_array, rg_steps):
        with pytest.raises(ValueError, match="rg_steps must be a non-negative integer"):
            MSVariables(random_binary_array, rg_steps=rg_steps)

    def test_accepts_numpy_integer_rg_steps(self, random_binary_array):
        msvars = MSVariables(random_binary_array, rg_steps=np.int64(2))
        assert msvars.cutoffs.tolist() == [32, 16, 8]

    def test_accepts_nested_list(self, random_binary_array):
        from_list = MSVariables(random_binary_array.tolist(), rg_steps=2)
        from_array = MSVariables(random_binary_array, rg_steps=2)
        np.testing.assert_allclose(
            from_list.get_MS_variables(1), from_array.get_MS_variables(1)
        )

    def test_rejects_one_dimensional_input(self, random_binary_array):
        with pytest.raises(ValueError, match="2D array"):
            MSVariables(random_binary_array[0], rg_steps=1)

    @pytest.mark.parametrize("k", [1.0, "1", 5, -1])
    def test_rejects_invalid_step(self, random_binary_array, k):
        msvars = MSVariables(random_binary_array, rg_steps=2)
        with pytest.raises(ValueError, match="Step k must be an integer"):
            msvars.get_MS_variables(k)


def _excess_kurtosis(values):
    values = np.ravel(values)
    centered = values - values.mean()
    return np.mean(centered**4) / np.mean(centered**2) ** 2 - 3


class TestMSVariablesKnownAnswers:
    """
    Checks against cases whose result is known in advance, independently of
    any reference implementation.
    """

    def test_gaussian_data_stays_gaussian(self, rng):
        # Every phi is a linear combination of jointly Gaussian variables,
        # hence exactly Gaussian at every step, whatever the correlations.
        mixing = rng.standard_normal((32, 32))
        data = mixing @ rng.standard_normal((32, 40000))
        msvars = MSVariables(data, rg_steps=5)
        for k in range(6):
            assert abs(_excess_kurtosis(msvars.get_MS_variables(k))) < 0.05

    @pytest.fixture(scope="class")
    def shared_probability_model(self):
        """
        Conditionally independent binary units of Nicoletti et al. (2020),
        Appendix B: at each time step every unit fires independently with
        the same probability xi, drawn uniformly from [0, 1].
        """
        rng = np.random.default_rng(0)
        N, T = 128, 40000
        xi = rng.random(T)
        data = (rng.random((N, T)) < xi).astype(float)
        return xi, MSVariables(data, rg_steps=7)

    def test_shared_probability_model_spectrum(self, shared_probability_model):
        # Covariance is 1/4 on the diagonal and 1/12 elsewhere, so there is
        # one eigenvalue 1/4 + (N - 1)/12 and N - 1 eigenvalues equal to 1/6.
        _, msvars = shared_probability_model
        N = msvars.cutoffs[0]
        assert msvars.eigenvalues[0] == pytest.approx(1 / 4 + (N - 1) / 12, rel=0.02)
        assert np.mean(msvars.eigenvalues[1:]) == pytest.approx(1 / 6, rel=0.01)

    def test_shared_probability_model_kurtosis_at_every_step(
        self, shared_probability_model
    ):
        # With r = K/N, phi_i = (xi - 1/2) + r * (own noise) + (Gaussian mix
        # of the other units' noise), which gives, for many units,
        #   <phi^2> = 1/12 + r/6
        #   <phi^4> = 1/80 + r/20 + r^2/10 - r^3/15 - r^4/30.
        # It runs from -2 (r = 1, z-scored fair coin) to -6/5 (r -> 0,
        # uniform distribution) and never approaches the Gaussian value 0.
        _, msvars = shared_probability_model
        r = msvars.cutoffs / msvars.cutoffs[0]
        fourth_moment = 1 / 80 + r / 20 + r**2 / 10 - r**3 / 15 - r**4 / 30
        expected = fourth_moment / (1 / 12 + r / 6) ** 2 - 3
        measured = [
            _excess_kurtosis(msvars.get_MS_variables(k))
            for k in range(msvars.rg_steps + 1)
        ]
        assert measured[0] == pytest.approx(-2, abs=1e-3)
        np.testing.assert_allclose(measured, expected, atol=0.06)

    def test_shared_probability_model_last_mode_recovers_latent_variable(
        self, shared_probability_model
    ):
        # With one mode kept, every unit becomes the z-scored population
        # rate, i.e. the hidden probability xi, which is uniform: bounded by
        # +-sqrt(3) once it has unit variance.
        xi, msvars = shared_probability_model
        phi = msvars.get_MS_variables(msvars.rg_steps)
        correlation = np.corrcoef(phi, xi)[-1, :-1]
        assert np.all(np.abs(correlation) > 0.98)
        assert np.max(np.abs(phi)) == pytest.approx(np.sqrt(3), rel=0.03)
