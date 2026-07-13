"""Tests for prg_toolbox.coarse_graining: similarity metrics and CGVariables."""

import numpy as np
import pytest

from prg_toolbox.coarse_graining import (
    CGVariables,
    _compute_pearson,
    _compute_spearman,
    _compute_cosine,
    _compute_hamming,
    _compute_mutual_information,
    _compute_random,
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
        y = x ** 3  # any strictly increasing transform preserves rank
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
            CGVariables(random_binary_array, cluster_method="not_a_real_method", rg_steps=1)

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
    def test_constant_rows_are_dropped_and_warned(self, binary_array_with_constant_rows):
        with pytest.warns(UserWarning, match="constant"):
            cg = CGVariables(binary_array_with_constant_rows, rg_steps=1)
        # 10 rows, 3 constant -> 7 survive
        assert cg.CG_timeseries[0].shape[0] == 7

    def test_dropped_indices_never_appear_in_lineage(self, binary_array_with_constant_rows):
        with pytest.warns(UserWarning):
            cg = CGVariables(binary_array_with_constant_rows, rg_steps=2)
        dropped = {2, 5, 7}
        for step_idx in cg.CG_cluster_idx:
            all_lineage = set(np.concatenate([np.atleast_1d(x) for x in step_idx]).tolist())
            assert dropped.isdisjoint(all_lineage)


# ---------------------------------------------------------------------------
# CGVariables NaN row filtering (e.g. z-score binarizing a zero-variance row)
# ---------------------------------------------------------------------------

class TestNaNRowFiltering:
    def test_nan_rows_are_dropped_and_warned_with_source_explanation(self, binary_array_with_nan_row):
        with pytest.warns(UserWarning, match="NaN"):
            cg = CGVariables(binary_array_with_nan_row, rg_steps=1)
        # 10 rows, 1 all-NaN -> 9 survive
        assert cg.CG_timeseries[0].shape[0] == 9

    def test_nan_row_index_named_in_warning(self, binary_array_with_nan_row):
        with pytest.warns(UserWarning, match=r"Row\(s\) \[4\]"):
            CGVariables(binary_array_with_nan_row, rg_steps=1)

    def test_dropped_nan_index_never_appears_in_lineage(self, binary_array_with_nan_row):
        with pytest.warns(UserWarning):
            cg = CGVariables(binary_array_with_nan_row, rg_steps=2)
        for step_idx in cg.CG_cluster_idx:
            all_lineage = set(np.concatenate([np.atleast_1d(x) for x in step_idx]).tolist())
            assert 4 not in all_lineage

    def test_remaining_data_is_still_processed_normally(self, binary_array_with_nan_row):
        with pytest.warns(UserWarning):
            cg = CGVariables(binary_array_with_nan_row, cluster_method="random", rg_steps=1)
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
    def test_warns_and_drops_leftover_variable(self, odd_binary_array):
        with pytest.warns(UserWarning, match="Odd number of variables"):
            cg = CGVariables(odd_binary_array, rg_steps=1)
        assert cg.CG_timeseries[1].shape[0] == 3  # 7 -> 3 pairs, 1 dropped

    def test_dropped_leftover_not_in_lineage(self, odd_binary_array):
        with pytest.warns(UserWarning):
            cg = CGVariables(odd_binary_array, rg_steps=1)
        lineage_step1 = set(np.concatenate([np.atleast_1d(x) for x in cg.CG_cluster_idx[1]]).tolist())
        assert len(lineage_step1) == 6  # one of the 7 original indices is missing


# ---------------------------------------------------------------------------
# CGVariables structural invariants
# ---------------------------------------------------------------------------

class TestCGVariablesInvariants:
    def test_shape_halves_at_each_step(self, cgvars, random_binary_array):
        N0 = random_binary_array.shape[0]
        for k, ts in enumerate(cgvars.CG_timeseries):
            assert ts.shape == (N0 // (2 ** k), random_binary_array.shape[1])

    def test_activity_is_conserved_across_scales(self, cgvars):
        # Coarse-graining only sums paired rows, so with even N and no drops,
        # total activity (sum over all variables and time) must be identical
        # at every RG scale.
        total_step0 = cgvars.CG_timeseries[0].sum()
        for ts in cgvars.CG_timeseries[1:]:
            assert ts.sum() == pytest.approx(total_step0)

    def test_lineage_partitions_surviving_variables_at_each_step(self, cgvars, random_binary_array):
        surviving = set(cgvars.CG_cluster_idx[0].tolist())
        for step_idx in cgvars.CG_cluster_idx:
            flat = np.concatenate([np.atleast_1d(x) for x in step_idx]).tolist()
            # no repeats within a scale
            assert len(flat) == len(set(flat))
            # every lineage index traces back to a surviving original variable
            assert set(flat).issubset(surviving)

    def test_duplicated_pairs_are_matched_by_pearson(self, duplicated_pairs_binary_array):
        # 4 base rows each duplicated -> pearson correlation should pair each
        # row with its exact duplicate at the first coarse-graining step.
        cg = CGVariables(duplicated_pairs_binary_array, cluster_method="pearson", rg_steps=1)
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
