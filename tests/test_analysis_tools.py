"""Tests for prg_toolbox.analysis_tools: preprocessing, binarization, aggregation."""

import numpy as np
import pytest

from prg_toolbox import observables as obs
from prg_toolbox.analysis_tools import (
    binary_array_from_stamps,
    binary_array_from_zscore,
    binary_array_from_zscore_maxima,
    binarize_data,
    discard_transient,
    slice_by_time_window,
    pick_random_sample,
    shuffle_isi,
    is_function_observable,
    average_observable_sample_values,
    average_across_windows_for_functions,
)


# ---------------------------------------------------------------------------
# Binarization
# ---------------------------------------------------------------------------

class TestBinaryArrayFromStamps:
    def test_known_spikes_produce_expected_matrix(self):
        # unit 0 spikes at t=0,15ms; unit 1 spikes at t=5ms. binsize=10ms.
        # nbins = 1 + floor(max_time / binsize) = 1 + int(15/10) = 2.
        stamps = np.array([
            [0.0, 0],
            [5.0, 1],
            [15.0, 0],
        ])
        binary = binary_array_from_stamps(stamps, binsize_ms=10.0)
        expected = np.array([
            [1, 1],  # unit 0: bin0 (t=0), bin1 (t=15 -> bin index 1)
            [1, 0],  # unit 1: bin0 (t=5 -> bin index 0)
        ])
        np.testing.assert_array_equal(binary, expected)

    def test_remaps_non_contiguous_unit_ids(self):
        stamps = np.array([
            [0.0, 7],
            [1.0, 3],
        ])
        binary = binary_array_from_stamps(stamps, binsize_ms=1.0)
        assert binary.shape[0] == 2  # only 2 unique units, remapped to 0/1


class TestBinaryArrayFromZscore:
    def test_threshold_crossing_is_detected(self):
        # row with an obvious single outlier
        x = np.array([[0.0, 0.0, 0.0, 0.0, 10.0]])
        binary = binary_array_from_zscore(x, threshold=1.0)
        assert binary[0, -1] == 1
        assert binary[0, :-1].sum() == 0

    def test_low_threshold_flags_more_bins(self):
        x = np.array([[0.0, 1.0, 2.0, 3.0, 10.0]])
        strict = binary_array_from_zscore(x, threshold=2.0)
        lenient = binary_array_from_zscore(x, threshold=0.0)
        assert lenient.sum() >= strict.sum()


class TestBinaryArrayFromZscoreMaxima:
    def test_detects_local_peak_above_threshold(self):
        x = np.array([[0.0, 1.0, 5.0, 1.0, 0.0, 1.0, 0.2, 1.0, 0.0]])
        binary = binary_array_from_zscore_maxima(x, threshold=1.0)
        assert binary[0, 2] == 1  # the clear peak at index 2
        assert binary.sum() >= 1


class TestBinarizeDataDispatch:
    def test_timeseries_with_none_method_passes_through(self):
        data = np.array([[1, 0, 1], [0, 1, 0]])
        result = binarize_data(data, data_format='timeseries', binary_method=None)
        np.testing.assert_array_equal(result, data)

    def test_timeseries_with_unknown_method_raises(self):
        data = np.zeros((2, 5))
        with pytest.raises(ValueError, match="Unknown binary_method"):
            binarize_data(data, data_format='timeseries', binary_method='not_a_method')

    def test_timeseries_zscore_threshold_dispatches_correctly(self):
        data = np.array([[0.0, 0.0, 0.0, 0.0, 10.0]])
        result = binarize_data(data, data_format='timeseries', binary_method='zscore_threshold', zscore_threshold=1.0)
        np.testing.assert_array_equal(result, binary_array_from_zscore(data, threshold=1.0))

    def test_non_timeseries_format_dispatches_to_stamps(self):
        stamps = np.array([[0.0, 0], [5.0, 1]])
        result = binarize_data(stamps, data_format='tabular', binsize_ms=10.0)
        np.testing.assert_array_equal(result, binary_array_from_stamps(stamps, binsize_ms=10.0))


# ---------------------------------------------------------------------------
# Transient discarding
# ---------------------------------------------------------------------------

class TestDiscardTransient:
    def test_timeseries_drops_leading_bins(self):
        data = np.arange(20).reshape(2, 10)
        result = discard_transient(data, data_format='timeseries', timeseries_binsize_ms=1, transient_time_ms=3)
        assert result.shape == (2, 7)
        np.testing.assert_array_equal(result, data[:, 3:])

    def test_timestamps_shifts_origin_to_zero(self):
        stamps = np.array([[5.0, 0], [10.0, 1], [20.0, 0]])
        result = discard_transient(stamps, data_format='tabular', transient_time_ms=10.0)
        # spike at t=5 is before the transient cutoff and dropped
        assert result.shape[0] == 2
        np.testing.assert_allclose(result[:, 0], [0.0, 10.0])


# ---------------------------------------------------------------------------
# Time-window slicing
# ---------------------------------------------------------------------------

class TestSliceByTimeWindow:
    def test_timeseries_window_count_no_overlap(self):
        data = np.zeros((3, 100))
        slices = slice_by_time_window(data, window_duration_ms=25, data_format='timeseries', timeseries_binsize_ms=1.0)
        assert len(slices) == 4
        for s in slices:
            assert s.shape == (3, 25)

    def test_timeseries_window_count_with_overlap(self):
        data = np.zeros((3, 100))
        slices = slice_by_time_window(data, window_duration_ms=20, data_format='timeseries',
                                       timeseries_binsize_ms=1.0, overlap_fraction=0.5)
        # step=10 bins, window=20 bins -> floor((100-20)/10)+1 = 9
        assert len(slices) == 9

    def test_timeseries_short_data_warns_and_returns_full_data_as_one_window(self):
        data = np.zeros((3, 10))
        with pytest.warns(UserWarning, match="longer than"):
            slices = slice_by_time_window(data, window_duration_ms=1000, data_format='timeseries', timeseries_binsize_ms=1.0)
        assert len(slices) == 1
        np.testing.assert_array_equal(slices[0], data)

    def test_timestamps_short_data_warns_and_returns_full_data_as_one_window(self):
        stamps = np.array([[1.0, 0], [2.0, 1]])
        with pytest.warns(UserWarning, match="longer than"):
            slices = slice_by_time_window(stamps, window_duration_ms=1000, data_format='tabular')
        assert len(slices) == 1


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

class TestPickRandomSample:
    def test_timeseries_same_seed_is_deterministic(self):
        data = np.arange(40).reshape(10, 4)
        a = pick_random_sample(data, sample_size=4, data_format='timeseries', random_seed=42)
        b = pick_random_sample(data, sample_size=4, data_format='timeseries', random_seed=42)
        np.testing.assert_array_equal(a, b)

    def test_timeseries_different_seeds_can_differ(self):
        data = np.arange(200).reshape(50, 4)
        a = pick_random_sample(data, sample_size=5, data_format='timeseries', random_seed=1)
        b = pick_random_sample(data, sample_size=5, data_format='timeseries', random_seed=2)
        assert not np.array_equal(a, b)

    def test_timeseries_oversized_sample_raises(self):
        data = np.zeros((5, 10))
        with pytest.raises(ValueError, match="larger than"):
            pick_random_sample(data, sample_size=10, data_format='timeseries')

    def test_timeseries_returns_exact_sample_size(self):
        data = np.arange(80).reshape(20, 4)
        result = pick_random_sample(data, sample_size=6, data_format='timeseries', random_seed=7)
        assert result.shape == (6, 4)

    def test_invalid_data_format_raises(self):
        data = np.zeros((5, 10))
        with pytest.raises(ValueError, match="data_format"):
            pick_random_sample(data, sample_size=2, data_format='not_a_format')


# ---------------------------------------------------------------------------
# ISI shuffling
# ---------------------------------------------------------------------------

class TestShuffleISI:
    def test_binary_preserves_total_spike_count_per_unit(self, rng):
        data = (rng.random((5, 200)) < 0.2).astype(int)
        shuffled = shuffle_isi(data, data_format='timeseries', random_seed=1)
        np.testing.assert_array_equal(data.sum(axis=1), shuffled.sum(axis=1))

    def test_binary_single_spike_unit_passes_through_unchanged(self):
        data = np.zeros((1, 50), dtype=int)
        data[0, 10] = 1
        shuffled = shuffle_isi(data, data_format='timeseries', random_seed=1)
        np.testing.assert_array_equal(data, shuffled)

    def test_binary_zero_spike_unit_stays_silent(self):
        data = np.zeros((1, 50), dtype=int)
        shuffled = shuffle_isi(data, data_format='timeseries', random_seed=1)
        assert shuffled.sum() == 0

    def test_timestamps_preserves_spike_count_per_unit(self, rng):
        times = np.sort(rng.uniform(0, 1000, size=60))
        units = rng.integers(0, 3, size=60)
        stamps = np.column_stack([times, units])
        shuffled = shuffle_isi(stamps, data_format='tabular', random_seed=1)
        orig_counts = {u: np.sum(stamps[:, 1] == u) for u in np.unique(stamps[:, 1])}
        shuf_counts = {u: np.sum(shuffled[:, 1] == u) for u in np.unique(shuffled[:, 1])}
        assert orig_counts == shuf_counts

    def test_timestamps_output_is_chronologically_sorted(self, rng):
        times = np.sort(rng.uniform(0, 1000, size=60))
        units = rng.integers(0, 3, size=60)
        stamps = np.column_stack([times, units])
        shuffled = shuffle_isi(stamps, data_format='tabular', random_seed=1)
        assert np.all(np.diff(shuffled[:, 0]) >= 0)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class TestIsFunctionObservable:
    @pytest.mark.parametrize("observable_cls", [obs.covariance_spectrum, obs.autocorrelation_function, obs.activity_distribution])
    def test_function_valued_observables(self, observable_cls):
        assert is_function_observable(observable_cls) is True

    @pytest.mark.parametrize("observable_cls", [obs.mean_variance, obs.log_silence_probability, obs.max_covariance_eigenvalue, obs.decay_time])
    def test_scalar_observables(self, observable_cls):
        assert is_function_observable(observable_cls) is False


class TestAverageObservableSampleValues:
    def test_max_covariance_eigenvalue_regression_no_crash(self, cgvars):
        # Regression test: this path used to pass an invalid max_ev=True
        # kwarg to get_scaling_exponent and crash with a TypeError whenever
        # nsamples > 1 was used with an eigenvalue-based observable.
        instance = obs.max_covariance_eigenvalue(cgvars)
        stacked = np.stack([instance.values, instance.values * 1.01], axis=0)
        result = average_observable_sample_values(instance, stacked)
        assert np.isfinite(result.exponent)

    def test_scalar_observable_averages_correctly(self, cgvars):
        instance = obs.mean_variance(cgvars)
        stacked = np.stack([instance.values, instance.values + 1.0], axis=0)
        result = average_observable_sample_values(instance, stacked)
        np.testing.assert_allclose(result.avg_across_windows, instance.values + 0.5)


class TestAverageAcrossWindowsForFunctions:
    def test_averages_matching_length_trials(self):
        rg_steps = 2
        trial_a = [np.array([1.0, 2.0]), np.array([10.0, 20.0])]
        trial_b = [np.array([3.0, 4.0]), np.array([30.0, 40.0])]
        avg, std = average_across_windows_for_functions([trial_a, trial_b], rg_steps)
        np.testing.assert_allclose(avg[0], [2.0, 3.0])
        np.testing.assert_allclose(avg[1], [20.0, 30.0])

    def test_truncates_to_shortest_trial_length(self):
        rg_steps = 1
        trial_a = [np.array([1.0, 2.0, 3.0])]
        trial_b = [np.array([5.0, 6.0])]  # shorter
        avg, std = average_across_windows_for_functions([trial_a, trial_b], rg_steps)
        assert len(avg[0]) == 2
        np.testing.assert_allclose(avg[0], [3.0, 4.0])
