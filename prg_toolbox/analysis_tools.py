"""
Copyright (c) 2026 Daniel Miranda Castro. Licensed under the MIT License.

Data preprocessing, result aggregation, and file I/O utilities.

This module provides the independent supporting utilities for the PRG pipeline. 
It handles data ingestion across formats, temporal shuffling for null models, 
and functions to aggregate, save, and plot observables.
"""
import os
import hashlib
import json
from datetime import datetime
import pickle
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
import scipy.io as sio
import pandas as pd
import dataclasses
import matplotlib.pyplot as plt


from . import observables as obs
from .utils import get_scaling_exponent
from .config import *
from . import plotting as plot

def _load_timestamps(file_or_path, format="tabular", time_col=1, unit_col=0, sep=r"\s+", header=None, scale_factor=1.0):
    r"""
    Loads neuronal spike timestamps and unit IDs from various file formats 
    into a standardized N x 2 NumPy array.

    The output array is guaranteed to be sorted chronologically by timestamp.

    Parameters
    ----------
    file_or_path : str
        The absolute path to the data file or the file itself.
    format : {'tabular', 'numpy_2col', 'phy'}, optional
        The format of the input data. 
        - 'tabular': Text-based formats like .csv, .tsv, .txt, or .gdf.
        - 'numpy_2col': A single .npy file containing a 2D array.
        Default is 'tabular'.
    time_col : int, optional
        The column index containing the timestamp data (0-indexed). Only 
        used if format is 'tabular' or 'numpy_2col'. Default is 1.
    unit_col : int, optional
        The column index containing the neuron/unit ID data (0-indexed). Only 
        used if format is 'tabular' or 'numpy_2col'. Default is 0.
    sep : str, optional
        The delimiter string used to separate values in text files. Use r"\s+" 
        for space-separated files (like .gdf). Only used if format is 'tabular'. 
        Default is ','.
    header : int or None, optional
        Row number to use as the column names in text files. Use None if there 
        is no header row. Only used if format is 'tabular'. Default is None.
    scale_factor : float, optional
        A multiplier applied to the raw timestamps to convert them into your 
        desired unit (e.g., converting sample counts to seconds). For example, 
        if data is sampled at 30kHz, use `scale_factor=1/30000`. Default is 1.0.

    Returns
    ----------
    timestamps : numpy.ndarray
        An N x 2 array where column 0 contains spike times and column 1 contains 
        corresponding neuron/unit IDs. The array is sorted in ascending order by 
        spike time. Choosing ms as the time unit is recommended for consistency 
        with other functions.
        """
    
    if format == "tabular":
        # Handles CSV, TSV, TXT, GDF
        df = pd.read_csv(file_or_path, sep=sep, header=header) if isinstance(file_or_path, str) else file_or_path
        
        # Extract just the two columns we care about in the right order
        times = df.iloc[:, time_col].to_numpy() * scale_factor
        units = df.iloc[:, unit_col].to_numpy()
        
        timestamps = np.column_stack((times, units))
        
    elif format == "numpy_2col":
        # Handles standard N x 2 numpy arrays
        raw_array = np.load(file_or_path) if isinstance(file_or_path, str) else file_or_path
        times = raw_array[:, time_col] * scale_factor
        units = raw_array[:, unit_col]
        
        timestamps = np.column_stack((times, units))
        
    else:
        raise ValueError(f"Unsupported format for timestamps: {format}")

    # Ensure chronological order
    timestamps = timestamps[timestamps[:, 0].argsort()]
    
    return timestamps

def _load_timeseries(filepath, delimiter=None, mat_key=None):
    """
    Loads dense timeseries data from various file formats.

    Parameters
    ----------
    filepath : str or pathlib.Path
        The path to the data file. Supported formats: .npy, .txt, .csv, .dat, .mat.
    delimiter : str, optional
        The string used to separate values in text files (e.g., ',', '\t'). 
        If None, it defaults to ',' for .csv and whitespace for .txt/.dat.
    mat_key : str, optional
        The dictionary key for the data array if loading a .mat file. If None 
        and the .mat file contains multiple arrays, a ValueError is raised.

    Returns
    -------
    data : numpy.ndarray
        A 2D dense array representing the timeseries.
    """
    # 1. Extract the file extension to route the loading logic
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    
    # 2. Branch based on file type
    if ext == '.npy':
        data = np.load(filepath)
        
    elif ext in ['.txt', '.csv', '.dat']:
        if delimiter is None and ext == '.csv':
            delimiter = ','
        data = np.loadtxt(filepath, delimiter=delimiter)
        
    elif ext == '.mat':

        mat_dict = sio.loadmat(filepath)
        
        if mat_key:
            data = mat_dict[mat_key]
        else:
            # Filter out MATLAB's internal metadata keys (which start with '__')
            valid_keys = [k for k in mat_dict.keys() if not k.startswith('__')]
            if len(valid_keys) == 1:
                data = mat_dict[valid_keys[0]]
            else:
                raise ValueError(
                    f"Multiple arrays found in .mat file: {valid_keys}. "
                    "Please specify exactly which one to load using the 'mat_key' parameter."
                )
    else:
        raise ValueError(f"Unsupported file format: '{ext}'. Please use .npy, .mat, .txt, .csv, or .dat.")

    # 3. Validation: Ensure it's a 2D matrix
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, but loaded array has shape {data.shape}. " 
                         "Timeseries data must be strictly 2D.")
        
    return data

def load_data(filepath, user_params: AnalysisParams = None):
    """
    Universal data loader for the PRG Toolbox. Routes files to the correct 
    parsing engine based on the specified data format.

    Parameters
    ----------
    filepath : str
        The absolute or relative path to the data file.
    user_params : AnalysisParams 
        Active parameter dataclass which define data loading settings in user_params.loading

    Returns
    -------
    numpy.ndarray
        The loaded and formatted data array.
    """
    # Standalone User Fallback: Instantiate default params if none are provided
    if user_params is None:
        params = DataLoadingParams()
    else:
        params = user_params.loading

    # Explicit routing and unpacking
    if params.data_format == 'timeseries':
        return _load_timeseries(
            filepath,
            delimiter=params.delimiter,
            mat_key=params.mat_key
        )
        
    elif params.data_format == 'tabular' or params.data_format == 'numpy_2col':
        return _load_timestamps(
            filepath,
            format=params.data_format,
            time_col=params.time_col,
            unit_col=params.unit_col,
            sep=params.sep,
            header=params.header,
            scale_factor=params.time_scale_factor
        )
        
    else:
        raise ValueError(
            f"Invalid data_format '{params.data_format}'. "
            "Must be either 'timeseries', 'tabular' or 'numpy_2col'."
        )

def _discard_transient_from_timestamps(timestamps, transient_time_ms=0):
    """Discards from original timestamps all events before the specified time.

    Parameters
    ----------
    timestamps (numpy array): An (total_spikes, 2) array, where 
                                timestamps[:, 0] is spike time in ms, and 
                                timestamps[:, 1] is the neuron ID.
    transient_time_ms (float/int): The duration of the transient period to discard.

    Returns
    ----------
    filtered_timestamps (numpy array): An array with 2 columns as the input, 
                                    but only containing spikes that occur after 
                                    the transient period, with times shifted so 
                                    that the first spike after the transient.
    """
    # 1. Use binary search to find the index where spikes cross the threshold
    # Since column 0 is already sorted, this is an O(log N) operation
    idx = np.searchsorted(timestamps[:, 0], transient_time_ms)
    
    # 2. Slice the matrix from that index onward (O(1) memory view allocation)
    filtered_timestamps = timestamps[idx:].copy()
    
    # 3. Shift remaining times to reset origin to t = 0
    filtered_timestamps[:, 0] -= transient_time_ms
    
    return filtered_timestamps

def _discard_transient_from_timeseries(data, timeseries_binsize_ms, transient_time_ms):
    """
    Discards the initial columns corresponding to the transient period from a dense matrix.

    Parameters
    ----------
    data : numpy.ndarray
        A dense 2D array of shape (N_neurons, T_bins) representing continuous 
        or binarized timeseries data.
    timeseries_binsize_ms : float or int
        The temporal resolution of a single bin in milliseconds (dt).
    transient_time_ms : float or int
        The total duration of the transient period to discard in milliseconds.

    Returns
    -------
    numpy.ndarray
        A view of the original array with the first `transient` columns removed.
    """
    transient = int(transient_time_ms/timeseries_binsize_ms)
    return data[:,transient:]

def discard_transient(data,data_format = 'timeseries', timeseries_binsize_ms = 1, transient_time_ms = 0):
    """
    Removes the initial transient period from data.
    
    This function acts as a wrapper that automatically applies the appropriate
    discarding method based on the data format.

    Parameters
    ----------
    data : numpy.ndarray
        The input neural data. Can be either a dense matrix (timeseries) or 
        a sparse 2-column array (timestamps).
    data_format : {'timeseries', 'tabular', 'numpy_2col'}, optional
        A string flag indicating the format of the input `data`. 
        Default is 'timeseries'.
    timeseries_binsize_ms : float or int, optional
        The size of a single time bin in ms. Only used if data_format is 
        'timeseries'. Default is 1.0.
    transient_time_ms : float or int, optional
        The duration of the initial recording period to discard in ms. 
        Default is 0.0.

    Returns
    -------
    filtered_data : numpy.ndarray
        The truncated data in the exact same format as the input.
    """
    if data_format == 'timeseries':
        filtered_data = _discard_transient_from_timeseries(data, timeseries_binsize_ms, transient_time_ms)
    else:
        filtered_data = _discard_transient_from_timestamps(data, transient_time_ms)
    return filtered_data

def _pick_random_sample_from_timestamps(timestamps, sample_size, random_seed = 123):
    """
    Subsample rows based on randomly selected unit indices.

    Parameters
    ----------
    timestamps : ndarray of floats     
        2D array with shape (n_spikes, 2)
    sample_size : integer                      
        number of unique units to sample
    random_seed : integer                      
        seed for reproducibility

    Returns
    ----------
    stamps_sample : ndarray of floats  
        filtered data
    """
    rng = np.random.default_rng(random_seed)
    original_idx = np.unique(timestamps[:, 1])
    selected_units = rng.choice(original_idx, size=sample_size, replace=False)

    mask = np.isin(timestamps[:, 1], selected_units)
    stamps_sample = timestamps[mask]
    return stamps_sample

def _pick_random_sample_from_timeseries(data, sample_size, random_seed=123):
    """
    Subsamples a dense timeseries matrix by randomly selecting rows.

    Parameters
    ----------
    timestamps : ndarray of floats     
        2D array with shape (n_spikes, 2)
    sample_size : integer                      
        number of unique units to sample
    random_seed : integer                      
        seed for reproducibility

    Returns
    ----------
    data[selected_rows, :] : ndarray of floats  
        filtered data
    """
    rng = np.random.default_rng(random_seed)
    n_neurons = data.shape[0]
    
    if sample_size > n_neurons:
        raise ValueError(
            f"Requested sample_size ({sample_size}) is larger than the number "
            f"of rows/units in the timeseries matrix ({n_neurons})."
        )
        
    # Choose random row indices and slice the matrix
    selected_rows = rng.choice(n_neurons, size=sample_size, replace=False)
    
    return data[selected_rows, :]


def pick_random_sample(data, sample_size, data_format='timeseries', random_seed=123):
    """
    Randomly subsamples a set number of units (neurons) from the data.

    Parameters
    ----------
    data : numpy.ndarray
        The input neural data. Can be either a dense matrix (timeseries) or 
        a sparse 2-column array (timestamps).
    sample_size : int
        The exact number of unique units to randomly sample.
    data_format : {'timeseries', 'tabular' or 'numpy_2col'}, optional
        A string flag indicating the format of the input `data`. 
        Default is 'timeseries'.
    random_seed : int, optional
        Seed for the random number generator to ensure reproducibility. 
        Default is 123.

    Returns
    -------
    numpy.ndarray
        The subsampled data in the exact same format as the input.
    """
    if data_format == 'timeseries':
        return _pick_random_sample_from_timeseries(data, sample_size, random_seed)
    elif data_format == 'tabular' or data_format == 'numpy_2col':
        return _pick_random_sample_from_timestamps(data, sample_size, random_seed)
    else:
        raise ValueError("data_format must be either 'timeseries', 'tabular' or 'numpy_2col'")
    

def _slice_timestamps(timestamps, window_duration_ms, overlap_fraction=0.0, t_start=0):
    """
    Slice an ordered timestamp array into sequential windows of a fixed duration.

    Splits chronological event sequences into overlapping or contiguous temporal blocks 
    of identical lengths. Any trailing data that cannot form a complete window of the 
    specified duration is automatically discarded to prevent statistical bias.

    Parameters
    ----------
    timestamps : ndarray of shape (n_spikes, 2)
        Spike timing matrix where column 0 contains event times and column 1 contains unit IDs.
        The array must be sorted chronologically by time.
    window_duration_ms : float
        Temporal duration desired for each generated slice window in milliseconds.
    overlap_fraction : float, optional
        The fractional overlap ratio between consecutive window boundaries, bounded between 
        [0.0, 1.0). Default is 0.0 (no overlap).
    t_start : float, optional
        The absolute initial time coordinate to align window generation offsets against. 
        Default is 0.0.

    Returns
    -------
    slices: list of ndarray
        A list of partitioned timestamp arrays. Each valid sub-array is cast to `float64`, 
        with its temporal baseline shifted so that the individual window's onset begins at 0.0. 
        Returns an empty list if the incoming data duration is shorter than a single window.
    """
    if len(timestamps) == 0:
        return timestamps
        
    t_end = timestamps[-1, 0]
    total_duration = t_end - t_start
    
    # If the total duration is shorter than a single window, no complete slice can be made
    if total_duration < window_duration_ms:
        return timestamps

    # Calculate step size based on overlap
    step_size = window_duration_ms * (1.0 - overlap_fraction)
    
    # Determine the number of complete windows that can fit
    # total_duration >= window_duration + (nslices - 1) * step_size
    if step_size > 0:
        nslices = int((total_duration - window_duration_ms) // step_size) + 1
    else:
        # If overlap is somehow configured to completely match step size (not recommended)
        nslices = 1

    slices = []
    for i in range(nslices):
        w_start = t_start + i * step_size
        w_end = w_start + window_duration_ms
        
        # Binary search for window boundaries
        idx_start = np.searchsorted(timestamps[:, 0], w_start)
        idx_end = np.searchsorted(timestamps[:, 0], w_end)
        
        window_slice = timestamps[idx_start:idx_end].copy()
        
        if len(window_slice) > 0:
            # Cast explicitly to ensure clean floating-point subtraction
            window_slice = window_slice.astype(np.float64)
            window_slice[:, 0] -= w_start
            
        slices.append(window_slice)
        
    return slices

def _slice_timeseries(data, window_duration_ms, timeseries_binsize_ms=1.0, overlap_fraction=0.0):
    """
    Slice an ordered matrix array into sequential windows of a fixed duration.

    Splits chronological event sequences into overlapping or contiguous temporal blocks 
    of identical lengths. Any trailing data that cannot form a complete window of the 
    specified duration is automatically discarded to prevent statistical bias.

    Parameters
    ----------
    data : ndarray of shape (n_units, n_timebins)
        Spike timing matrix with units in rows and time bins in columns (N,T).
    window_duration_ms : float
        Temporal duration desired for each generated slice window in milliseconds.
    timeseries_binsize_ms : float or int, optional
        The temporal resolution of a single bin in ms. Only used if data_format 
        is 'timeseries'. Default is 1.0.
    overlap_fraction : float, optional
        The fractional overlap ratio between consecutive window boundaries, bounded between 
        [0.0, 1.0). Default is 0.0 (no overlap).

    Returns
    -------
    slices : list of ndarray
        A list of partitioned matrix arrays. 
    """
    # Convert time specifications into column indices (bins)
    window_bins = int(window_duration_ms // timeseries_binsize_ms)
    total_bins = data.shape[1]
    
    if total_bins < window_bins:
        return []
        
    step_bins = int(window_bins * (1.0 - overlap_fraction))
    
    if step_bins > 0:
        nslices = int((total_bins - window_bins) // step_bins) + 1
    else:
        nslices = 1
        
    slices = []
    for i in range(nslices):
        start_bin = i * step_bins
        end_bin = start_bin + window_bins
        
        # Fast column slicing (O(1) memory views)
        slices.append(data[:, start_bin:end_bin])
        
    return slices

def slice_by_time_window(data, window_duration_ms, data_format='timeseries', 
                    timeseries_binsize_ms=1.0, overlap_fraction=0.0, t_start=0.0):
    """
    Slices chronological data into sequential windows of a fixed duration.
    
    Splits data into overlapping or contiguous temporal blocks. Any trailing data 
    that cannot form a complete window is automatically discarded to prevent 
    statistical bias in the downstream PRG analysis.

    Parameters
    ----------
    data : numpy.ndarray
        The input data. Can be either a dense matrix (timeseries) or 
        a 2-column array (timestamps).
    window_duration_ms : float
        Temporal duration desired for each generated slice window in milliseconds.
    data_format : {'timeseries', 'tabular', 'numpy_2col'}, optional
        A string flag indicating the format of the input `data`. 
        Default is 'timeseries'.
    timeseries_binsize_ms : float or int, optional
        The temporal resolution of a single bin in ms. Only used if data_format 
        is 'timeseries'. Default is 1.0.
    overlap_fraction : float, optional
        The fractional overlap ratio between consecutive window boundaries, bounded 
        between [0.0, 1.0). Default is 0.0 (no overlap).
    t_start : float, optional
        The absolute initial time coordinate. Only relevant if data_format is 
        'tabular' or 'numpy_2col'. Default is 0.0.

    Returns
    -------
    list of numpy.ndarray
        A list of partitioned data arrays in the same format as the input. 
        For timestamps, the temporal baseline is shifted so each window begins at t=0.
    """
    if data_format == 'timeseries':
        return _slice_timeseries(
            data, 
            window_duration_ms, 
            timeseries_binsize_ms, 
            overlap_fraction
        )
    elif data_format == 'tabular' or data_format == 'numpy_2col':
        return _slice_timestamps(
            data, 
            window_duration_ms, 
            overlap_fraction, 
            t_start
        )
    else:
        raise ValueError("data_format must be either 'timeseries', 'tabular' or 'numpy_2col'")
    

def _shuffle_isi_timestamps(timestamps, random_seed=None):
    """
    Creates a surrogate dataset by shuffling the Inter-Spike Intervals (ISIs) 
    independently for each individual unit.
    
    This preserves each unit's average firing rate and ISI distribution 
    (first-order statistics) while destroying temporal correlations, 
    bursting structures, and cross-unit synchrony.

    Parameters
    ----------
    timestamps : numpy.ndarray
        An N x 2 array where column 0 is times and column 1 is unit IDs.
    random_seed : int or None, optional
        Seed for the random number generator to for reproducibility.
        Default is None.

    Returns
    -------
    surrogate_timestamps : numpy.ndarray
        A new N x 2 array sorted chronologically with ISI-shuffled spikes.
    """
    rng = np.random.default_rng(random_seed)
    
    # Extract unique unit IDs present in this dataset
    unique_units = np.unique(timestamps[:, 1])
    
    # We will collect the newly generated spikes in a list of arrays
    shuffled_units_data = []
    
    for unit in unique_units:
        # 1. Isolate the spike times for this specific unit
        unit_mask = (timestamps[:, 1] == unit)
        unit_times = timestamps[unit_mask, 0]
        
        # If a neuron only spiked 0 or 1 time, it has no ISIs to shuffle
        if len(unit_times) <= 1:
            # Keep the spike(s) exactly as they were
            shuffled_times = unit_times
        else:
            # 2. Calculate the Inter-Spike Intervals (ISIs)
            isis = np.diff(unit_times)
            
            # 3. Randomly shuffle the order of the intervals
            rng.shuffle(isis)
            
            # Reconstruct the spike train using the cumulative sum.
            # We anchor the first spike to its original starting time 
            # to preserve the absolute onset of the neuron's activity.
            t0 = unit_times[0]
            shuffled_times = np.concatenate(([t0], t0 + np.cumsum(isis)))
            
        # Combine the new times with the unit ID column
        unit_shuffled_array = np.column_stack((shuffled_times, np.full_like(shuffled_times, unit)))
        shuffled_units_data.append(unit_shuffled_array)
        
    # Concatenate all reconstructed units back into a single matrix
    surrogate_timestamps = np.vstack(shuffled_units_data)
    
    # Re-sort chronologically by time
    surrogate_timestamps = surrogate_timestamps[surrogate_timestamps[:, 0].argsort()]
    
    return surrogate_timestamps

def _shuffle_isi_binary(binary_data, random_seed=None):
    """
    Creates a surrogate dataset by shuffling the Inter-Spike Intervals (ISIs) 
    independently for each individual unit.
    
    This preserves each unit's average firing rate and ISI distribution 
    (first-order statistics) while destroying temporal correlations, 
    bursting structures, and cross-unit synchrony.

    Parameters
    ----------
    binary_data : ndarray of shape (n_units, n_timebins)
        Spike timing matrix with units in rows and time bins in columns (N,T).
    random_seed : int or None, optional
        Seed for the random number generator to for reproducibility.
        Default is None.

    Returns
    -------
    surrogate_data : numpy.ndarray
        A new N x T array with ISI-shuffled spikes.
    """
    rng = np.random.default_rng(random_seed)
    
    # Pre-allocate a clean matrix of zeros with the exact same shape
    surrogate_data = np.zeros_like(binary_data)
    
    # Iterate through each row (neuron/voxel)
    for i in range(binary_data.shape[0]):
        # Find the exact bin indices where this unit is active (i.e., where data == 1)
        spike_idx = np.nonzero(binary_data[i, :])[0]
        
        if len(spike_idx) <= 1:
            # Not enough spikes to shuffle, map them directly to the new matrix
            surrogate_data[i, spike_idx] = binary_data[i, spike_idx]
        else:
            # Calculate the intervals between active bins
            isis = np.diff(spike_idx)
            
            # Randomly shuffle the order of the intervals
            rng.shuffle(isis)
            
            # Reconstruct the new active bin indices, anchoring to the first spike
            t0 = spike_idx[0]
            new_spike_idx = np.concatenate(([t0], t0 + np.cumsum(isis)))
            
            # Map the spikes back into the clean matrix
            surrogate_data[i, new_spike_idx] = binary_data[i, spike_idx]
            
    return surrogate_data

def shuffle_isi(data, data_format='timeseries', random_seed=None):
    """
    Creates a surrogate dataset by shuffling the Inter-Spike Intervals (ISIs) 
    independently for each individual unit.
    
    This preserves each unit's average firing rate and ISI distribution 
    (first-order statistics) while destroying temporal correlations, 
    bursting structures, and cross-unit synchrony.

    Parameters
    ----------
    data : numpy.ndarray
        The input neural data. Can be either a dense binary matrix (timeseries) 
        or a sparse 2-column array (timestamps).
    data_format : {'timeseries', 'tabular', 'numpy_2col'}, optional
        A string flag indicating the format of the input `data`. 
        Default is 'timeseries'.
    random_seed : int, optional
        Seed for the random number generator to ensure reproducibility.
        Default is None.

    Returns
    -------
    numpy.ndarray
        The surrogate dataset in the exact same format as the input.
    """
    if data_format == 'timeseries':
        return _shuffle_isi_binary(data, random_seed)
    elif data_format == 'tabular' or data_format == 'numpy_2col':
        return _shuffle_isi_timestamps(data, random_seed)
    else:
        raise ValueError("data_format must be either 'timeseries', 'tabular' or 'numpy2col'.")

def binary_array_from_stamps(x,binsize_ms):
    """
    Creates a binary array out of timestamps for a chosen bin size

    Parameters
    ----------
    x : ndarray of floats
        timeStamps in (t,neuron_j) form
    binsize_ms : float 
        amount of time in each array slot (ms)

    Returns
    -------
    binary_array : ndarray of integers
        (N,nbins) array with 1's on spike times t for neuron N
    """

    # map IDs so the first neuron is neuron 0, second is 1 and so on
    unique_ids = np.unique(x[:, 1])
    mapping = {old: new for new, old in enumerate(unique_ids)}

    # apply mapping to second column
    remapped = x.copy()
    remapped[:, 1] = np.vectorize(mapping.get)(x[:, 1])

    # declare binary array based on last spike and number of units
    nbins = int(1+(np.max(remapped[:,0]))/binsize_ms)
    N = len(np.unique(remapped[:,1]))
    binary_array = np.zeros((N,nbins),dtype=int)
    T = len(binary_array[0,:])
    for i,j in remapped:
        timeSlot = int(i/binsize_ms)
        if timeSlot<T:
            binary_array[int(j),timeSlot] = 1

    return binary_array

def binary_array_from_zscore(x, threshold):
    """
    Binarizes continuous time series data based on a row-wise z-score threshold.

    Parameters
    ----------
    x : numpy.ndarray
        Continuous time series matrix of shape (N, T), where N is the number of 
        variables and T is the number of time points.
    threshold : float
        The number of standard deviations above the mean required to register 
        a binary event (1).

    Returns
    -------
    binary_array : numpy.ndarray
        Binarized integer array containing only active, valid rows.
    clu_list : numpy.ndarray
        Array of remaining variable indices, preserving their original positions 
        relative to the input matrix.
    """
    # Vectorized calculation of z-scores across the time axis (columns)
    z_scores = stats.zscore(x, axis=1)
    
    # Generate the binary matrix using a boolean comparison
    b_matrix = (z_scores > threshold).astype(int)
    
    # Create a mask to filter out rows with no activity or containing NaN values
    has_activity = np.count_nonzero(b_matrix, axis=1) > 0
    has_nan = np.isnan(z_scores).any(axis=1)
    mask = has_activity & ~has_nan
    
    # Apply the mask to preserve original spatial index tracking
    # (This is not currently in use, but may be useful)
    clu_list = np.arange(len(x))
    clu_list = clu_list[mask]
    binary_array = b_matrix[mask]
    
    return binary_array

def binary_array_from_zscore_maxima(x, threshold):
    """
    Binarizes continuous time series data by detecting local maxima (peaks) 
    that exceed a specified row-wise z-score threshold.

    Parameters
    ----------
    x : numpy.ndarray
        Continuous time series matrix of shape (N, T), where N is the number of 
        variables and T is the number of time points.
    threshold : float
        The number of standard deviations a local maximum must exceed to be 
        considered a binary event (1).

    Returns
    -------
    binary_array : numpy.ndarray
        Binarized integer array containing only active, valid rows.
    clu_list : numpy.ndarray
        Array of remaining variable indices, preserving their original positions 
        relative to the input matrix.
    """
    N, T = x.shape
    
    # Vectorized calculation of z-scores across the time axis (columns)
    z_scores = stats.zscore(x, axis=1)
    
    # Initialize an empty binary matrix matching the shape of the input
    b_matrix = np.zeros((N, T), dtype=int)
    
    # Extract peaks for each row from the precomputed z-score matrix
    for i in range(N):
        # Find local maxima indices that pass the height criteria
        peaks, _ = find_peaks(z_scores[i], height=threshold)
        b_matrix[i, peaks] = 1

    # Create a mask to filter out rows with no activity or containing NaN values
    has_activity = np.count_nonzero(b_matrix, axis=1) > 0
    has_nan = np.isnan(z_scores).any(axis=1)
    mask = has_activity & ~has_nan
    
    # Apply the mask to preserve original spatial index tracking
    clu_list = np.arange(N)[mask]
    binary_array = b_matrix[mask]
    
    return binary_array

def binarize_data(data, data_format='timeseries', binary_method=None, 
             binsize_ms=1.0, zscore_threshold=2.0):
    """
    Transforms raw neural data into a dense binary matrix.
    
    If the data is already a binarized timeseries, it passes through untouched.
    
    Parameters
    ----------
    data : numpy.ndarray
        The input data (either timestamps or continuous timeseries).
    data_format : {'timeseries', 'timestamps'}, optional
        The format of the input data. Default is 'timeseries'.
    binary_method : {'zscore_threshold', 'zscore_maxima', None}, optional
        The method used to binarize continuous timeseries data. If None, 
        'timeseries' data is assumed to already be a dense binary matrix and is 
        returned untouched. Default is None. Other formats will be assumed to be
        event timestamps and binarized according to its event times.
    binsize_ms : float, optional
        The temporal resolution for binning timestamp data. Default is 1.0.
    zscore_threshold : float, optional
        The threshold applied when using z-score binarization methods. 
        Default is 2.0.

    Returns
    -------
    numpy.ndarray
        A dense 2D binary matrix of shape (N_neurons, T_bins).
    """
    if data_format !='timeseries':
        # Convert sparse events to dense binary matrix
        return binary_array_from_stamps(data, binsize_ms=binsize_ms)
        
    elif data_format == 'timeseries':
        # Process continuous data
        if binary_method == 'zscore_threshold':
            return binary_array_from_zscore(data, threshold=zscore_threshold)
            
        elif binary_method == 'zscore_maxima':
            return binary_array_from_zscore_maxima(data, threshold=zscore_threshold)
            
        elif binary_method is None:
            # Data is already binarized (pass-through)
            return data
            
        else:
            raise ValueError(f"Unknown binary_method: '{binary_method}'. "
                             "Use 'zscore_threshold', 'zscore_maxima', or None.")
    else:
        raise ValueError("data_format must be either 'timeseries', 'tabular' or 'numpy_2col'")

def is_function_observable(observable):
    function_list = [obs.covariance_spectrum, obs.autocorrelation_function, obs.activity_distribution]
    if observable in function_list:
        return True
    else:
        return False

            
def average_observable_sample_values(CG_observable, stacked_results):
    if type(CG_observable) != obs.max_covariance_eigenvalue and type(CG_observable) != obs._avalanche_covariance_eigenvalue:
        CG_observable.avg_across_windows = np.mean(stacked_results, axis=0)
        CG_observable.std_across_windows = np.std(stacked_results, axis=0)
        CG_observable.exponent, CG_observable.exponent_error, CG_observable.exponent_r2 = \
            get_scaling_exponent(CG_observable.avg_across_windows)
    else:
        # for observables that are eigenvalues of clusters, the first point is not valid (eigenvalue of a single variable)
        CG_observable.avg_across_windows = np.mean(stacked_results, axis=0)
        CG_observable.std_across_windows = np.std(stacked_results, axis=0)
        CG_observable.exponent, CG_observable.exponent_error, CG_observable.exponent_r2 = \
            get_scaling_exponent(CG_observable.avg_across_windows[1:], max_ev=True)
    return CG_observable

def average_across_windows_for_functions(values,rg_steps):
    """
    Compute the mean and standard deviation across windows for
    function-valued observables at each PRG iteration.

    This function is intended for observables whose outcome at each
    renormalization group (RG) step is not a scalar but a function
    (i.e., a vector evaluated over some domain, such as time, scale,
    or lag).

    For each RG step k, the function stacks the function-valued results
    obtained from different windows (or trials), and computes the mean
    and standard deviation across windows at each point of the function
    domain.

    Parameters
    ----------
    values : list of list of numpy.ndarray
        Function-valued observable results across windows and RG steps.
        The outer list indexes windows (or trials), and the inner list
        indexes RG steps.

        Specifically, ``values[i][k]`` is a 1D numpy array representing
        the observable as a function at RG step k for window i.
        All windows are assumed to have the same number of RG steps and
        the same function length at each step.

    rg_steps : int
        Number of renormalization group (coarse-graining) steps.

    Returns
    -------
    avg_across_windows : list of numpy.ndarray
        List of length ``rg_steps``. Entry k is a 1D array giving the
        pointwise mean of the function-valued observable across windows
        at RG step k.

    std_across_windows : list of numpy.ndarray
        List of length ``rg_steps``. Entry k is a 1D array giving the
        pointwise standard deviation of the function-valued observable
        across windows at RG step k.
    """
    ntrials = len(values)
    
    avg_across_windows = [None] * rg_steps
    std_across_windows = [None] * rg_steps

    for k in range(rg_steps):
        # find the minimum length across all trials 
        # (autocorrelation functions can have different lengths due to different window sizes)
        min_len = min(len(values[i][k]) for i in range(ntrials))
        values_per_step = np.zeros((ntrials, min_len))
        
        for i in range(ntrials):
            values_per_step[i] = values[i][k][:min_len]
            
        avg_across_windows[k] = np.mean(values_per_step, axis=0)
        std_across_windows[k] = np.std(values_per_step, axis=0)
        
    return avg_across_windows, std_across_windows

def average_observable_sample_values_for_functions(CG_observable, stacked_results, rg_steps, raw_timeseries):
    CG_observable.avg_across_windows, CG_observable.std_across_windows = average_across_windows_for_functions(stacked_results, rg_steps)

    if type(CG_observable) == obs.covariance_spectrum:
        CG_observable.exponent, CG_observable.exponent_error, CG_observable.exponent_r2 = \
        get_scaling_exponent(CG_observable.avg_across_windows[-1][:CG_observable.fit_length], spectrum = True)
        CG_observable.exponent = -CG_observable.exponent
        CG_observable.mp_x_fit, CG_observable.mp_y_fit, CG_observable.mp_lambda_plus, CG_observable.mp_sigma, CG_observable.mp_Q = CG_observable.get_marchenko_pastur(raw_timeseries)
        CG_observable.pdf_pl_exponent, CG_observable.pdf_pl_normalization_constant = CG_observable.get_distribution_power_law()
        
    CG_observable.rg_steps = rg_steps

    return CG_observable

def average_across_windows_for_functions(values,rg_steps):
    """
    Compute the mean and standard deviation across windows for
    function-valued observables at each PRG iteration.

    This function is intended for observables whose outcome at each
    renormalization group (RG) step is not a scalar but a function
    (i.e., a vector evaluated over some domain, such as time, scale,
    or lag).

    For each RG step k, the function stacks the function-valued results
    obtained from different windows (or trials), and computes the mean
    and standard deviation across windows at each point of the function
    domain.

    Parameters
    ----------
    values : list of list of numpy.ndarray
        Function-valued observable results across windows and RG steps.
        The outer list indexes windows (or trials), and the inner list
        indexes RG steps.

        Specifically, ``values[i][k]`` is a 1D numpy array representing
        the observable as a function at RG step k for window i.
        All windows are assumed to have the same number of RG steps and
        the same function length at each step.

    rg_steps : int
        Number of renormalization group (coarse-graining) steps.

    Returns
    -------
    avg_across_windows : list of numpy.ndarray
        List of length ``rg_steps``. Entry k is a 1D array giving the
        pointwise mean of the function-valued observable across windows
        at RG step k.

    std_across_windows : list of numpy.ndarray
        List of length ``rg_steps``. Entry k is a 1D array giving the
        pointwise standard deviation of the function-valued observable
        across windows at RG step k.
    """
    ntrials = len(values)
    
    avg_across_windows = [None] * rg_steps
    std_across_windows = [None] * rg_steps

    for k in range(rg_steps):
        # find the minimum length across all trials 
        # (autocorrelation functions can have different lengths due to different window sizes)
        min_len = min(len(values[i][k]) for i in range(ntrials))
        values_per_step = np.zeros((ntrials, min_len))
        
        for i in range(ntrials):
            values_per_step[i] = values[i][k][:min_len]
            
        avg_across_windows[k] = np.mean(values_per_step, axis=0)
        std_across_windows[k] = np.std(values_per_step, axis=0)
        
    return avg_across_windows, std_across_windows

def make_plots_for_observables(result_dict, 
                               prg_params: AnalysisParams, 
                               show_plots=False, 
                               save_plots=False, plots_path=None, 
                               file_key=None, 
                               figsize=(8, 8)):
    """
    Data visualization for calculated metrics.

    Iterates through the requested observables list, initializes separate 
    matplotlib figure canvases, and maps the results data directly to their 
    corresponding visualization module while applying user style configurations.

    Parameters
    ----------
    result_dict : dict
        Calculated metrics data compiled from `run_PRG`.
    prg_params : AnalysisParams
        Active parameter dataclass defining style overrides and active metrics.
    show_plots : bool, optional
        If True, displays the generated figure canvas on screen. Default is False.
    save_plots : bool, optional
        If True, exports the generated figures to disk as PNG files. Default is False.
    plots_path : str, optional
        Directory destination path targeting export file writes. Default is None.
    file_key : str, optional
        A unique file prefix string added to the export file names. Default is None.
    figsize : tuple, optional
        The layout dimensions (width, height) in inches for the figure canvas. 
        Default is (8, 8).
    """
    plot_call_list = {"mean_variance": plot.plot_mean_variance,
                    "log_silence_probability": plot.plot_log_silence_probability,
                    "max_covariance_eigenvalue": plot.plot_max_covariance_eigenvalue,
                    "avalanche_covariance_eigenvalue": plot.plot_avalanche_covariance_eigenvalue,
                    "covariance_spectrum": plot.plot_covariance_spectrum,
                    "activity_distribution": plot.plot_activity_distribution,
                    "autocorrelation_function": plot.plot_autocorrelation_function,
                    "decay_time": plot.plot_decay_time}

    observable_call_list = prg_params.observables

    for call in observable_call_list:
        fig = plt.figure(figsize=figsize)
        observable_name = call.__name__
        observable_object = result_dict[call.__name__]
        plot_call_list[observable_name](observable_object, style_config=prg_params)
        if show_plots:
            plt.show()
        elif save_plots and plots_path is not None:
            plot_file = os.path.join(plots_path, f"{file_key}_{observable_name}.png")
            plt.savefig(plot_file)
            plt.close()

    
def save_manifest(files, prg_params: AnalysisParams):
    """
    Export runtime parameter details and generate unique directory validation hashes.

    Parameters
    ----------
    files : list of str
        List of file paths used this runtime sequence.
    prg_params : AnalysisParams
        Active master configuration dataclass holding operational parameters.

    Returns
    -------
    str
        A unique MD5/SHA tracking hash representing this distinct configuration state.
    """
    # create parent folder for analysis results
    data_dir = os.path.dirname(files[0])
    simulation_folder_name = os.path.basename(data_dir)
    save_path = os.path.join('./results/', simulation_folder_name)
    os.makedirs(save_path, exist_ok=True)

    clean_params = dataclasses.asdict(prg_params)
    # Convert objects to strings: [obs.mean_variance] -> ["mean_variance"]
    clean_params["observables"] = [obs.__name__ for obs in prg_params.observables]
    # sort_keys=True ensures the hash is identical even if you write the dict in a different order later
    params_str = json.dumps(clean_params, sort_keys=True)
    full_hash = hashlib.sha256(params_str.encode('utf-8')).hexdigest()
    analysis_hash = full_hash[:40]  # Truncate to 40 chars to keep paths manageable

    # create analysis-specific save directory
    analysis_save_path = os.path.join(save_path, f"analysis_{analysis_hash}")
    os.makedirs(analysis_save_path, exist_ok=True)

    # save human-readable manifest JSON (only write if it doesn't exist to save I/O)
    manifest_file = os.path.join(analysis_save_path, "analysis_manifest.json")

    manifest = {
        "files": [os.path.basename(f) for f in files],
        "prg_params": clean_params,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=4)

    results_path = os.path.join(analysis_save_path, "results")
    os.makedirs(results_path, exist_ok=True)
    plots_path = os.path.join(analysis_save_path, "plots")
    os.makedirs(plots_path, exist_ok=True)

    return results_path, plots_path

def save_result_dictionaries(result_dict, prg_params,file_key, analysis_save_path):
    """
    Serialize observable values and exponents into a pickle file.

    Parameters
    ----------
    result_dict : dict
        A summary dictionary mapping observable names to their populated objects.
    prg_params : AnalysisParams
        The active parameter dataclass settings utilized during the run execution.
    file_key : str
        The original input data file name used to name the output pickle.
    analysis_save_path : str
        The directory destination folder where the output file will be written.
    """
    file_results = {}

    output_file = os.path.join(analysis_save_path, file_key.replace('.gdf', '.pkl'))
    for observable in prg_params.observables:
        name = observable.__name__
        data_obj = result_dict[observable.__name__]

        file_results[name] = {
            "exponent": getattr(data_obj, "exponent", None),
            "exponent_error": getattr(data_obj, "exponent_error", None),
            "exponent_r2": getattr(data_obj, "exponent_r2", None),

            # store arrays
            "avg_across_windows": data_obj.avg_across_windows if hasattr(data_obj, "avg_across_windows") else None,
            "std_across_windows": data_obj.std_across_windows if hasattr(data_obj, "std_across_windows") else None,
        }
        if name == "covariance_spectrum":
            file_results[name]["mp_x_fit"] = getattr(data_obj, "mp_x_fit", None)
            file_results[name]["mp_y_fit"] = getattr(data_obj, "mp_y_fit", None)
            file_results[name]["mp_lambda_plus"] = getattr(data_obj, "mp_lambda_plus", None)
            file_results[name]["mp_sigma"] = getattr(data_obj, "mp_sigma", None)
            file_results[name]["mp_Q"] = getattr(data_obj, "mp_Q", None)
            file_results[name]["pdf_pl_exponent"] = getattr(data_obj, "pdf_pl_exponent", None)
            file_results[name]["pdf_pl_normalization_constant"] = getattr(data_obj, "pdf_pl_normalization_constant", None)

    # ------------------ PACKAGE EVERYTHING ------------------
    output_dict = {
        "source_file": file_key,
        # We don't necessarily need to save params in EVERY pickle because we 
        # have the manifest, but we keep it for redundancy.
        "params": prg_params, 
        "results": file_results
    }

    # ------------------ SAVE ------------------
    with open(output_file, "wb") as f:
        pickle.dump(output_dict, f)

    print(f"Saved → {output_file}")


