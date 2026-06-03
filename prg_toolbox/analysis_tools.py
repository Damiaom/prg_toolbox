#@title set parameters and run
import os
import hashlib
import json
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

import prg_toolbox as prg
from .utils import get_scaling_exponent

def load_timestamps(file_path, format="tabular", time_col=1, unit_col=0, sep=r"\s+", header=None, scale_factor=1.0):
    r"""
    Loads neuronal spike timestamps and unit IDs from various file formats 
    into a standardized N x 2 NumPy array.

    The output array is guaranteed to be sorted chronologically by timestamp.

    Parameters
    ----------
    file_path : str
        The absolute path to the data file.
    format : {'tabular', 'numpy_2d', 'phy'}, optional
        The format of the input data. 
        - 'tabular': Text-based formats like .csv, .tsv, .txt, or .gdf.
        - 'numpy_2d': A single .npy file containing a 2D array.
        - 'phy': A directory containing split 1D arrays (`spike_times.npy` 
          and `spike_clusters.npy`).
        Default is 'tabular'.
    time_col : int, optional
        The column index containing the timestamp data (0-indexed). Only 
        used if format is 'tabular' or 'numpy_2d'. Default is 1.
    unit_col : int, optional
        The column index containing the neuron/unit ID data (0-indexed). Only 
        used if format is 'tabular' or 'numpy_2d'. Default is 0.
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
    """
    if format == "tabular":
        # Handles CSV, TSV, TXT, GDF uniformly
        df = pd.read_csv(file_path, sep=sep, header=header)
        
        # Extract just the two columns we care about in the right order
        times = df.iloc[:, time_col].to_numpy() * scale_factor
        units = df.iloc[:, unit_col].to_numpy()
        
        timestamps = np.column_stack((times, units))
        
    elif format == "numpy_2d":
        # Handles standard N x 2 numpy arrays
        raw_array = np.load(file_path)
        times = raw_array[:, time_col] * scale_factor
        units = raw_array[:, unit_col]
        
        timestamps = np.column_stack((times, units))
        
    else:
        raise ValueError(f"Unsupported format for timestamps: {format}")

    # Ensure chronological order
    timestamps = timestamps[timestamps[:, 0].argsort()]
    
    return timestamps

def discard_transient(binary_time_series, binsize, dt_ms, transient_time_ms):
    binsize_ms = binsize*dt_ms 
    transient_bins = int(transient_time_ms // binsize_ms)
    return binary_time_series[:, transient_bins:]

def pick_random_sample_from_stamps(timestamps, sample_size, random_seed = 123):
    """
    Subsample rows based on randomly selected unit indices.

    Args:
        timestamps (numpy array of floats)      : 2D array with shape (n_spikes, 2)
        sample_size (int)                       : number of unique units to sample
        random_seed (int)                       : seed for reproducibility

    Returns:
        stamps_sample (numpy array of floats)   : filtered data
    """
    rng = np.random.default_rng(random_seed)
    original_idx = np.unique(timestamps[:, 1])
    selected_units = rng.choice(original_idx, size=sample_size, replace=False)

    mask = np.isin(timestamps[:, 1], selected_units)
    stamps_sample = timestamps[mask]
    return stamps_sample

def binary_array_from_stamps(x,binSize):
    """
    Creates a binary array out of timestamps for a chosen bin size

    Args:
        x(numpy array of integers)            : timeStamps in (t,neuron_j) form
        binSize(scalar)                       : amount of time in each array slot (ms)

    Returns:
        binary_array(numpy array of integers) : (N,nbins) array with 1's on spike times t for neuron N
    """

    # map IDs so the first neuron is neuron 0, second is 1 and so on
    unique_ids = np.unique(x[:, 1])
    mapping = {old: new for new, old in enumerate(unique_ids)}

    # apply mapping to second column
    remapped = x.copy()
    remapped[:, 1] = np.vectorize(mapping.get)(x[:, 1])

    # declare binary array based on last spike and number of units
    nbins = int((np.max(remapped[:,0])+1)/binSize)
    N = len(np.unique(remapped[:,1]))
    binary_array = np.zeros((N,nbins),dtype=int)
    T = len(binary_array[0,:])
    for i,j in remapped:
        timeSlot = int(i/binSize)
        if timeSlot<T:
            binary_array[int(j),timeSlot] = 1

    return binary_array

def is_function_observable(observable):
    function_list = [prg.covariance_spectrum, prg.autocorrelation_function, prg.activity_distribution]
    if observable in function_list:
        return True
    else:
        return False

            
def average_observable_sample_values(CG_observable, stacked_results):
    if type(CG_observable) != prg.max_covariance_eigenvalue and type(CG_observable) != prg.avalanche_covariance_eigenvalue:
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

    if type(CG_observable) == prg.covariance_spectrum:
        CG_observable.exponent, CG_observable.exponent_error, CG_observable.exponent_r2 = \
        get_scaling_exponent(CG_observable.avg_across_windows[-1][:CG_observable.fit_length], spectrum = True)
        CG_observable.exponent = -CG_observable.exponent
        CG_observable.mp_x_fit, CG_observable.mp_y_fit, CG_observable.mp_lambda_plus, CG_observable.mp_sigma, CG_observable.mp_Q = CG_observable.get_marchenko_pastur(raw_timeseries)
        CG_observable.pdf_pl_exponent, CG_observable.pdf_pl_normalization_constant = CG_observable.get_distribution_power_law()
        
    CG_observable.rg_steps = rg_steps

    return CG_observable

def run_PRG(timestamps, user_params=None):
    """
    Compute the average of a coarse-grained observable over multiple random samples.

    Args:
        observable_call_list(list of callables): list of functions applied to CGVariables object (e.g. mean_variance)
        path_to_data (str)                     : path to timestamp data file
        nsamples (int)                         : number of random subsamples
        binsize (int)                          : bin size for time discretization
        samplesize (int)                       : number of timestamps per subsample
        rg_steps (int)                         : number of coarse-graining steps
        random_seed (int)                      : seed for random number generation in subsampling

    Returns:
        CG_observable_dict (dict)              : dictionary with observable names as keys
        CG_observable (object)                 : observable object with averaged results
                                                in the following attributes:
                                                    avg_across_windows (numpy array)
                                                    exponent (float)
                                                    exponent_error (float)
                                                    exponent_r2 (float)
    """
    prg_params = get_default_prg_params()
    if user_params:
        prg_params.update(user_params)

    # explicitly extract parameters for clarity
    observable_call_list = prg_params["observables"]
    rg_steps = prg_params["rg_steps"]
    binsize = prg_params["binsize"]
    dt_ms = prg_params["dt_ms"]
    discard_transient_time_ms = prg_params["discard_transient_time_ms"]
    nsamples = prg_params["nsamples"]
    samplesize = prg_params["samplesize"]
    random_seed = prg_params["random_seed"]

    # one list per observable
    observable_stacker = {obs.__name__: [] for obs in observable_call_list}

    # run coarse-graining and store observable values for each sample
    for i in range(nsamples):
        subsample = pick_random_sample_from_stamps(timestamps, samplesize, random_seed = random_seed*i*nsamples) if samplesize is not None else timestamps
        binary_time_series = binary_array_from_stamps(subsample, binsize)
        if discard_transient_time_ms > 0:
            binary_time_series = discard_transient(binary_time_series, binsize, dt_ms, discard_transient_time_ms)
        cgvar = prg.CGVariables(binary_time_series, rg_steps=rg_steps)


        for call in observable_call_list:
            CG_observable = call(cgvar)
            observable_stacker[call.__name__].append(CG_observable.avg_across_windows)
        # clear_output(wait=True)
        # print(f"sample {i+1}/{nsamples} done at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

    # average each observable according to its structure
    result_dict = {}

    for call in observable_call_list:
        key = call.__name__
        if not is_function_observable(call):
            stacked_data = np.stack(observable_stacker[key], axis=0)
            # reuse structure (has to double calculate, but is
            # cleaner than instantiating empty object)
            CG_observable = call(cgvar)  
            CG_observable = average_observable_sample_values(CG_observable, stacked_data)
        else:
            stacked_data = observable_stacker[key]
            CG_observable = call(cgvar)
            CG_observable = average_observable_sample_values_for_functions(CG_observable, stacked_data, rg_steps, binary_time_series)

        result_dict[key] = CG_observable
        # print(f"Averaged observable: {key} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return result_dict 

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
                               observable_call_list, 
                               show_plots=False, 
                               save_plots=False, plots_path=None, 
                               file_key=None, 
                               figsize=(8, 8)):
        plot_call_list = {"mean_variance": prg.plot_mean_variance,
                        "log_silence_probability": prg.plot_log_silence_probability,
                        "max_covariance_eigenvalue": prg.plot_max_covariance_eigenvalue,
                        "avalanche_covariance_eigenvalue": prg.plot_avalanche_covariance_eigenvalue,
                        "covariance_spectrum": prg.plot_covariance_spectrum,
                        "activity_distribution": prg.plot_activity_distribution,
                        "autocorrelation_function": prg.plot_autocorrelation_function,
                        "decay_time": prg.plot_decay_time}

        for call in observable_call_list:
            fig = plt.figure(figsize=figsize)
            observable_name = call.__name__
            observable_object = result_dict[call.__name__]
            plot_call_list[observable_name](observable_object)
            if show_plots:
                plt.show()
            elif save_plots and plots_path is not None:
                plot_file = os.path.join(plots_path, f"{file_key}_{observable_name}.png")
                plt.savefig(plot_file)
                plt.close()

    
def save_manifest(files, prg_params):

    # create parent folder for analysis results
    data_dir = os.path.dirname(files[0])
    simulation_folder_name = os.path.basename(data_dir)
    save_path = os.path.join('./results/', simulation_folder_name)
    os.makedirs(save_path, exist_ok=True)

    clean_params = prg_params.copy()
    # Convert objects to strings: [prg.mean_variance] -> ["mean_variance"]
    clean_params["observables"] = [obs.__name__ for obs in prg_params["observables"]]
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
    file_results = {}

    output_file = os.path.join(analysis_save_path, file_key.replace('.gdf', '.pkl'))
    for observable in prg_params["observables"]:
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

def get_default_prg_params():    
    return {        
        "observables": [
            prg.mean_variance, 
            prg.log_silence_probability, 
            prg.max_covariance_eigenvalue, 
            prg.covariance_spectrum, 
            prg.activity_distribution
        ],  
        "rg_steps": 7, 
        "binsize": 1, 
        "dt_ms": 1,
        "discard_transient_time_ms": 500,
        "samplesize": None, 
        "nsamples": 1, 
        "random_seed": 123
    }
def run_PRG_in_directory(file_directory, 
                        skipped_files_list = [],
                        file_format = "tabular",
                        user_params = None,
                        show_plots = False,
                        save_plots = False,
                        save_results = False):

    prg_params = get_default_prg_params()
    if user_params:
        prg_params.update(user_params)

    if not show_plots and not save_results and not save_plots:
        print("Warning: You are not showing or saving any results. Set show_plots or save_plots or save_results to True to see or keep your results.")
    
    all_files = [f for f in file_directory]
    N = len(all_files)
    results_path, plots_path = save_manifest(file_directory, prg_params) if save_results else (None, None)

    for i, path in enumerate(all_files, start=1):

        file_key = os.path.basename(path)
        
        # Optionally you can filter to skip unwanted files
        if i in skipped_files_list or file_key in skipped_files_list:
            print(f"[{i}/{N}] {file_key} skipped.")
            continue

        print(f"[{i}/{N}] Processing file: {file_key}")
        timestamps = load_timestamps(path, file_format, time_col=1, unit_col=0, sep=r"\s+", header=None, scale_factor=1.0)

        # Calculating observables and averaging across samples
        result_dict = run_PRG(
            timestamps = timestamps,
            user_params = prg_params
        )

        # Show plots for the observables
        if show_plots or save_plots:
            make_plots_for_observables(result_dict, prg_params["observables"], show_plots, save_plots, plots_path, file_key)

        if save_results:
            save_result_dictionaries(result_dict, prg_params, file_key, results_path)
            

# ==========================================
# 1. PARALLELIZATION SETUP
# ==========================================

def process_single_file(path, i, N, 
                        skipped_files_list, 
                        prg_params, 
                        show_plots, save_plots, save_results, 
                        results_path, plots_path):
    
    file_key = os.path.basename(path)
    
    if i in skipped_files_list or file_key in skipped_files_list:
        print(f"[{i}/{N}] {file_key} skipped.")
        return

    print(f"[{i}/{N}] Processing file: {file_key}")
    
    timestamps = load_timestamps(path, format="tabular", time_col=1, unit_col=0, sep=r"\s+", header=None, scale_factor=1.0)

    result_dict = run_PRG(
            timestamps = timestamps,
            user_params = prg_params
        )
    
    if show_plots or save_plots:
        make_plots_for_observables(result_dict, prg_params["observables"], show_plots, save_plots, plots_path, file_key)

    if save_results:
        save_result_dictionaries(result_dict, prg_params, file_key, results_path)

def run_PRG_in_directory_parallel(file_directory, 
                        skipped_files_list = [],
                        file_format = "tabular",
                        user_params = None,
                        show_plots = False,
                        save_plots = False,
                        save_results = False,
                        num_cores_to_use=None):
    prg_params = get_default_prg_params()
    if user_params:
        prg_params.update(user_params)

    if not show_plots and not save_results and not save_plots:
        print("Warning: You are not showing or saving any results.")
    
    gdf_files = [f for f in file_directory if f.endswith('.gdf')]
    N = len(gdf_files)

    results_path, plots_path = None, None
    if save_results:
        results_path, plots_path = save_manifest(file_directory, prg_params, skipped_files_list)

    # Launching the parallel pool
    if not num_cores_to_use:
        num_cores_to_use = max(1, os.cpu_count() - 3)
    with ProcessPoolExecutor(max_workers=num_cores_to_use) as executor:
        futures = [
            executor.submit(
                process_single_file, 
                path, i, N, skipped_files_list, file_format, prg_params, 
                show_plots, save_plots, save_results, results_path, plots_path
            )
            for i, path in enumerate(gdf_files, start=1)
        ]
        
        # This loop forces Python to wait for all files and prints errors if any crash
        for future in futures:
            future.result()