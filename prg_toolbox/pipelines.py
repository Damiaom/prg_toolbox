from .config import *
from .analysis_tools import *

def run_PRG(timestamps, user_params: AnalysisParams = None):
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
    if user_params is None:
        params = AnalysisParams()
    elif not isinstance(user_params, AnalysisParams):
        raise TypeError(
            f"Expected 'user_params' to be an instance of AnalysisParams, "
            f"got {type(user_params).__name__} instead."
        )
    else:
        params = user_params

    # Parameter extraction via attributes
    observable_call_list = params.observables
    rg_steps = params.rg_steps
    cluster_method = params.cluster_method
    
    # Subsampling parameters
    nsamples = params.subsampling.nsamples
    samplesize = params.subsampling.samplesize
    random_seed = params.subsampling.random_seed
    
    # Time windowing parameters
    binsize = params.time_slicing.binary_binsize
    time_window_ms = params.time_slicing.window_duration_ms
    slice_overlap = params.time_slicing.overlap_fraction
    discard_transient_time_ms = params.time_slicing.discard_transient_time_ms


    # Initialize storage tracking metrics across all generated realizations
    observable_stacker = {obs.__name__: [] for obs in observable_call_list}

    # Strip the transient period if specified, ensuring that only the steady-state dynamics are analyzed
    timestamps = discard_transient(timestamps, discard_transient_time_ms) if discard_transient_time_ms > 0 else timestamps

    time_windows_list = slice_timestamps_by_window(timestamps, time_window_ms, slice_overlap) if time_window_ms is not None else [timestamps]

    # Evaluation Loop (Slices x Subsamples)
    for slice_idx, current_slice in enumerate(time_windows_list):
        for sample_idx in range(nsamples):
            
            # Unique deterministic seed mapping combining slice iteration and sample iteration
            combined_seed = random_seed * (slice_idx + 1) * (sample_idx + 1)
            
            # Execute unit/spatial subsampling within the current clean temporal window
            if samplesize is not None:
                subsample = pick_random_sample_from_stamps(current_slice, samplesize, random_seed=combined_seed)
            else:
                subsample = current_slice
            
            # Construct configuration space mapping
            binary_time_series = binary_array_from_stamps(subsample, binsize)
            cgvar = prg.CGVariables(binary_time_series, cluster_method=cluster_method, rg_steps=rg_steps)

            # Evaluate observables on the current spatio-temporal realization
            for call in observable_call_list:
                CG_observable = call(cgvar)
                observable_stacker[call.__name__].append(CG_observable.avg_across_windows)

    # 4. Structural Aggregation and Averaging
    result_dict = {}
    for call in observable_call_list:
        key = call.__name__
        if not is_function_observable(call):
            stacked_data = np.stack(observable_stacker[key], axis=0)
            CG_observable = call(cgvar)  
            CG_observable = average_observable_sample_values(CG_observable, stacked_data)
        else:
            stacked_data = observable_stacker[key]
            CG_observable = call(cgvar)
            CG_observable = average_observable_sample_values_for_functions(CG_observable, stacked_data, rg_steps, binary_time_series)

        result_dict[key] = CG_observable

    return result_dict


def run_PRG_in_directory(file_directory, 
                        skipped_files_list = [],
                        file_format = "tabular",
                        user_params = None,
                        show_plots = False,
                        save_plots = False,
                        save_results = False):

    if user_params is None:
        prg_params = AnalysisParams()
    elif not isinstance(user_params, AnalysisParams):
        raise TypeError(
            f"Expected 'user_params' to be an instance of AnalysisParams, "
            f"got {type(user_params).__name__} instead."
        )
    else:
        prg_params = user_params

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
            make_plots_for_observables(result_dict, prg_params.observables, show_plots, save_plots, plots_path, file_key)

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
        make_plots_for_observables(result_dict, prg_params.observables, show_plots, save_plots, plots_path, file_key)

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
    if user_params is None:
        prg_params = AnalysisParams()
    elif not isinstance(user_params, AnalysisParams):
        raise TypeError(
            f"Expected 'user_params' to be an instance of AnalysisParams, "
            f"got {type(user_params).__name__} instead."
        )
    else:
        prg_params = user_params

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