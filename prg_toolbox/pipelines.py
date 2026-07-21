"""
Copyright (c) 2026 Daniel Miranda Castro. Licensed under the MIT License.

Pipelines for full PRG workflows.

This module provides high-level workflows to execute the PRG analysis pipeline
across single data files or entire datasets.

Core Functions:
    `run_PRG`: Executes a single analysis loop based on an AnalysisParams configuration.
    `run_PRG_in_directory`: Runs the pipeline across a collection of files within a directory, 
      automating plotting, data packing, and exporting results to a uniquely hashed folder.
    Parallel Variations: Parallelized versions of both workflows are available to speed up 
      computations across larger datasets.
"""
from concurrent.futures import ProcessPoolExecutor
import os
import time
import numpy as np

from .coarse_graining import CGVariables
from .config import *
from .analysis_tools import *
from .verbosity import timed_step, print_if_full


def run_PRG(data, user_params: AnalysisParams = None):
    """
    Execute the PRG analysis pipeline over spatio-temporal splits.

    This function isolates steady-state windows from chronological timestamps, 
    subsamples spatial records, binarizes events, and computes the requested 
    observables across geometric renormalization scales.

    Parameters
    ----------
    data : numpy.ndarray
        The input data. Can be either a dense matrix (timeseries) or 
        a 2-column timestamps array(Column 0: time, Column 1: Unit ID).
    user_params : AnalysisParams, optional
        Configuration dataclass defining steps, slices, methods, and metrics. 
        If None, initializes standard factory defaults.

    Returns
    -------
    result_dict
        A summary dictionary mapping observable class identifiers to their 
        calculated values, scaling exponents and diagnostics.
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
    verbose = params.verbose

    # Data loading parameters
    data_format = params.loading.data_format
    binary_method = params.loading.binary_method
    zscore_threshold = params.loading.zscore_threshold

    # Subsampling parameters
    nsamples = params.subsampling.nsamples
    samplesize = params.subsampling.samplesize
    random_seed = params.subsampling.random_seed
    
    # Time windowing parameters
    binsize = params.time_slicing.binary_binsize_ms
    time_window_ms = params.time_slicing.window_duration_ms
    slice_overlap = params.time_slicing.overlap_fraction
    discard_transient_time_ms = params.time_slicing.discard_transient_time_ms


    # Initialize storage tracking metrics across all generated realizations
    observable_stacker = {obs.__name__: [] for obs in observable_call_list}

    run_start = time.perf_counter() if verbose == "full" else None

    # Strip the transient period if specified, ensuring that only the steady-state dynamics are analyzed
    with timed_step("discard_transient", verbose):
        data = discard_transient(data,data_format = data_format, timeseries_binsize_ms = binsize, transient_time_ms = discard_transient_time_ms) if discard_transient_time_ms > 0 else data

    with timed_step("slice_by_time_window", verbose):
        time_windows_list = slice_by_time_window(data,
                                                 data_format = data_format,
                                                 window_duration_ms=time_window_ms,
                                                 timeseries_binsize_ms=binsize,
                                                 overlap_fraction=slice_overlap, t_start=0.0,
                                                 verbose=verbose) if time_window_ms is not None else [data]

    n_slices = len(time_windows_list)

    # Evaluation Loop (Slices x Subsamples)
    for slice_idx, current_slice in enumerate(time_windows_list):
        for sample_idx in range(nsamples):

            print_if_full(f"time slice {slice_idx + 1}/{n_slices}, sample {sample_idx + 1}/{nsamples}:", verbose)

            # Unique deterministic seed mapping combining slice iteration and sample iteration
            combined_seed = random_seed * (slice_idx + 1) * (sample_idx + 1)

            # Execute unit/spatial subsampling within the current temporal window
            if samplesize is not None:
                with timed_step("pick_random_sample", verbose, indent="  "):
                    subsample = pick_random_sample(current_slice, samplesize, data_format=data_format, random_seed=combined_seed)
            else:
                subsample = current_slice

            # Binarize time series and run coarse graining
            with timed_step("binarize_data", verbose, indent="  "):
                binary_time_series = binarize_data(subsample,
                                                   data_format = data_format,
                                                   binary_method=binary_method,
                                                   binsize_ms=binsize,
                                                   zscore_threshold=zscore_threshold)
            with timed_step("coarse_graining", verbose, indent="  "):
                cgvar = CGVariables(binary_time_series, cluster_method=cluster_method, rg_steps=rg_steps, verbose=verbose)

            # Evaluate observables on the current spatio-temporal realization
            for call in observable_call_list:
                with timed_step(f"observable: {call.__name__}", verbose, indent="  "):
                    CG_observable = call(cgvar)
                    observable_stacker[call.__name__].append(CG_observable.avg_across_windows)
                if hasattr(CG_observable, "exponent"):
                    print_if_full(f"    {call.__name__}: exponent = {CG_observable.exponent:.4g}", verbose)

    # Structural Aggregation and Averaging
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

        if hasattr(CG_observable, "exponent"):
            print_if_full(
                f"{key}: exponent = {CG_observable.exponent:.4g} ± {CG_observable.exponent_error:.4g} (R²={CG_observable.exponent_r2:.3f})",
                verbose,
            )

        result_dict[key] = CG_observable

    if verbose == "full":
        print(f"[timing] run_PRG total: {time.perf_counter() - run_start:.3f}s")

    return result_dict


def run_PRG_in_directory(file_directory,
                        skipped_files_list = None,
                        user_params = None,
                        show_plots = False,
                        save_plots = False,
                        save_results = False):
    """
    Run the PRG analysis pipeline sequentially across a directory of data files.

    Loops through all specified file paths, extracts chronological event 
    timestamps, runs the multi-scale coarse-graining analysis loop, and 
    optionally exports visualization plots and pickled result dictionaries.

    Parameters
    ----------
    file_directory : list of str
        Collection of absolute or relative file paths to analyze.
    skipped_files_list : list, optional
        File names or 1-based integer indices to exclude from processing. 
        Default is an empty list.
    user_params : AnalysisParams, optional
        Configuration settings for the PRG pipeline. If None, default 
        parameters are initialized.
    show_plots : bool, optional
        If True, displays matplotlib visualization figures on screen. 
        Default is False.
    save_plots : bool, optional
        If True, exports generated figure plots as PNG files. Default is False.
    save_results : bool, optional
        If True, saves result matrices and scaling exponents to a uniquely 
        hashed manifest directory. Default is False.
    """

    if skipped_files_list is None:
        skipped_files_list = []

    if user_params is None:
        prg_params = AnalysisParams()
    elif not isinstance(user_params, AnalysisParams):
        raise TypeError(
            f"Expected 'user_params' to be an instance of AnalysisParams, "
            f"got {type(user_params).__name__} instead."
        )
    else:
        prg_params = user_params

    if not show_plots and not save_results and not save_plots and prg_params.verbose != "silent":
        print("Warning: You are not showing or saving any results. Set show_plots or save_plots or save_results to True to see or keep your results.")

    all_files = [f for f in os.listdir(file_directory)]
    N = len(all_files)
    results_path, plots_path = save_manifest(all_files, prg_params) if save_results else (None, None)

    for i, path in enumerate(all_files, start=1):

        file_key = os.path.basename(path)
        path = os.path.join(file_directory, path)
        # Optionally you can filter to skip unwanted files
        if i in skipped_files_list or file_key in skipped_files_list:
            if prg_params.verbose != "silent":
                print(f"[{i}/{N}] {file_key} skipped.")
            continue

        if prg_params.verbose != "silent":
            print(f"[{i}/{N}] Processing file: {file_key}")
        data = load_data(path, user_params = prg_params)

        # Calculating observables and averaging across samples
        result_dict = run_PRG(
            data = data,
            user_params = prg_params
        )

        # Show plots for the observables
        if show_plots or save_plots:
            make_plots_for_observables(result_dict, prg_params, show_plots, save_plots, plots_path, file_key)

        if save_results:
            save_result_dictionaries(result_dict, prg_params, file_key, results_path)
            

# ==========================================
# PARALLELIZATION SETUP
# ==========================================

def process_single_file(path, i, N, 
                        skipped_files_list, 
                        prg_params, 
                        show_plots, save_plots, save_results, 
                        results_path, plots_path):
    
    file_key = os.path.basename(path)

    if i in skipped_files_list or file_key in skipped_files_list:
        if prg_params.verbose != "silent":
            print(f"[{i}/{N}] {file_key} skipped.")
        return

    if prg_params.verbose != "silent":
        print(f"[{i}/{N}] Processing file: {file_key}")

    data = load_data(path, load_params=prg_params)

    result_dict = run_PRG(
            data = data,
            user_params = prg_params
        )
    
    if show_plots or save_plots:
        make_plots_for_observables(result_dict, prg_params.observables, show_plots, save_plots, plots_path, file_key)

    if save_results:
        save_result_dictionaries(result_dict, prg_params, file_key, results_path)

def run_PRG_in_directory_parallel(file_directory,
                        skipped_files_list = None,
                        user_params = None,
                        show_plots = False,
                        save_plots = False,
                        save_results = False,
                        num_cores_to_use=None):
    if skipped_files_list is None:
        skipped_files_list = []

    if user_params is None:
        prg_params = AnalysisParams()
    elif not isinstance(user_params, AnalysisParams):
        raise TypeError(
            f"Expected 'user_params' to be an instance of AnalysisParams, "
            f"got {type(user_params).__name__} instead."
        )
    else:
        prg_params = user_params

    if not show_plots and not save_results and not save_plots and prg_params.verbose != "silent":
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
                path, i, N, skipped_files_list, prg_params, 
                show_plots, save_plots, save_results, results_path, plots_path
            )
            for i, path in enumerate(gdf_files, start=1)
        ]
        
        # This loop forces Python to wait for all files and prints errors if any crash
        for future in futures:
            future.result()