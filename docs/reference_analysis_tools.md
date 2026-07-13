# Analysis Tools (`analysis_tools.py`)

This module provides the essential data preprocessing, partitioning, and result handling utilities required to support the main coarse graining and data analysis methods. It acts as the operational bridge between raw empirical/computational recordings and the calculations of the PRG pipeline.

The data preparation workflow generally begins by taking chronological event timestamps. These continuous event streams are then partitioned into fixed temporal windows and discretized. The module maps continuous spike times into a binary state matrix, where temporal bins containing one or more events are assigned an active state of 1, and silent bins are assigned a state of 0. This procedure creates data format required to run PRG methods. The module also provides aggregation functions that give the alternative to compute ensemble averages and fit scaling exponents across multiple subsampled network realizations.

Beyond data processing, the module also implements statistical controls. It generates surrogate null-model datasets by independently shuffling inter-spike intervals (ISIs), which can be analysed to compare real data results with its surrogate counterparts.

Finally, all methods related to saving results at the end of PRG pipelines are contained in this module.

::: prg_toolbox.analysis_tools
    options:
        members:
            - average_across_windows_for_functions
            - binary_array_from_stamps
            - discard_transient
            - load_data
            - make_plots_for_observables
            - pick_random_sample
            - save_manifest
            - save_result_dictionaries
            - shuffle_isi
            - slice_by_time_window
