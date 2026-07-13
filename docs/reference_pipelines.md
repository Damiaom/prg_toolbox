# Pipelines (`pipelines.py`)

This module provides the high-level coordination for the analysis workflows. It connects the underlying configuration parameters, data preprocessing tools, and PRG routines into single executable processes.

The primary functions within this module manage the execution of the full PRG loop. It includes routines to analyze individual data files, as well as functions designed to process entire directories. For directory-level sweeps, the pipeline automates the sequential loading of files, the application of the coarse-graining and observable tracking steps, and the export of both numerical results and generated visualization figures.

To handle computationally intensive workloads or large collections of datasets, the module also implements parallelized execution paths. These parallel functions distribute the independent file processing tasks across multiple CPU cores, allowing for concurrent analysis across a directory.

::: prg_toolbox.pipelines