"""
Copyright (c) 2026 Daniel Miranda Castro. Licensed under the MIT License.

Configuration parameters for PRG analysis.

This module defines the dataclasses used to configure settings and parameters 
for the Phenomenological Renormalization Group (PRG) pipeline, providing a 
centralized configuration layer for functions like `run_PRG`.

The configuration parameters are grouped into the following specialized classes:

    * `PlotStyleConfig`: Passes matplotlib kwargs into downstream plotting scripts.

    * `SubsamplingParams`: Controls parameters that perform the analysis to multiple subsets of the 
        full data and yield averaged results.

    * `TimeWindowingParams`: Governs data preprocessing parameters and time window slicing to perform to
    multiple (temporal) subsets of the data, also yielding averaged results.

    * `AnalysisParams`: The master configuration container with PRG-related parameters and previous classes.
"""

from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict, Any, Union
from . import observables as obs
from .verbosity import validate_verbosity

@dataclass
class PlotStyleConfig:
    """
    Configuration options for downstream visualization and matplotlib layouts.

    Bundles keyword arguments and layout flags to pass customized design configurations
    directly into specialized plotting functions.

    Attributes
    ----------
    legend : bool, optional
        Flag dictating whether a legend box should be rendered on the figure canvas. 
        Default is True.
    colors : str or dict of str to str, optional
        Overrides line colors for the 'data', 'surrogate', and 'reference'
        plot elements. A plain color string (e.g. "green") sets only the
        'data' color; a dict may set any subset of the three keys, e.g.
        ``{"data": "green"}`` -- keys left unspecified keep their default
        color. Default is None (uses all defaults).
    palette : str or dict of str to str, optional
        Overrides the colormap(s) used to color successive RG-step curves
        in function-valued observable plots (e.g. covariance_spectrum,
        autocorrelation_function, activity_distribution). A plain colormap
        name (e.g. "plasma") sets only the 'data' colormap; a dict may set
        either or both of 'data' and 'surrogate' -- keys left unspecified
        keep their default colormap. Default is None (uses all defaults).
    plot_kwargs : dict of str to Any, optional
        Keyword arguments passed directly to `matplotlib.pyplot.plot` calls 
        (e.g., ``linestyle``, ``linewidth``, ``marker``). Default is an empty factory dictionary.
    fill_kwargs : dict of str to Any, optional
        Keyword arguments passed directly to `matplotlib.pyplot.fill_between` calls 
        for error bounds visualization (e.g., ``alpha``, ``hatch``). Default is an empty factory dictionary.
    label_kwargs : dict of str to Any, optional
        Keyword arguments governing title, xlabel, and ylabel properties 
        (e.g., ``fontsize``, ``fontweight``, ``labelpad``). Default is an empty factory dictionary.
    legend_kwargs : dict of str to Any, optional
        Keyword arguments configuring legend properties (e.g., ``loc``, ``frameon``, 
        ``ncol``, ``facecolor``). Default is an empty factory dictionary.
    tick_kwargs : dict of str to Any, optional
        Keyword arguments configuring axis tick look-and-feel variables passed 
        to `ax.tick_params` (e.g., ``labelsize``, ``direction``). Default is an empty factory dictionary.

    Note: Kwargs left blank will trigger the default kwargs defined in set_default_kwargs() at plotting/plot_imports.py,
    """
    # Global/shared layout flags
    show_legend: bool = True
    colors: Optional[Union[str, Dict[str, str]]] = None
    palette: Optional[Union[str, Dict[str, str]]] = None
    
    # Specific matplotlib styling overrides
    plot_kwargs: Dict[str, Any] = field(default_factory=dict)
    fill_kwargs: Dict[str, Any] = field(default_factory=dict)
    label_kwargs: Dict[str, Any] = field(default_factory=dict)
    legend_kwargs: Dict[str, Any] = field(default_factory=dict)
    tick_kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataLoadingParams:
    """Configuration for data ingestion."""
    data_format: str = "timeseries" # 'timeseries', 'tabular' or 'numpy_2col'
    binary_method: Optional[str] = None # None, 'zscore_threshold' or 'zscore_maxima'
    zscore_threshold: float = 2.0

    # --- Timestamp specific ---
    time_col: int = 1
    unit_col: int = 0
    sep: str = r"\s+"
    header: Optional[int] = None
    time_scale_factor: float = 1.0
    
    # --- Timeseries specific ---
    delimiter: Optional[str] = None
    mat_key: Optional[str] = None

@dataclass
class SubsamplingParams:
    """
    Parameters for subsampling of variables (neurons).

    Attributes
    ----------
    samplesize : int, optional
        Number of variables to randomly select per sample draw. If None, the 
        entire population is used. Default is None.
    nsamples : int, optional
        Number of independent random draws to perform for calculating ensemble averages.
        Default is 1.
    random_seed : int, optional
        Seed used to initialize the random number generator for reproducible draws.
        Default is 123.
    """
    samplesize: Optional[int] = None
    nsamples: int = 1
    random_seed: int = 123

@dataclass
class TimeWindowingParams:
    """
    Parameters for temporal windowing and data binarization.

    Attributes
    ----------
    window_duration_ms : float, optional
        Duration of individual trial time windows in milliseconds. If None, the 
        entire time series is analyzed as a single window. Default is None.
    overlap_fraction : float, optional
        Fractional overlap between consecutive sliding windows, bounded between [0.0, 1.0).
        Default is 0.0 (no overlap).
    discard_transient_time_ms : float, optional
        Initial data period to discard before windowing begins (e.g., to skip simulation 
        transients). Default is 0.0.
    binary_binsize_ms : int, optional
        The bin width in milliseconds used to aggregate and binarize raw data into 0 and 1 states.
        Default is 1.
    """
    window_duration_ms: Optional[float] = None
    overlap_fraction: float = 0.0
    discard_transient_time_ms: float = 0.0
    binary_binsize_ms: float = 1.0 

@dataclass
class AnalysisParams:
    """
    Master parameters of the PRG analysis workflow.

    Attributes
    ----------
    observables : list of Callable, optional
        List of uninstantiated observable classes to calculate at each step. 
        Defaults to evaluating: `mean_variance`, `log_silence_probability`, 
        `max_covariance_eigenvalue`, `covariance_spectrum`, and `activity_distribution`.
    rg_steps : int, optional
        Number of iterative coarse-graining reduction loops to apply. Default is 7 
        (stores the raw variables' statistics).
    cluster_method : str, optional
        The metric used to pair similar variables during coarse graining. 
        Options include: 'pearson', 'spearman', 'mutual_information', 'cosine', 
        'hamming', and 'random'. Default is "pearson".
    data_format : str, optional
        Settings for data ingestion. Options are 'timeseries' for (N,T) arrays, 'tabular' for
        timestamps stored in csv/pandas dataframes and 'numpy_2col' for timestamps arrays.
        Default is 'timeseries'.
    subsampling : SubsamplingParams, optional
        Configuration settings for spatial subsampling. Default is a standard
        SubsamplingParams object which performs no subsampling.
    time_slicing : TimeWindowingParams, optional
        Configuration settings for temporal windowing. Default is a standard
        TimeWindowingParams object which performs no temporal windowing.
    plot_style : PlotStyleConfig, optional
        Look-and-feel style settings for generating figures. Default is a standard
        PlotStyleConfig object which trigger the default kwargs defined in
        set_default_kwargs() at plotting/plot_imports.py.
    verbose : str, optional
        Controls console output during the analysis. Options are:
        'silent' (no prints, no warnings), 'warnings' (only warning-style
        messages, e.g. dropped variables during coarse graining or
        misconfiguration notices), or 'full' (warnings plus per-step timing
        and per-observable exponents printed as the analysis progresses).
        Default is 'warnings'.
    """
    observables: List[Callable] = field(default_factory=lambda: [
        obs.mean_variance,
        obs.log_silence_probability,
        obs.max_covariance_eigenvalue,
        obs.covariance_spectrum,
        obs.activity_distribution
        # obs.autocorrelation_function,
        # obs.decay_time
    ])
    rg_steps: int = 7
    cluster_method: str = "pearson"  # Options: 'pearson', 'spearman', 'mutual_info', 'cosine', 'hamming', 'random'

    loading: DataLoadingParams = field(default_factory=DataLoadingParams)
    subsampling: SubsamplingParams = field(default_factory=SubsamplingParams)
    time_slicing: TimeWindowingParams = field(default_factory=TimeWindowingParams)
    plot_style: PlotStyleConfig = field(default_factory=PlotStyleConfig)
    verbose: str = "warnings"  # Options: 'silent', 'warnings', 'full'

    def __post_init__(self):
        validate_verbosity(self.verbose)

