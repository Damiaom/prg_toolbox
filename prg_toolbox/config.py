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
from typing import List, Callable, Optional, Dict, Any
from . import observables as obs 

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
    colors : dict of str to str, optional
        A dictionary mapping explicit keys to standard matplotlib color codes. Default is None.
    palette : dict of str to str, optional
        A dictionary mapping explicit keys to standard matplotlib color codes. Default is None.
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
    legend: bool = True
    colors: Optional[Dict[str, str]] = None
    palette: Optional[Dict[str, str]] = None
    
    # Specific matplotlib styling overrides
    plot_kwargs: Dict[str, Any] = field(default_factory=dict)
    fill_kwargs: Dict[str, Any] = field(default_factory=dict)
    label_kwargs: Dict[str, Any] = field(default_factory=dict)
    legend_kwargs: Dict[str, Any] = field(default_factory=dict)
    tick_kwargs: Dict[str, Any] = field(default_factory=dict)

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

    
    subsampling: SubsamplingParams = field(default_factory=SubsamplingParams)
    time_slicing: TimeWindowingParams = field(default_factory=TimeWindowingParams)
    plot_style: PlotStyleConfig = field(default_factory=PlotStyleConfig)

