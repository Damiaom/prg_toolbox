from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict, Any
import .observables as obs 

@dataclass
class PlotStyleConfig:
    # Global/shared layout flags
    legend: bool = True
    colors: Optional[Dict[str, str]] = None
    
    # Specific matplotlib styling overrides
    plot_kwargs: Dict[str, Any] = field(default_factory=dict)
    fill_kwargs: Dict[str, Any] = field(default_factory=dict)
    label_kwargs: Dict[str, Any] = field(default_factory=dict)
    legend_kwargs: Dict[str, Any] = field(default_factory=dict)
    tick_kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SubsamplingParams:
    samplesize: Optional[int] = None
    nsamples: int = 1
    random_seed: int = 123

@dataclass
class TimeWindowingParams:
    window_duration_ms: Optional[float] = None
    overlap_fraction: float = 0.0
    discard_transient_time_ms: float = 0.0
    binary_binsize: int = 1

@dataclass
class AnalysisParams:
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

