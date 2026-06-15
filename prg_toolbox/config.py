from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict, Any
import prg_toolbox as prg 

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
        prg.mean_variance, 
        prg.log_silence_probability, 
        prg.max_covariance_eigenvalue, 
        prg.covariance_spectrum, 
        prg.activity_distribution
        # prg.autocorrelation_function, 
        # prg.decay_time 
    ])
    rg_steps: int = 7
    cluster_method: str = "pearson"  # Options: 'pearson', 'spearman', 'mutual_info', 'cosine', 'hamming', 'random'

    
    subsampling: SubsamplingParams = field(default_factory=SubsamplingParams)
    time_slicing: TimeWindowingParams = field(default_factory=TimeWindowingParams)
    plot_style: PlotStyleConfig = field(default_factory=PlotStyleConfig)

