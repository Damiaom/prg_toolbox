# prg_toolbox/__init__.py

"""
PRG Toolbox: A collection of tools for real-space Renormalization Group 
analysis and visualization of scaling exponents.
"""

# 1. Core Data Classes
from .coarse_graining import CGVariables
from .observables import (
    mean_variance, 
    log_silence_probability, 
    max_covariance_eigenvalue, 
    covariance_spectrum, 
    autocorrelation_function, 
    decay_time,
    activity_distribution,
    _avalanche_covariance_eigenvalue
)

# 3. Helper/Utility Functions (Optional - sometimes used in notebooks)
from .utils import get_scaling_exponent
from . import analysis_tools as tools
from . import plotting as plot
from . import config as config
from .pipelines import *

# Define what is accessible when someone does 'from prg_toolbox import *'
__all__ = [
    "CGVariables",
    "mean_variance",
    "log_silence_probability",
    "max_covariance_eigenvalue",
    "covariance_spectrum",
    "autocorrelation_function",
    "decay_time",
    "activity_distribution",
    "avalanche_covariance_eigenvalue",
    "plot_mean_variance",
    "plot_log_silence_probability",
    "plot_max_covariance_eigenvalue",
    "plot_covariance_spectrum",
    "plot_autocorrelation_function",
    "plot_decay_time",
    "plot_activity_distribution"
    "plot_avalanche_covariance_eigenvalue",
    "analysis_tools",
    "config_params",
    "pipelines"
]
