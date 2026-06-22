# prg_toolbox/__init__.py
"""
Copyright (c) 2026 Daniel Miranda Castro. Licensed under the MIT License.

PRG Toolbox: A collection of tools for real-space Renormalization Group 
analysis and visualization of scaling exponents.
"""

# Core Data Classes
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

# Helper/Utility Functions
from .utils import get_scaling_exponent
from . import plotting as plot          # uses utils with ..utils powerLaw_function
from . import config as config          # uses observables
from . import analysis_tools as tools   # uses observables, utils, plot and config
from .pipelines import *                # uses observables, config and analysis_tools


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
    "_avalanche_covariance_eigenvalue",
    "plot",
    "tools",
    "config_params",
    "get_scaling_exponent",
    "pipelines"
]
