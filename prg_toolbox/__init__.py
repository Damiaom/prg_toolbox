# prg_toolbox/__init__.py
"""
Copyright (c) 2026 Daniel Miranda Castro. Licensed under the MIT License.

PRG Toolbox: A collection of tools for real-space Renormalization Group
analysis and visualization of scaling exponents.
"""

# Core Data Classes
from . import analysis_tools as tools  # uses observables, utils, plot and config
from . import config as config  # uses observables
from . import datasets as datasets  # uses pooch to download example data
from . import plotting as plot  # uses utils with ..utils powerLaw_function
from .coarse_graining import CGVariables
from .observables import (
    _avalanche_covariance_eigenvalue,
    activity_distribution,
    autocorrelation_function,
    covariance_spectrum,
    decay_time,
    log_silence_probability,
    max_covariance_eigenvalue,
    mean_variance,
)
from .pipelines import *  # uses observables, config and analysis_tools

# Helper/Utility Functions
from .utils import get_scaling_exponent

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
    "config",
    "get_scaling_exponent",
    "pipelines",
    "datasets",
]
