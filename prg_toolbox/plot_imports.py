import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Import all the shared helper functions
# from .plot_helper_funcs import (
#     set_default_kwargs, 
#     set_values_and_kwargs, 
#     style_axes, 
#     apply_log_axes, 
#     set_colors_from_palette
# )

# Import the data classes for type-checking across all plot files
from .observables import (
    mean_variance, 
    log_silence_probability, 
    max_covariance_eigenvalue, 
    _avalanche_covariance_eigenvalue,
    decay_time, 
    covariance_spectrum, 
    autocorrelation_function,
    activity_distribution
)

# Shared utility for power laws
from .utils import powerLaw_function

def set_default_kwargs(colors = None):    
    """
    Defines default plotting styles and colors.

    Args:
        colors (list or None):  Optional color list to override defaults
                                [data, surrogate, reference]

    Returns:
        DEFAULT_LINE_KWARGS (dict)   : Line plot defaults
        DEFAULT_FILL_KWARGS (dict)   : Fill (std. error band) defaults
        DEFAULT_LABEL_KWARGS (dict)  : Axis label defaults
        DEFAULT_LEGEND_KWARGS (dict) : Legend defaults
        COLORS (dict)                : Color mapping for data, surrogate, reference
    """
    DEFAULT_LINE_KWARGS = {
        "linewidth": 2.0,
        "linestyle": "None",
        "marker": "o",
        "markersize": 4,
        "markeredgewidth": 0.0,
    }

    DEFAULT_FILL_KWARGS = {
        "alpha": 0.5,
    }

    DEFAULT_LABEL_KWARGS = {
        "fontsize": 12,
        "labelpad": 6,
    }

    DEFAULT_LEGEND_KWARGS = {
        "fontsize": 10,
        "frameon": True,
        "framealpha": 0.9,
        "loc": "best",
    }

    if colors == None:
        COLORS = {
            "data": "tab:purple",
            "surrogate": "tab:blue",
            "reference": "0.7",
        }
    else:
        COLORS = {
            "data": colors[0],
            "surrogate": colors[-2],
            "reference": colors[-1],
        }


    return DEFAULT_LINE_KWARGS, DEFAULT_FILL_KWARGS, DEFAULT_LABEL_KWARGS, DEFAULT_LEGEND_KWARGS, COLORS

def apply_log_axes(ax, base=2):
    ax.set_xscale("log", base=base)
    ax.set_yscale("log")

def style_axes(ax, tick_kw):
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.tick_params(**tick_kw)

def extract_data_from_object(data_object):    
    """
    Args:
        data_object (object): Object with avg_across_windows and std_across_windows

    Returns:
        data (dict): Dictionary with x, y, error bounds, and exponent info
    """
    y = data_object.avg_across_windows
    y_low= y - data_object.std_across_windows
    y_high= y + data_object.std_across_windows
    x = np.array([2**i for i in range(len(y))])

    data = {
        "y": y,
        "x": x,
        "y_low": y_low,
        "y_high": y_high,
        "exponent": getattr(data_object, "exponent", None),
        "error": getattr(data_object, "exponent_error", None),
        "r2": getattr(data_object, "exponent_r2", None),
    }
    return data

def extract_data_from_dictionary(data_dict):
    """
    Args:
        data_dict (dict): Dictionary with keys 'avg', 'std', and optional exponent info

    Returns:
        data (dict): Dictionary with x, y, error bounds, and exponent info
    """
    y = data_dict["avg"]
    y_low= y - data_dict["std"]
    y_high= y + data_dict["std"]
    x = np.array([2**i for i in range(len(y))])

    data = {
        "y": y,
        "x": x,
        "y_low": y_low,
        "y_high": y_high,
        "exponent": data_dict.get("exponent", None),
        "error": data_dict.get("error", None),
        "r2": data_dict.get("r2", None),
    }

    return data

def set_colors_from_palette(number_of_colors, palette=None, data_or_surrogate='data'):
    """
    Generates a list of colors from a colormap.

    Args:
        number_of_colors (int)       : Number of colors to generate
        palette (tuple or None)      : (data_cmap, surrogate_cmap)
        data_or_surrogate (str)      : Selects which palette to use

    Returns:
        colors (list): List of RGBA colors
    """
    if palette is not None:
        cmap = plt.get_cmap(palette[0]) if data_or_surrogate == 'data' else plt.get_cmap(palette[1])
    elif data_or_surrogate == 'data':
        cmap = plt.get_cmap('magma')
    else:
        cmap = plt.get_cmap('viridis')
    colorlist = np.linspace(0.1, 0.8, number_of_colors)
    colors = [cmap(c) for c in colorlist]
    
    return colors

def set_values_and_kwargs(DEFAULT_LINE_KWARGS, DEFAULT_FILL_KWARGS,
                          plot_kwargs, fill_kwargs, COLORS,
                          data, data_or_surrogate = "data"):
    """
    Prepares plotting values and style kwargs.

    Args:
        DEFAULT_LINE_KWARGS (dict)
        DEFAULT_FILL_KWARGS (dict)
        plot_kwargs (dict or None)
        fill_kwargs (dict or None)
        COLORS (dict)
        data (object or dict)       : Result object or dictionary
        data_or_surrogate (str)     : 'data' or 'surrogate'

    Returns:
        values (dict)  : Extracted plotting data
        plot_kw (dict) : Line plot kwargs
        fill_kw (dict) : Fill plot kwargs
    """
    
    plot_kw = {**DEFAULT_LINE_KWARGS, "color":COLORS[data_or_surrogate], **(plot_kwargs or {})}
    fill_kw = {**DEFAULT_FILL_KWARGS, "color":COLORS[data_or_surrogate], **(fill_kwargs or {})}

    if isinstance(data, mean_variance) or isinstance(data, log_silence_probability) or isinstance(data, max_covariance_eigenvalue) or isinstance(data, _avalanche_covariance_eigenvalue) or isinstance(data, decay_time):
        values = extract_data_from_object(data)
    elif isinstance(data, dict):
        values = extract_data_from_dictionary(data)
    else:
        raise ValueError("Data must be either a valid object (mean_variance, log_silence_probability, max_covariance_eigenvalue, or decay_time) or a dictionary.")
    
    return values, plot_kw, fill_kw

