from .plot_imports import *

def labels_autocorrelation_function(data, surrogate_data = None):
    """
    Creates axis labels and legend entries for autocorrelation function plots.

    Parameters
    ----------
        data (dict)                  : Dictionary with autocorrelation function
        surrogate_data (dict or None): Optional surrogate data

    Returns
    ----------
        label_dict (dict): Contains xlabel, ylabel, and legend entries
    """
    result_legend = r'$(C_{size} = %d)$' % (2**(len(data["y"])-1))

    if surrogate_data is not None:
        surrogate_legend = 'Surrogate'
    else:
        surrogate_legend = None

    legend = [result_legend, surrogate_legend] if surrogate_legend else [result_legend]
    label_dict = {
        "xlabel": r"Time bins",
        "ylabel": "Normalized Autocorrelation",
        "legend": legend
    }
    return label_dict

def extract_autocorrelation_function_from_object(data_object):
    """
    Extracts autocorrelation function data from an object.

    Parameters
    ----------
        data_object (object): Object with avg_across_windows and std_across_windows

    Returns
    ----------
        data (dict): Dictionary with x, y, error bounds, and metadata
    """
    y = data_object.avg_across_windows
    y_low= [y[k] - data_object.std_across_windows[k] for k in range(len(y))]
    y_high= [y[k] + data_object.std_across_windows[k] for k in range(len(y))]
    x = [np.arange(len(y[k])) - (len(y[k])-1)/2 for k in range(len(y))]
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

def extract_autocorrelation_function_from_dictionary(data_dict):
    """
    Extracts autocorrelation function data from a dictionary.

    Parameters
    ----------
        data_dict (dict): Dictionary with 'avg_across_windows' and 'std_across_windows' entries

    Returns
    ----------
        data (dict): Dictionary with x, y, error bounds, and metadata
    """
    y = data_dict["avg_across_windows"]
    y_low= [y[k] - data_dict["std_across_windows"][k] for k in range(len(y))]
    y_high= [y[k] + data_dict["std_across_windows"][k] for k in range(len(y))]
    x = [np.arange(len(y[k])) - (len(y[k])-1)/2 for k in range(len(y))]
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

def set_autocorrelation_function_values_and_kwargs(DEFAULT_LINE_KWARGS, DEFAULT_FILL_KWARGS,
                          plot_kwargs, fill_kwargs, palette,
                          data, data_or_surrogate = "data"):
    """
    Prepares autocorrelation function values and per-curve plotting kwargs.

    Parameters
    ----------
        data (object or dict)        : Main data
        surrogate_data (same type)   : Optional surrogate data
        ax (matplotlib axis or None) : Axis to plot on
        plot_kwargs (dict or None)   : Line plot kwargs
        fill_kwargs (dict or None)   : Fill kwargs
        label_kwargs (dict or None)  : Axis label kwargs
        legend_kwargs (dict or None) : Legend kwargs
        tick_kwargs (dict or None)   : Tick styling kwargs
        palette (list, tuple or None): Color palettes
        legend (bool)                : Whether to display legend

    Returns
    ----------
        None
    """
    data_type_name = type(data).__name__

    if data_type_name == "autocorrelation_function":
        values = extract_autocorrelation_function_from_object(data)
    elif isinstance(data, dict):
        values = extract_autocorrelation_function_from_dictionary(data)
    else:
        raise ValueError("Data must be either a autocorrelation_function object or a dictionary.")
    
    number_of_colors = len(values["y"])
    colors_by_iteration = set_colors_from_palette(number_of_colors,palette, data_or_surrogate=data_or_surrogate)
    plot_kw = [{**DEFAULT_LINE_KWARGS, "color":colors_by_iteration[k], **(plot_kwargs or {})} for k in range(len(values["y"]))]
    fill_kw = [{**DEFAULT_FILL_KWARGS, "color":colors_by_iteration[k], **(fill_kwargs or {})} for k in range(len(values["y"]))]
    for k in range(len(values["y"])- 1):
        if "alpha" in plot_kw[k]:
            plot_kw[k]["alpha"] = fill_kw[k]["alpha"]*k/(len(values["y"]))
        else:
            plot_kw[k]["alpha"] = k/(len(values["y"]))
            
        if "alpha" in fill_kw[k]:
            fill_kw[k]["alpha"] = fill_kw[k]["alpha"]*k/(len(values["y"]))
        else:
            fill_kw[k]["alpha"] = k/(len(values["y"]))
    return values, plot_kw, fill_kw

def draw_plot_autocorrelation_function(values, ax, plot_kw=None, fill_kw=None):
    """
    Draws autocorrelation function curves.

    Parameters
    ----------
        values (dict)        : Dictionary with x, y, y_low, y_high
        ax (matplotlib axis) : Axis to plot on
        plot_kw (list)       : List of line kwargs
        fill_kw (list)       : List of fill kwargs

    Returns
    ----------
        None
    """
    for k in range(len(values["y"])):

        # Main line
        if k != len(values["y"])-1:
            ax.plot(values["x"][k], values["y"][k], label="_nolegend_", **plot_kw[k])
        else:
            ax.plot(values["x"][k], values["y"][k], **plot_kw[k])

        # Std error bar
        if values["y_low"][k] is not None and values["y_high"][k] is not None:
            ax.fill_between(values["x"][k], values["y_low"][k], values["y_high"][k], label="_nolegend_", **fill_kw[k])


def limit_plot_at_zero(x, y, ax, excess_bins=5):
    center = len(y) // 2
    right_side = y[center:]
    zero_idx = np.argwhere(right_side <= 0)[0]
    lag = x[center+zero_idx+excess_bins]
    xmin = x[center] - lag
    xmax = x[center] + lag
    ax.set_xlim(xmin, xmax)

def plot_autocorrelation_function(
    data,
    surrogate_data = None,
    ax=None,
    style_config=None,
    plot_kwargs=None,
    fill_kwargs=None,
    label_kwargs=None,
    legend_kwargs=None,
    tick_kwargs = None,
    palette=None,
    legend=True
    ):
    """
    Plots autocorrelation function with multiple curves and optional surrogate.

    Parameters
    ----------
        data (object or dict)        : Main data
        surrogate_data (same type)   : Optional surrogate data
        ax (matplotlib axis or None) : Axis to plot on
        style_config (object or None): Global AnalysisParams or PlotStyleConfig dataclass
        plot_kwargs (dict or None)   : Line plot kwargs
        fill_kwargs (dict or None)   : Fill kwargs
        label_kwargs (dict or None)  : Axis label kwargs
        legend_kwargs (dict or None) : Legend kwargs
        tick_kwargs (dict or None)   : Tick styling kwargs
        palette (list, tuple or None): Color palettes
        legend (bool)                : Whether to display legend

    Returns
    ----------
        None
    """ 

    # Extract kwargs from the configuration object if provided
    if style_config is not None:
        # Handle if the user passes the full AnalysisParams instead of just PlotStyleConfig
        if hasattr(style_config, "plot_style"):
            style_config = style_config.plot_style
            
        config_dict = dataclasses.asdict(style_config)
        
        # Function arguments override configuration object parameters
        plot_kwargs = plot_kwargs or config_dict.get("plot_kwargs")
        fill_kwargs = fill_kwargs or config_dict.get("fill_kwargs")
        label_kwargs = label_kwargs or config_dict.get("label_kwargs")
        legend_kwargs = legend_kwargs or config_dict.get("legend_kwargs")
        tick_kwargs = tick_kwargs or config_dict.get("tick_kwargs")
        palette = palette or config_dict.get("palette")
        legend = legend or config_dict.get("show_legend")

    #----------- Set up labels and colors -------------------
    ax = ax or plt.gca()
    DEFAULT_LINE_KWARGS, DEFAULT_FILL_KWARGS, DEFAULT_LABEL_KWARGS, DEFAULT_LEGEND_KWARGS, _ = set_default_kwargs()

    #------------ Results plot ----------------------------- (Merge defaults with user-provided kwargs)

    values, plot_kw, fill_kw = set_autocorrelation_function_values_and_kwargs(DEFAULT_LINE_KWARGS, DEFAULT_FILL_KWARGS,
                                                     plot_kwargs, fill_kwargs, palette,
                                                     data, data_or_surrogate = "data")
    draw_plot_autocorrelation_function(values, ax, plot_kw, fill_kw)

    #------------ Surrogate plot ---------------------------- (if present)
    if surrogate_data is not None and type(surrogate_data) == type(data): 
        values_surrogate, plot_kw, fill_kw = set_autocorrelation_function_values_and_kwargs(DEFAULT_LINE_KWARGS, DEFAULT_FILL_KWARGS,
                                                        plot_kwargs, fill_kwargs, palette,
                                                        surrogate_data, data_or_surrogate = "surrogate")
        draw_plot_autocorrelation_function(values_surrogate, ax, plot_kw, fill_kw)

    elif surrogate_data is not None:
        raise ValueError("Surrogate data must be of the same type as the result data.")
    else:
        values_surrogate = None


    #----------- Axis scaling and styling -------------------
    if tick_kwargs is not None:
        style_axes(ax, tick_kwargs)
    limit_plot_at_zero(values["x"][-1], values["y"][-1], ax)

    #----------- Labels and title ---------------------------
    label_kw = {**DEFAULT_LABEL_KWARGS, **(label_kwargs or {})}
    legend_kw = {**DEFAULT_LEGEND_KWARGS, **(legend_kwargs or {})}
    all_labels = labels_autocorrelation_function(values, values_surrogate)

    ax.set_xlabel(all_labels["xlabel"], **label_kw)
    ax.set_ylabel(all_labels["ylabel"], **label_kw) 
    if legend:
        ax.legend(labels = all_labels["legend"], **legend_kw)
   
