from .plot_imports import *

def labels_decay_time(data, surrogate_data = None):
    """
    Creates axis labels and legend entries for decay time plots.

    Args:
        data (dict)                  : Dictionary with exponent and error
        surrogate_data (dict or None): Optional surrogate data dictionary

    Returns:
        label_dict (dict): Contains xlabel, ylabel, and legend entries
    """
    result_legend = r'$z = %.2f \pm %.2f$' % (
        data["exponent"],
        data["error"],
    )
    if surrogate_data is not None:
        surrogate_legend = r'$z_{Surrogate} = %.2f \pm %.2f$' % (
            surrogate_data["exponent"],
            surrogate_data["error"],
        )
    else:
        surrogate_legend = None

    legend = [result_legend, surrogate_legend] if surrogate_legend else [result_legend]
    label_dict = {
        "xlabel": r"$C_{size}$",
        "ylabel": "Autocorrelation decay time",
        "legend": legend
    }
    return label_dict

def draw_plot_decay_time(values, ax, plot_kw=None, fill_kw=None):
    """
    Draws decay time data and fitted power-law.

    Args:
        values (dict)            : Dictionary with x, y, y_low, y_high, exponent
        ax (matplotlib axis)     : Axis to plot on
        plot_kw (dict or None)   : Line plot kwargs
        fill_kw (dict or None)   : Fill (error band) kwargs

    Returns:
        None
    """
    # Main line
    ax.plot(values["x"], values["y"], label="_nolegend_", **plot_kw)

    # Std error bar
    if values["y_low"] is not None and values["y_high"] is not None:
        ax.fill_between(values["x"], values["y_low"], values["y_high"], label="_nolegend_", **fill_kw)

    log_y0 = np.mean(np.log(values["y"]) - values["exponent"] * np.log(values["x"]))
    y0 = np.exp(log_y0)
    ax.plot(values["x"], powerLaw_function(values["x"], y0, values["exponent"]), color=plot_kw["color"],linestyle = '--', lw=3, alpha = 0.8)

def plot_decay_time(
    data,
    surrogate_data = None,
    ax=None,
    style_config=None,
    plot_kwargs=None,
    fill_kwargs=None,
    label_kwargs=None,
    legend_kwargs=None,
    tick_kwargs = None,
    colors=None,
    legend=True
    ):
    """
    Plots decay time scaling with optional surrogate and reference.

    Args:
        data (object or dict)        : Main data (result object or dictionary)
        surrogate_data (same type)   : Optional surrogate data
        ax (matplotlib axis or None) : Axis to plot on
        style_config (object or None): Global AnalysisParams or PlotStyleConfig dataclass
        plot_kwargs (dict or None)   : Line plot kwargs
        fill_kwargs (dict or None)   : Fill (std. error band) kwargs
        label_kwargs (dict or None)  : Axis label kwargs
        legend_kwargs (dict or None) : Legend kwargs
        tick_kwargs (dict or None)   : Tick styling kwargs
        colors (list or None)        : Custom color scheme
        legend (bool)                : Whether to display legend

    Returns:
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
        colors = colors or config_dict.get("colors")

    #----------- Set up labels and colors -------------------
    ax = ax or plt.gca()
    DEFAULT_LINE_KWARGS, DEFAULT_FILL_KWARGS, DEFAULT_LABEL_KWARGS, DEFAULT_LEGEND_KWARGS, COLORS = set_default_kwargs(colors)

    #------------ Results plot ----------------------------- (Merge defaults with user-provided kwargs)

    values, plot_kw, fill_kw = set_values_and_kwargs(DEFAULT_LINE_KWARGS, DEFAULT_FILL_KWARGS,
                                                     plot_kwargs, fill_kwargs, COLORS,
                                                     data, data_or_surrogate = "data")
    draw_plot_decay_time(values, ax, plot_kw, fill_kw)

    #------------ Surrogate plot ---------------------------- (if present)
    if surrogate_data is not None and type(surrogate_data) == type(data):
         
        values_surrogate, plot_kw, fill_kw = set_values_and_kwargs(DEFAULT_LINE_KWARGS, DEFAULT_FILL_KWARGS,
                                                        plot_kwargs, fill_kwargs, COLORS,
                                                        surrogate_data, data_or_surrogate = "surrogate")
        draw_plot_decay_time(values_surrogate, ax, plot_kw, fill_kw)
        
    elif surrogate_data is not None:
        raise ValueError("Surrogate data must be of the same type as the result data.")
    else:
        values_surrogate = None

    #------------ Reference plot ---------------------------- 
    plot_kw = {**DEFAULT_LINE_KWARGS, "color":COLORS["reference"], **(plot_kwargs or {}), "linestyle": "--", "marker":"None", "alpha":0.6}
    ax.plot(values["x"], values["y"][0]*np.ones(len(values["x"])), **plot_kw)

    #----------- Axis scaling and styling -------------------
    apply_log_axes(ax, base=2)    
    if tick_kwargs is not None:
        style_axes(ax, tick_kwargs)

    #----------- Labels and title ---------------------------
    label_kw = {**DEFAULT_LABEL_KWARGS, **(label_kwargs or {})}
    legend_kw = {**DEFAULT_LEGEND_KWARGS, **(legend_kwargs or {})}
    all_labels = labels_decay_time(values, values_surrogate)
    ax.set_xlabel(all_labels["xlabel"], **label_kw)
    ax.set_ylabel(all_labels["ylabel"], **label_kw) 
    if legend and config_dict.get("show_legend"):
        ax.legend(labels = all_labels["legend"], **legend_kw)
   
