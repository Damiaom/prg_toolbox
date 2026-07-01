from .plot_imports import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def labels_marchenko_pastur(spectrum):
    """
    Creates axis labels and legend entries for Marchenko-Pastur fits.
    """
    legend_mp = 'Marchenko-Pastur' + '\n'
    legend_mp += r'($\sigma^2=%.1g$)' % spectrum.mp_sigma

    pdf_exponent = 1 + (1 / spectrum.exponent)
    legend_pl = 'Power-law fit' + '\n'
    legend_pl += r'($\tilde\mu=%.2g$)' % pdf_exponent
    label_dict = {
        "xlabel": "Eigenvalue",
        "ylabel": "Probability Density",
        "legend": [legend_mp, legend_pl]
    }
    return label_dict

def set_marchenko_pastur_kwargs(palette, DEFAULT_LINE_KWARGS, DEFAULT_FILL_KWARGS,
                                hist_kwargs, plot_kwargs, vline_kwargs):
    """
    Prepares kwargs for the empirical histogram, MP fit line, and lambda+ boundary.
    """
    colors_mp = set_colors_from_palette(10, palette, data_or_surrogate="data")
    hist_kw = {"density": True, "color": colors_mp[-4], "edgecolor": colors_mp[-7], "alpha": 0.7, **(hist_kwargs or {})}
    plot_kw = {**DEFAULT_LINE_KWARGS, "color": colors_mp[2], "linestyle": "--", "linewidth": 3, **(plot_kwargs or {})}
    vline_kw = {**DEFAULT_LINE_KWARGS, "color": colors_mp[1], "linestyle": ":", "linewidth": 2, **(vline_kwargs or {})}
    
    return hist_kw, plot_kw, vline_kw

def draw_powerlaw_in_distribution(spectrum):

    pdf_exponent = spectrum.pdf_pl_exponent
    tail_start = spectrum.mp_lambda_plus
    pl_x_fit = np.geomspace(tail_start, np.max(spectrum.avg_across_windows[-1]), 100)
    pl_y_fit = spectrum.pdf_pl_normalization_constant * ((pl_x_fit / tail_start) ** -pdf_exponent)
        
    return pl_x_fit, pl_y_fit

def get_optimal_inset_loc(x, y):
    """
    Determines the emptiest quadrant in a log-log space to safely place an inset.
    """
    # Filter strictly positive values to prevent log10 domain errors
    valid_mask = (x > 0) & (y > 0)
    log_x = np.log10(x[valid_mask])
    log_y = np.log10(y[valid_mask])
    
    # Normalize data to a 0.0 - 1.0 scale to map directly to visual plotting space
    norm_x = (log_x - np.min(log_x)) / (np.max(log_x) - np.min(log_x))
    norm_y = (log_y - np.min(log_y)) / (np.max(log_y) - np.min(log_y))
    
    # Define boolean masks for the four visual quadrants
    quadrants = {
        'upper right': (norm_x > 0.5) & (norm_y > 0.5),
        'upper left':  (norm_x <= 0.5) & (norm_y > 0.5),
        'lower right': (norm_x > 0.5) & (norm_y <= 0.5),
        'lower left':  (norm_x <= 0.5) & (norm_y <= 0.5)
    }
    
    # Count the number of data points falling into each quadrant
    counts = {loc: np.sum(mask) for loc, mask in quadrants.items()}
    
    # Return the string key of the quadrant with the minimum data density
    return min(counts, key=counts.get)

def draw_plot_marchenko_pastur(spectrum, ax, hist_kw=None, plot_kw=None, vline_kw=None):
    """
    Draws the empirical eigenvalue histogram and the theoretical MP distributions directly from the object.
    """
    # To avoid bad plot limits due to zero or near-zero eigenvalues, we set the minimum bin edge to a small positive value (e.g., 1e-6) instead of zero.
    min_eig = np.min(spectrum.avg_across_windows[-1][spectrum.avg_across_windows[-1] > 1e-6]) 
    max_eig = np.max(spectrum.avg_across_windows[-1])
    
    # 40 bins evenly spaced in log-space
    log_bins = np.geomspace(min_eig, max_eig, 40)

    ax.hist(spectrum.avg_across_windows[-1], bins=log_bins, label="_nolegend_", **hist_kw)
    ax.plot(spectrum.mp_x_fit, spectrum.mp_y_fit, color="grey", linestyle="--", linewidth=2)

    pl_x, pl_y = draw_powerlaw_in_distribution(spectrum)
    ax.plot(pl_x, pl_y, **{**plot_kw, "marker": "none", "linewidth": 2, "linestyle": "-"})

def plot_marchenko_pastur(
    spectrum,
    ax=None,
    hist_kwargs=None,
    plot_kwargs=None,
    vline_kwargs=None,
    label_kwargs=None,
    legend_kwargs=None,
    tick_kwargs=None,
    legend=True
    ):
    """
    Plots the Marchenko-Pastur null hypothesis fit against empirical covariance eigenvalues.

    Args:
        spectrum (covariance_spectrum): Object with pre-calculated MP attributes.
        ax (matplotlib axis)          : Axis to plot on
        # ... [standard kwargs]
    """
    # Safety check: Ensure the math has been done
    if not hasattr(spectrum, 'mp_sigma'):
        raise AttributeError(
            "The spectrum object lacks MP fit data. "
            "Call 'spectrum.fit_marchenko_pastur(raw_timeseries)' before plotting."
        )

    ax = ax or plt.gca()
    
    # Fetch defaults using your established pattern
    DEFAULT_LINE_KWARGS, DEFAULT_FILL_KWARGS, DEFAULT_LABEL_KWARGS, DEFAULT_LEGEND_KWARGS, _ = set_default_kwargs()

    # Prepare plotting keyword arguments
    hist_kw, plot_kw, vline_kw = set_marchenko_pastur_kwargs(
        DEFAULT_LINE_KWARGS, DEFAULT_FILL_KWARGS, 
        hist_kwargs, plot_kwargs, vline_kwargs
    )

    # Draw the plot
    draw_plot_marchenko_pastur(spectrum, ax, hist_kw, plot_kw, vline_kw)

    # Apply axis scaling and styling
    apply_log_axes(ax, base=10)
    if tick_kwargs is not None:
        style_axes(ax, tick_kwargs)

    # Apply labels and titles
    label_kw = {**DEFAULT_LABEL_KWARGS, **(label_kwargs or {})}
    legend_kw = {**DEFAULT_LEGEND_KWARGS, **(legend_kwargs or {})}
    all_labels = labels_marchenko_pastur(spectrum)

    ax.set_xlabel(all_labels["xlabel"], **label_kw)
    ax.set_ylabel(all_labels["ylabel"], **label_kw) 
    
    if legend:
        ax.legend(labels=all_labels["legend"], **legend_kw)
    plt.title("Eigenvalue Distribution")

def labels_covariance_spectrum(data, surrogate_data = None):
    """
    Creates axis labels and legend entries for covariance spectrum plots.

    Args:
        data (dict)                  : Dictionary with exponent, error, and spectrum data
        surrogate_data (dict or None): Optional surrogate data dictionary

    Returns:
        label_dict (dict): Contains xlabel, ylabel, and legend entries
    """
    result_legend = r'$\mu = %.2f \pm %.2f$' %(
        data["exponent"],
        data["error"])
    result_legend+= "\n" + r'$(C_{size} = %d)$' % (len(data["y"][-1]))

    if surrogate_data is not None:
        surrogate_legend = r'$\mu_{Surrogate} = %.2f \pm %.2f$' % (
            surrogate_data["exponent"],
            surrogate_data["error"],
        )
    else:
        surrogate_legend = None

    legend = [result_legend, surrogate_legend] if surrogate_legend else [result_legend]
    label_dict = {
        "xlabel": r"$rank/N$",
        "ylabel": "Eigenvalue",
        "legend": legend
    }
    return label_dict

def extract_covariance_spectrum_from_object(data_object):
    """
    Extracts covariance spectrum data from an object.

    Args:
        data_object (object): Object with avg_across_windows and std_across_windows

    Returns:
        data (dict): Dictionary with x, y, error bounds (each one array per iteration) and exponent info
    """
    y = data_object.avg_across_windows
    y_low= [y[k] - data_object.std_across_windows[k] for k in range(len(y))]
    y_high= [y[k] + data_object.std_across_windows[k] for k in range(len(y))]
    x = [np.array([(i+1)/(2**k) for i in range(len(y[k]))]) for k in range(len(y))]
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

def extract_covariance_spectrum_from_dictionary(data_dict):
    """
    Extracts covariance spectrum data from a dictionary.

    Args:
        data_dict (dict): Dictionary with 'avg' and 'std' entries

    Returns:
        data (dict): Dictionary with x, y, error bounds (each one array per iteration) and exponent info
    """
    y = data_dict["avg"]
    y_low= [y[k] - data_dict["std"][k] for k in range(len(y))]
    y_high= [y[k] + data_dict["std"][k] for k in range(len(y))]
    x = [np.array([(i+1)/(2**k) for i in range(len(y[k]))]) for k in range(len(y))]
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

def set_covariance_spectrum_values_and_kwargs(DEFAULT_LINE_KWARGS, DEFAULT_FILL_KWARGS,
                          plot_kwargs, fill_kwargs, palette,
                          data, data_or_surrogate = "data"):
    """
    Prepares covariance spectrum values and per-curve plotting kwargs.

    Args:
        DEFAULT_LINE_KWARGS (dict)
        DEFAULT_FILL_KWARGS (dict)
        plot_kwargs (dict or None)
        fill_kwargs (dict or None)
        palette (tuple or None)      : Color palettes
        data (object or dict)        : Input data
        data_or_surrogate (str)      : 'data' or 'surrogate'

    Returns:
        values (dict)   : Extracted data
        plot_kw (list)  : List of line kwargs (one per curve)
        fill_kw (list)  : List of fill kwargs (one per curve)
    """ 
    data_type_name = type(data).__name__

    if data_type_name == "covariance_spectrum":
        values = extract_covariance_spectrum_from_object(data)
    elif isinstance(data, dict):
        values = extract_covariance_spectrum_from_dictionary(data)
    else:
        raise ValueError("Data must be either a covariance_spectrum object or a dictionary.")
    
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

def draw_plot_covariance_spectrum(values, ax, plot_kw=None, fill_kw=None):
    """
    Draws covariance spectrum curves and fitted power-law to the last iteration.

    Args:
        values (dict)        : Dictionary with x, y, y_low, y_high, exponent
        ax (matplotlib axis) : Axis to plot on
        plot_kw (list)       : List of line kwargs
        fill_kw (list)       : List of fill kwargs

    Returns:
        None
    """
    exponent = -values["exponent"]
    for k in range(1, len(values["y"])):

        # Main line
        ax.plot(values["x"][k], values["y"][k], label="_nolegend_", **plot_kw[k])

        # Std error bar
        if values["y_low"][k] is not None and values["y_high"][k] is not None:
            ax.fill_between(values["x"][k], values["y_low"][k], values["y_high"][k], label="_nolegend_", **fill_kw[k])

    log_y0 = np.mean(np.log(values["y"][-1][1:int(len(values["y"][-1])/10)]) - exponent * np.log(values["x"][-1][1:int(len(values["y"][-1])/10)]))
    y0 = np.exp(log_y0)
    ax.plot(values["x"][-1], powerLaw_function(values["x"][-1], y0, exponent), color=plot_kw[-1]["color"],linestyle = '--', lw=3, alpha = 0.8)
    ax.set_ylim(1e-5,10)

def plot_covariance_spectrum(
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
    legend=True,
    add_marchenko_pastur_inset=True,                    
    inset_bounds=[0.15, 0.15, 0.45, 0.35],
    mp_hist_kwargs=None, mp_plot_kwargs=None, mp_vline_kwargs=None
    ):
    """
    Plots covariance spectrum with one curve per PRG iteration and optional surrogate and Marchenko-Pastur fit.

    Args:
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
        add_marchenko_pastur_inset   : Whether to plot the Marchenko-Pastur fit as an inset (bool)
        inset_bounds (list)          : [left, bottom, width, height] of the inset

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
        palette = palette or config_dict.get("palette")

    #----------- Set up labels and colors -------------------
    ax = ax or plt.gca()
    DEFAULT_LINE_KWARGS, DEFAULT_FILL_KWARGS, DEFAULT_LABEL_KWARGS, DEFAULT_LEGEND_KWARGS, _ = set_default_kwargs()

    #------------ Results plot ----------------------------- (Merge defaults with user-provided kwargs)

    values, plot_kw, fill_kw = set_covariance_spectrum_values_and_kwargs(DEFAULT_LINE_KWARGS, DEFAULT_FILL_KWARGS,
                                                     plot_kwargs, fill_kwargs, palette,
                                                     data, data_or_surrogate = "data")
    draw_plot_covariance_spectrum(values, ax, plot_kw, fill_kw)

    #------------ Surrogate plot ---------------------------- (if present)
    if surrogate_data is not None and type(surrogate_data) == type(data):
        values_surrogate, plot_kw, fill_kw = set_covariance_spectrum_values_and_kwargs(DEFAULT_LINE_KWARGS, DEFAULT_FILL_KWARGS,
                                                        plot_kwargs, fill_kwargs, palette,
                                                        surrogate_data, data_or_surrogate = "surrogate")
        draw_plot_covariance_spectrum(values_surrogate, ax, plot_kw, fill_kw)
        
    elif surrogate_data is not None:
        raise ValueError("Surrogate data must be of the same type as the result data.")
    else:
        values_surrogate = None

    #----------- Axis scaling and styling -------------------
    apply_log_axes(ax, base=10)
    if tick_kwargs is not None:
        style_axes(ax, tick_kwargs)

    #----------- Labels and title ---------------------------
    label_kw = {**DEFAULT_LABEL_KWARGS, **(label_kwargs or {})}
    legend_kw = {**DEFAULT_LEGEND_KWARGS, **(legend_kwargs or {})}
    all_labels = labels_covariance_spectrum(values, values_surrogate)

    ax.set_xlabel(all_labels["xlabel"], **label_kw)
    ax.set_ylabel(all_labels["ylabel"], **label_kw) 
    if legend and config_dict.get("show_legend"):
        ax.legend(labels = all_labels["legend"], **legend_kw)
        
    #----------- Marchenko-Pastur Inset ---------------------
    if add_marchenko_pastur_inset:
        # Safety check to ensure the math has been done on the object
        if not hasattr(data, 'mp_sigma'):
            raise AttributeError(
                "The spectrum object lacks Marchenko-Pastur fit data."
            )

        inset_location = get_optimal_inset_loc(values['x'][-1], values['y'][-1])
        axins = inset_axes(
            ax, 
            width="35%",   # Size relative to the parent bounding box
            height="35%", 
            loc=inset_location, 
            borderpad=4.0  # Padding between the inset and the parent axes edges
            )
        dynamic_legend_kw = {"bbox_to_anchor": (1.05, 0.5)}
 
        
        # Prepare kwargs for the MP plot (assuming these functions from earlier are in your file)
        hist_kw, plot_mp_kw, vline_kw = set_marchenko_pastur_kwargs(palette,
            DEFAULT_LINE_KWARGS, DEFAULT_FILL_KWARGS, mp_hist_kwargs, mp_plot_kwargs, mp_vline_kwargs
        )
        
        # Draw the MP fit into the inset axis
        draw_plot_marchenko_pastur(data, axins, hist_kw, plot_mp_kw, vline_kw)
        
        # Style the inset (log scale, smaller fonts to prevent crowding)
        apply_log_axes(axins, base=10)
        
        # Using a slightly smaller font size for the inset to distinguish from main axes
        inset_label_kw = {**label_kw, "fontsize": label_kw.get("fontsize", 12) * 0.75}
        axins.set_xlabel("Eigenvalue", **inset_label_kw)
        axins.set_ylabel("Density", **inset_label_kw)
        axins.tick_params(axis='both', which='major', labelsize=8)
        mp_labels = labels_marchenko_pastur(data)
        axins.legend(mp_labels["legend"], fontsize=8, frameon=True, framealpha=0.9, **dynamic_legend_kw)
        