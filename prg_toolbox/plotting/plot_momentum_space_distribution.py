from .plot_imports import *

# Momentum-space distributions are densities on a fine, continuous grid of
# bins, so they are drawn as lines rather than the markers used by default
# for the discrete observables.
CONTINUOUS_LINE_KWARGS = {"linestyle": "-", "marker": "None"}


def labels_momentum_space_distribution(data, surrogate_data=None):
    """
    Creates axis labels and legend entries for momentum-space distribution plots.

    Parameters
    ----------
        data (dict)                  : Dictionary with momentum-space distribution
        surrogate_data (dict or None): Optional surrogate data

    Returns
    ----------
        label_dict (dict): Contains xlabel, ylabel, and legend entries
    """
    result_legend = r"$(K = N/%d)$" % (2 ** (len(data["y"]) - 1))

    if surrogate_data is not None:
        surrogate_legend = "Surrogate"
    else:
        surrogate_legend = None

    legend = (
        [result_legend, surrogate_legend, "Gaussian"]
        if surrogate_legend
        else [result_legend, "Gaussian"]
    )
    label_dict = {
        "xlabel": r"Momentum-space variable $\phi$",
        "ylabel": "Probability Density",
        "legend": legend,
    }
    return label_dict


def extract_momentum_space_distribution_from_object(data_object):
    """
    Extracts momentum-space distribution data from an object.

    Parameters
    ----------
        data_object (object): Object with bin_centers, avg_across_windows and std_across_windows

    Returns
    ----------
        data (dict): Dictionary with x, y, error bounds, and metadata
    """
    y = data_object.avg_across_windows
    y_low = [y[k] - data_object.std_across_windows[k] for k in range(len(y))]
    y_high = [y[k] + data_object.std_across_windows[k] for k in range(len(y))]
    x = [np.asarray(data_object.bin_centers) for _ in range(len(y))]
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


def extract_momentum_space_distribution_from_dictionary(data_dict):
    """
    Extracts momentum-space distribution data from a dictionary.

    Parameters
    ----------
        data_dict (dict): Dictionary with 'bin_centers', 'avg_across_windows' and 'std_across_windows'

    Returns
    ----------
        data (dict): Dictionary with x, y, error bounds, and metadata
    """
    y = data_dict["avg_across_windows"]
    y_low = [y[k] - data_dict["std_across_windows"][k] for k in range(len(y))]
    y_high = [y[k] + data_dict["std_across_windows"][k] for k in range(len(y))]
    x = [np.asarray(data_dict["bin_centers"]) for _ in range(len(y))]
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


def set_momentum_space_distribution_values_and_kwargs(
    DEFAULT_LINE_KWARGS,
    DEFAULT_FILL_KWARGS,
    plot_kwargs,
    fill_kwargs,
    palette,
    data,
    data_or_surrogate="data",
):
    """
    Prepares momentum-space distribution values and per-curve plotting kwargs.

    Parameters
    ----------
        plot_kwargs (dict or None)   : Line plot kwargs
        fill_kwargs (dict or None)   : Fill kwargs
        palette (str, dict, or None) : Overrides the colormap(s) coloring
                                        successive RG-step curves. A plain
                                        colormap name sets only the 'data'
                                        colormap; a dict may set either or
                                        both of 'data', 'surrogate' (unset
                                        keys keep their defaults).
        data (object or dict)        : Main data
        data_or_surrogate (str)      : Identifier for data type

    Returns
    ----------
        values (dict)  : Extracted plotting data
        plot_kw (list) : Line kwargs, one dict per RG step
        fill_kw (list) : Fill kwargs, one dict per RG step
    """
    data_type_name = type(data).__name__
    if data_type_name == "momentum_space_distribution":
        values = extract_momentum_space_distribution_from_object(data)
    elif isinstance(data, dict):
        values = extract_momentum_space_distribution_from_dictionary(data)
    else:
        raise ValueError(
            "Data must be either a momentum_space_distribution object or a dictionary."
        )

    number_of_colors = len(values["y"])
    colors_by_iteration = set_colors_from_palette(
        number_of_colors, palette, data_or_surrogate=data_or_surrogate
    )
    plot_kw = [
        {
            **DEFAULT_LINE_KWARGS,
            **CONTINUOUS_LINE_KWARGS,
            "color": colors_by_iteration[k],
            **(plot_kwargs or {}),
        }
        for k in range(len(values["y"]))
    ]
    fill_kw = [
        {**DEFAULT_FILL_KWARGS, "color": colors_by_iteration[k], **(fill_kwargs or {})}
        for k in range(len(values["y"]))
    ]
    # Fade the curves with many modes: each curve's own alpha (1 if unset)
    # is scaled by k / n_curves, leaving the last curve untouched.
    n_curves = len(values["y"])
    for k in range(n_curves - 1):
        plot_kw[k]["alpha"] = plot_kw[k].get("alpha", 1.0) * k / n_curves
        fill_kw[k]["alpha"] = fill_kw[k].get("alpha", 1.0) * k / n_curves

    return values, plot_kw, fill_kw


def draw_plot_momentum_space_distribution(values, ax, plot_kw=None, fill_kw=None):
    """
    Draws momentum-space distribution curves.

    Empty bins are left out (masked) so that the curves are not broken on
    the logarithmic y axis.

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
        y = np.asarray(values["y"][k], dtype=float)
        nonzero = y > 0

        # Main line
        if k != len(values["y"]) - 1:
            ax.plot(
                values["x"][k][nonzero],
                y[nonzero],
                label="_nolegend_",
                **plot_kw[k],
            )
        else:
            ax.plot(values["x"][k][nonzero], y[nonzero], **plot_kw[k])

        # Std error band, only where its lower edge is positive: a band
        # reaching zero has no lower end on the logarithmic y axis.
        if values["y_low"][k] is not None and values["y_high"][k] is not None:
            y_low = np.asarray(values["y_low"][k], dtype=float)
            ax.fill_between(
                values["x"][k],
                y_low,
                np.asarray(values["y_high"][k], dtype=float),
                where=nonzero & (y_low > 0),
                label="_nolegend_",
                **fill_kw[k],
            )


def draw_reference_standard_normal(values, ax):
    """
    Draws the standard normal density as the Gaussian reference.

    Momentum-space variables have zero mean and unit variance by
    construction, so the trivial (Gaussian) fixed point is exactly the
    standard normal distribution and needs no fit.
    """
    x = np.linspace(np.min(values["x"][-1]), np.max(values["x"][-1]), 500)
    gaussian = np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)
    ax.plot(x, gaussian, linestyle="--", alpha=0.6, lw=3, color="grey")


def find_bottom(y_values):
    flat = np.array([x for sub in y_values for x in sub])
    positive_y = flat[flat > 0]
    return np.min(positive_y)


def plot_momentum_space_distribution(
    data,
    surrogate_data=None,
    ax=None,
    style_config=None,
    plot_kwargs=None,
    fill_kwargs=None,
    label_kwargs=None,
    legend_kwargs=None,
    tick_kwargs=None,
    palette=None,
    legend=None,
):
    """
    Plots momentum-space distributions with multiple curves and optional surrogate.

    One curve is drawn per coarse-graining step, from K = N (all principal
    components kept; lighter, transparent) to K = N / 2^k_max (opaque), along
    with the standard normal distribution as the Gaussian reference.

    Parameters
    ----------
        data (object or dict)        : Main data (momentum_space_distribution
                                        object, or a dictionary with
                                        'bin_centers', 'avg_across_windows'
                                        and 'std_across_windows')
        surrogate_data (same type)   : Optional surrogate data
        ax (matplotlib axis or None) : Axis to plot on
        style_config (object or None): Global AnalysisParams or PlotStyleConfig dataclass
        plot_kwargs (dict or None)   : Line plot kwargs
        fill_kwargs (dict or None)   : Fill kwargs
        label_kwargs (dict or None)  : Axis label kwargs
        legend_kwargs (dict or None) : Legend kwargs
        tick_kwargs (dict or None)   : Tick styling kwargs
        palette (str, dict, or None) : Overrides the colormap(s) coloring
                                        successive RG-step curves. A plain
                                        colormap name sets only the 'data'
                                        colormap; a dict may set either or
                                        both of 'data', 'surrogate' (unset
                                        keys keep their defaults).
        legend (bool or None)        : Whether to display legend. If None
                                        (default), follows `show_legend`
                                        of style_config when one is given,
                                        and is shown otherwise.

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
        # An explicit legend argument wins over the configuration object
        if legend is None:
            legend = config_dict.get("show_legend")

    if legend is None:
        legend = True

    # ----------- Set up labels and colors -------------------
    ax = ax or plt.gca()
    (
        DEFAULT_LINE_KWARGS,
        DEFAULT_FILL_KWARGS,
        DEFAULT_LABEL_KWARGS,
        DEFAULT_LEGEND_KWARGS,
        _,
    ) = set_default_kwargs()

    # ------------ Results plot ----------------------------- (Merge defaults with user-provided kwargs)

    values, plot_kw, fill_kw = set_momentum_space_distribution_values_and_kwargs(
        DEFAULT_LINE_KWARGS,
        DEFAULT_FILL_KWARGS,
        plot_kwargs,
        fill_kwargs,
        palette,
        data,
        data_or_surrogate="data",
    )
    draw_plot_momentum_space_distribution(values, ax, plot_kw, fill_kw)

    # ------------ Surrogate plot ---------------------------- (if present)
    if surrogate_data is not None and type(surrogate_data) == type(data):
        values_surrogate, plot_kw, fill_kw = (
            set_momentum_space_distribution_values_and_kwargs(
                DEFAULT_LINE_KWARGS,
                DEFAULT_FILL_KWARGS,
                plot_kwargs,
                fill_kwargs,
                palette,
                surrogate_data,
                data_or_surrogate="surrogate",
            )
        )
        draw_plot_momentum_space_distribution(values_surrogate, ax, plot_kw, fill_kw)
    elif surrogate_data is not None:
        raise ValueError("Surrogate data must be of the same type as the result data.")
    else:
        values_surrogate = None

    # ------------ Reference plot ----------------------------
    draw_reference_standard_normal(values, ax)

    # ----------- Axis scaling and styling -------------------
    all_curves = list(values["y"])
    if values_surrogate is not None:
        all_curves += list(values_surrogate["y"])
    y_min = find_bottom(all_curves)
    ax.set_yscale("log")
    if tick_kwargs is not None:
        style_axes(ax, tick_kwargs)
    ax.set_ylim(bottom=y_min)
    # The K = N curve (z-scored raw data) is drawn fully transparent by the
    # alpha ramp and, for binary data, is a few narrow spikes; leave it out
    # of the upper limit so it does not squash the visible curves.
    visible_curves = values["y"][1:] or values["y"]
    visible_x = values["x"][1:] or values["x"]
    if values_surrogate is not None:
        visible_curves = list(visible_curves) + list(values_surrogate["y"][1:])
        visible_x = list(visible_x) + list(values_surrogate["x"][1:])
    ax.set_ylim(top=2 * max(np.max(y) for y in visible_curves))
    # Likewise, wide user-chosen bins can leave long empty stretches; limit
    # the x axis to where the visible curves have mass.
    occupied_x = np.concatenate(
        [x[np.asarray(y) > 0] for x, y in zip(visible_x, visible_curves, strict=True)]
    )
    margin = 0.02 * (occupied_x.max() - occupied_x.min())
    ax.set_xlim(occupied_x.min() - margin, occupied_x.max() + margin)

    # ----------- Labels and title ---------------------------
    label_kw = {**DEFAULT_LABEL_KWARGS, **(label_kwargs or {})}
    legend_kw = {**DEFAULT_LEGEND_KWARGS, **(legend_kwargs or {})}
    all_labels = labels_momentum_space_distribution(values, values_surrogate)
    ax.set_xlabel(all_labels["xlabel"], **label_kw)
    ax.set_ylabel(all_labels["ylabel"], **label_kw)
    if legend:
        ax.legend(labels=all_labels["legend"], **legend_kw)
