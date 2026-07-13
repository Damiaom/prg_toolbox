# Plotting library (`prg_toolbox.plotting`)

The visualization module provides an interface for plotting the statistical observables calculated during the Phenomenological Renormalization Group (PRG) analysis. The plotting functions (e.g., `plot_mean_variance`, `plot_covariance_spectrum`, `plot_activity_distribution`) share a common structure and accept essentially the same parameters. One exception is the color parameter `colors` for single line plots and `palette` for multiple curve plots.

See [Styling your plots](plotting_examples.md) for worked, rendered examples of both.

**Matplotlib Styling and Configurations**

The functions accept Matplotlib keyword arguments to modify the layout, colors, and line styles (see list in the example below). Users can input these configurations using two methods:

* **Global Configuration:** Passing the `AnalysisParams` dataclass (or any `AnalysisParams.plot_style` / `PlotSyleConfig` object directly) directly into the `style_config` parameter. 
* **Explicit Overrides:** Passing dictionaries to specific arguments, such as `plot_kwargs`, `fill_kwargs`, `label_kwargs`, `legend_kwargs`, or `tick_kwargs`. If a user provides both a global `style_config` and an explicit dictionary for the same parameter, the explicit dictionary takes precedence.

**Surrogate Data Comparisons**

Every plotting function accepts an optional `surrogate_data` parameter. When a secondary dataset (such as the output of a shuffled null model) is passed to this argument alongside the primary `data`, the function renders both results on the same canvas. The module visually distinguishes the surrogate data to facilitate baseline comparisons.

**Axes Management and Multi-Plotting**

All functions accept an optional `ax` parameter, expecting a Matplotlib `Axes` object. If an existing `ax` is provided, the function renders the data directly onto that canvas. Users can plot multiple datasets on the same figure by passing the same `ax` variable to multiple plotting function calls.

**Available Plotting Functions**

The module includes the following specialized functions for visualizing PRG observables:

* `plot_mean_variance`
* `plot_log_silence_probability`
* `plot_max_covariance_eigenvalue`
* `plot_covariance_spectrum`
* `plot_autocorrelation_function`
* `plot_decay_time`
* `plot_activity_distribution`

Below is an example of the standard input arguments shared across the plotting module:

```python
def plot_observable_name(
    data,
    surrogate_data=None,
    ax=None,
    style_config=None,
    plot_kwargs=None,
    fill_kwargs=None,
    label_kwargs=None,
    legend_kwargs=None,
    tick_kwargs=None,
    colors=None,
    legend=True
):
    """
    Plots mean-variance scaling with optional surrogate and reference.

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
        colors (str, dict, or None)  : Custom color scheme (for multi-curve plots, `palette`)
        legend (bool)                : Whether to display legend

        # Exclusive to plot_covariance_spectrum
        add_marchenko_pastur_inset   : Whether to plot the Marchenko-Pastur fit as an inset (bool)
        inset_bounds (list)          : [left, bottom, width, height] of the inset
    Returns:
        None
    """
```


