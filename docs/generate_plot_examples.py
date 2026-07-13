"""Generates the static PNGs used in docs/plotting_examples.md.

Run manually (not part of the test suite / CI) whenever the plotting API
or these example figures need to be regenerated:

    python docs/generate_plot_examples.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import prg_toolbox as prg
from prg_toolbox import CGVariables, mean_variance, activity_distribution
from prg_toolbox import plot
from prg_toolbox.analysis_tools import binarize_data, pick_random_sample, shuffle_isi

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(DOCS_DIR, "assets", "plotting_examples")
os.makedirs(OUT_DIR, exist_ok=True)

# One of the real V4 cortex spike recordings bundled as example data.
EXAMPLE_SPIKE_FILE = os.path.join(
    DOCS_DIR, "..", "example_notebooks", "example_data_directory", "spikes-V4-5E.gdf"
)

DPI = 150
FIGSIZE = (6, 4.5)

# Colors sampled from the 'magma' colormap, used to theme every custom example.
MAGMA_DATA = "#782281"
MAGMA_SURROGATE = "#feb77e"
MAGMA_REFERENCE = "#3f3f3f"


def build_data():
    """
    Real data ('data'): 64 randomly-sampled units from a V4 cortex spike
    recording. 'surrogate' is its ISI-shuffled null model, which destroys
    temporal correlations while preserving each unit's firing rate.
    """
    params = prg.config.AnalysisParams()
    params.loading.data_format = "tabular"
    timestamps = prg.tools.load_data(EXAMPLE_SPIKE_FILE, user_params=params)

    subsample = pick_random_sample(timestamps, sample_size=64, data_format="tabular", random_seed=7)
    data_binary = binarize_data(subsample, data_format="tabular", binsize_ms=5.0)
    surrogate_binary = shuffle_isi(data_binary, data_format="timeseries", random_seed=7)
    return data_binary, surrogate_binary


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    data_binary, surrogate_binary = build_data()

    cg_data = CGVariables(data_binary, cluster_method="pearson", rg_steps=5)
    cg_surr = CGVariables(surrogate_binary, cluster_method="pearson", rg_steps=5)

    mv_data = mean_variance(cg_data)
    mv_surr = mean_variance(cg_surr)
    ad_data = activity_distribution(cg_data)
    ad_surr = activity_distribution(cg_surr)

    # --- mean_variance: default style ---------------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot.plot_mean_variance(mv_data, surrogate_data=mv_surr, ax=ax)
    save(fig, "mean_variance_default.png")

    # --- mean_variance: custom style -----------------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot.plot_mean_variance(
        mv_data,
        surrogate_data=mv_surr,
        ax=ax,
        colors={"data": MAGMA_DATA, "surrogate": MAGMA_SURROGATE, "reference": MAGMA_REFERENCE},
        plot_kwargs={"marker": "*", "markersize": 22, "markeredgewidth": 1.0, "markeredgecolor": "black"},
        fill_kwargs={"alpha": 0.3, "hatch": "//"},
        label_kwargs={"fontsize": 15, "fontweight": "bold"},
        legend_kwargs={"fontsize": 11, "loc": "upper left", "frameon": False},
        tick_kwargs={"labelsize": 11, "direction": "in", "length": 6},
    )
    save(fig, "mean_variance_custom.png")

    # --- activity_distribution: default style ---------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot.plot_activity_distribution(ad_data, surrogate_data=ad_surr, ax=ax)
    save(fig, "activity_distribution_default.png")

    # --- activity_distribution: custom style -----------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot.plot_activity_distribution(
        ad_data,
        surrogate_data=ad_surr,
        ax=ax,
        palette={"data": "magma", "surrogate": "inferno"},
        plot_kwargs={"marker": "*", "markersize": 10, "linestyle": "-", "linewidth": 1.5},
        fill_kwargs={"alpha": 0.25},
        label_kwargs={"fontsize": 15, "fontweight": "bold"},
        legend_kwargs={"fontsize": 11, "loc": "lower center", "frameon": False},
        tick_kwargs={"labelsize": 11, "direction": "in", "length": 6},
    )
    save(fig, "activity_distribution_custom.png")


if __name__ == "__main__":
    main()
