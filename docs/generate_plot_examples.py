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
from prg_toolbox import plot
from prg_toolbox.analysis_tools import pick_random_sample, shuffle_isi

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


def build_results():
    """
    Real data ('data'): 3 independent random 64-unit subsamples of a V4
    cortex spike recording, averaged via run_PRG (nsamples=3) so the
    resulting observables carry a genuine std_across_windows spread --
    that's what fill_kwargs renders. 'surrogate' is the same pipeline run
    on the ISI-shuffled recording, a null model that destroys temporal
    correlations while preserving each unit's firing rate.

    The recording has ~9500 units, and shuffle_isi's per-unit scan is
    O(units x total_spikes) -- shuffling the full recording takes minutes.
    We only ever need 64 units per sample, so a 500-unit pool (picked once)
    is plenty of room for run_PRG's own nsamples=3 draws to differ, while
    keeping shuffle_isi (and everything downstream) fast.
    """
    params = prg.config.AnalysisParams()
    params.loading.data_format = "tabular"
    params.time_slicing.binary_binsize_ms = 5.0
    params.rg_steps = 5
    params.cluster_method = "pearson"
    params.subsampling.samplesize = 64
    params.subsampling.nsamples = 3
    params.subsampling.random_seed = 7
    params.observables = [prg.mean_variance, prg.activity_distribution]

    timestamps = prg.tools.load_data(EXAMPLE_SPIKE_FILE, user_params=params)
    pool = pick_random_sample(timestamps, sample_size=500, data_format="tabular", random_seed=7)
    surrogate_pool = shuffle_isi(pool, data_format="tabular", random_seed=7)

    data_result = prg.run_PRG(pool, user_params=params)
    surrogate_result = prg.run_PRG(surrogate_pool, user_params=params)
    return data_result, surrogate_result


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    data_result, surrogate_result = build_results()

    mv_data = data_result["mean_variance"]
    mv_surr = surrogate_result["mean_variance"]
    ad_data = data_result["activity_distribution"]
    ad_surr = surrogate_result["activity_distribution"]

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
        legend_kwargs={"fontsize": 11, "loc": "upper right", "frameon": False},
        tick_kwargs={"labelsize": 11, "direction": "in", "length": 6},
    )
    save(fig, "activity_distribution_custom.png")


if __name__ == "__main__":
    main()
