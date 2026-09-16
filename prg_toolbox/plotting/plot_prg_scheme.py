import matplotlib as mpl
from matplotlib.patches import ConnectionPatch, Rectangle

from ..coarse_graining import CGVariables
from .plot_imports import *


class ClusterNode:
    def __init__(self, level, idx_in_level, cluster_ids, timeseries):
        self.level = level
        self.idx_in_level = idx_in_level
        self.cluster_ids = cluster_ids
        self.timeseries = timeseries
        self.children = []
        self.y = None
        self.color = None


def build_tree(CG_variables, top_level, neuron_tracked):
    def _build(level, marker):
        arr = np.atleast_1d(CG_variables.CG_cluster_idx[level])
        idx_in_level = next(
            idx for idx, clu in enumerate(arr) if marker in np.atleast_1d(clu)
        )
        cluster_ids = np.atleast_1d(arr[idx_in_level])
        ts = CG_variables.CG_timeseries[level][idx_in_level]
        node = ClusterNode(level, idx_in_level, cluster_ids, ts)

        if len(cluster_ids) > 1:
            clu_size = len(cluster_ids)
            marker1 = cluster_ids[0]
            marker2 = cluster_ids[int(clu_size / 2)]
            node.children = [_build(level - 1, marker1), _build(level - 1, marker2)]
        return node

    return _build(top_level, neuron_tracked)


def collect_by_level(root):
    levels = {}

    def _collect(node):
        levels.setdefault(node.level, []).append(node)
        for c in node.children:
            _collect(c)

    _collect(root)
    return levels


def make_toy_binary_data(
    n_neurons=8, n_bins=200, base_rate=0.01, corr_strength=0.7, seed=0
):
    """
    Generates toy binary (0/1) data with a known hierarchical correlation
    structure: neurons are recursively paired (0&1, 2&3, 4&5, ...), and each
    pairing level shares a common latent drive, so pairs are strongly
    correlated, pairs-of-pairs moderately correlated, and so on.

    n_neurons should be a power of 2 for a clean binary hierarchy.
    Returns an (n_neurons, n_bins) binary array.
    """
    rng = np.random.default_rng(seed)
    assert (n_neurons & (n_neurons - 1)) == 0, "n_neurons should be a power of 2"
    n_levels = int(np.log2(n_neurons))

    # shared latent drive for every group at every level of the hierarchy
    # latents[lvl] has 2**lvl independent drives (lvl=0 -> 1 population-wide drive)
    latents = [rng.random((2**lvl, n_bins)) < base_rate for lvl in range(n_levels + 1)]

    spikes = np.zeros((n_neurons, n_bins), dtype=int)
    for neuron in range(n_neurons):
        signal = rng.random(n_bins) < base_rate  # private noise
        for lvl in range(n_levels, -1, -1):
            group_idx = neuron // (n_neurons // (2**lvl))
            mask = rng.random(n_bins) < corr_strength
            signal = signal | (latents[lvl][group_idx] & mask)
        spikes[neuron] = signal.astype(int)

    return spikes


def _get_correlation(CG_variables, node):
    """Pairwise correlation between a node's two children."""
    child1, child2 = node.children
    corr_matrix = CG_variables.CG_correlation_matrices[node.level]
    return corr_matrix[child1.idx_in_level, child2.idx_in_level]


def _collect_merges(root, CG_variables):
    """Walk the tree and return a list of (node, correlation) for every
    internal (merged) node."""
    merges = []

    def _walk(node):
        if node.children:
            merges.append((node, _get_correlation(CG_variables, node)))
            for child in node.children:
                _walk(child)

    _walk(root)
    return merges


def plot_prg_scheme(
    top_level=3,
    neuron_tracked=0,
    time_bins=100,
    spacing=0.4,
    trace_scale=1.2,
    col_width=3.8,
    row_height=0.52,
    corr_cmap="magma",
):
    """
    A leftmost panel shows the pairwise-similarity matrix behind the
    first round of pairing, with the winning pairs highlighted in their
    connector's color. Raster/bar traces at each scale are connected
    level-to-level by curves colored the same way. Bar heights are on an
    absolute scale shared across all panels, so activity visibly grows
    (rather than appearing constant) as units are pooled at coarser scales.
    """
    toy_data = make_toy_binary_data(
        n_neurons=2 ** (1 + top_level), n_bins=time_bins, corr_strength=0.7, seed=1
    )
    CG_variables = CGVariables(toy_data, rg_steps=top_level)
    root = build_tree(CG_variables, top_level, neuron_tracked)

    leaf_counter = [0]

    def _assign_y(node):
        if not node.children:
            node.y = leaf_counter[0]
            leaf_counter[0] += 1
        else:
            for c in node.children:
                _assign_y(c)
            node.y = np.mean([c.y for c in node.children])

    _assign_y(root)
    levels = collect_by_level(root)
    n_levels = top_level + 1
    n_leaves = len(levels[0])
    leaves_sorted = sorted(levels[0], key=lambda n: -n.y)

    merges = _collect_merges(root, CG_variables)
    corr_values = [c for _, c in merges]
    norm = mpl.colors.Normalize(vmin=min(corr_values), vmax=max(corr_values))
    full_cmap = mpl.colormaps[corr_cmap]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        f"{corr_cmap}_clipped", full_cmap(np.linspace(0.0, 0.82, 256))
    )

    ylim = (-0.5 * spacing, (n_leaves - 0.5) * spacing)
    global_max_count = 2**top_level
    x_pad = max(2, round(time_bins * 0.08))
    x_anchor_left = -x_pad * 0.5
    x_anchor_right = time_bins - 1 + x_pad * 0.5

    # figure = one dedicated matrix column on the left + n_levels raster columns
    fig, all_axes = plt.subplots(
        1,
        n_levels + 1,
        figsize=(col_width * n_levels + col_width * 1.25, row_height * n_leaves),
        gridspec_kw={"width_ratios": [1.25] + [1] * n_levels},
    )
    mat_ax = all_axes[0]
    axes = list(all_axes[1:])
    raster_color = "#2b2b2b"
    for level in range(n_levels):
        ax = axes[level]
        for node in levels[level]:
            ts = node.timeseries[-time_bins:]
            idx = np.nonzero(ts > 0)[0]
            heights = (ts[idx] / global_max_count) * trace_scale
            y_base = node.y * spacing
            ax.vlines(
                idx, y_base, y_base + heights, color=raster_color, lw=2.1, alpha=1.0
            )
            ax.axhline(y_base, color="grey", lw=0.8, alpha=0.4, zorder=0)

        ax.set_title(f"$C_{{size}}={2**level}$", fontsize=10)
        ax.set_xlim(-x_pad, time_bins - 1 + x_pad)
        ax.set_ylim(*ylim)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        if level == 0:
            ax.set_ylabel("unit / cluster", fontsize=9)
            for unit_number, leaf in enumerate(leaves_sorted, start=1):
                ax.text(
                    -x_pad * 0.7,
                    leaf.y * spacing,
                    str(unit_number),
                    fontsize=8,
                    fontweight="bold",
                    color="#555555",
                    ha="right",
                    va="center",
                )

    plt.subplots_adjust(wspace=0.12, top=0.68, bottom=0.04)

    for node, corr in merges:
        color = cmap(norm(corr))
        for child in node.children:
            con = ConnectionPatch(
                xyA=(x_anchor_right, child.y * spacing),
                coordsA=axes[child.level].transData,
                xyB=(x_anchor_left, node.y * spacing),
                coordsB=axes[node.level].transData,
                axesA=axes[child.level],
                axesB=axes[node.level],
                connectionstyle="arc3,rad=0.15",
                color=color,
                lw=1.2,
                alpha=0.85,
                capstyle="round",
                zorder=5,
            )
            fig.add_artist(con)
            axes[child.level].plot(
                x_anchor_right,
                child.y * spacing,
                "o",
                color=color,
                ms=3.5,
                zorder=6,
                mec="white",
                mew=0.4,
            )
            axes[node.level].plot(
                x_anchor_left,
                node.y * spacing,
                "o",
                color=color,
                ms=3.5,
                zorder=6,
                mec="white",
                mew=0.4,
            )

    # ------------------------------------------------------------------
    # Leftmost panel: pairwise-similarity matrix behind the
    # first round of pairing (C_size=1 -> C_size=2), lower triangle only.
    # The 4 winning pairs are painted in their connector's color.
    # ------------------------------------------------------------------
    first_round_merges = [(n, c) for n, c in merges if n.level == 1]
    leaf_positions = [leaf.idx_in_level for leaf in leaves_sorted]
    display_idx_of = {pos: i for i, pos in enumerate(leaf_positions)}
    base_corr_matrix = CG_variables.CG_correlation_matrices[1]
    submatrix = base_corr_matrix[np.ix_(leaf_positions, leaf_positions)]
    n_units = len(leaves_sorted)

    mat_ax.set_facecolor("white")

    for i in range(n_units):
        for j in range(n_units):
            if j >= i:
                continue
            mat_ax.text(
                j,
                i,
                f"{submatrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="#666666",
                zorder=2,
            )

    # the 4 winning pairs, painted in their connector's color
    for node, corr in first_round_merges:
        c1, c2 = node.children
        i, j = display_idx_of[c1.idx_in_level], display_idx_of[c2.idx_in_level]
        color = cmap(norm(corr))
        r, c = max(i, j), min(i, j)
        mat_ax.add_patch(
            Rectangle(
                (c - 0.5, r - 0.5), 1, 1, facecolor=color, edgecolor="none", zorder=2.5
            )
        )
        mat_ax.text(
            c,
            r,
            f"{corr:.2f}",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="white",
            zorder=3,
        )

    mat_ax.set_xlim(-0.5, n_units - 0.5)
    mat_ax.set_ylim(n_units - 0.5, -0.5)
    mat_ax.set_xticks(range(n_units))
    mat_ax.set_xticklabels(range(1, n_units + 1), fontsize=8)
    mat_ax.set_yticks(range(n_units))
    mat_ax.set_yticklabels(range(1, n_units + 1), fontsize=8)
    mat_ax.set_xticks(np.arange(-0.5, n_units, 1), minor=True)
    mat_ax.set_yticks(np.arange(-0.5, n_units, 1), minor=True)
    mat_ax.grid(which="minor", color="#dddddd", lw=0.6)
    mat_ax.tick_params(length=0)
    for spine in mat_ax.spines.values():
        spine.set_color("#aaaaaa")
        spine.set_linewidth(0.8)
    mat_ax.set_title("pairwise similarity\n(round 1)", fontsize=11)

    # pairing curves outside the matrix's left edge
    curve_x = -1.4
    for node, corr in first_round_merges:
        c1, c2 = node.children
        i, j = display_idx_of[c1.idx_in_level], display_idx_of[c2.idx_in_level]
        top_idx, bottom_idx = min(i, j), max(i, j)
        color = cmap(norm(corr))
        con = ConnectionPatch(
            xyA=(curve_x, top_idx),
            coordsA=mat_ax.transData,
            xyB=(curve_x, bottom_idx),
            coordsB=mat_ax.transData,
            connectionstyle="arc3,rad=0.4",
            color=color,
            lw=1.4,
            alpha=0.9,
            capstyle="round",
            zorder=4,
            clip_on=False,
        )
        mat_ax.add_artist(con)
        for u in (top_idx, bottom_idx):
            mat_ax.plot(
                curve_x,
                u,
                "o",
                color=color,
                ms=4,
                zorder=5,
                mec="white",
                mew=0.4,
                clip_on=False,
            )

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.7, 0.92, 0.18, 0.03])  # horizontal, top-right
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.set_label("pairwise similarity", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"PRG coarse-graining scheme: {2**top_level} units",
        x=mat_ax.get_position().x0,
        y=0.98,
        fontsize=14,
        fontweight="bold",
        ha="left",
    )

    fig.set_dpi(300)  # publication-quality default if saved without dpi=...

    # return fig
