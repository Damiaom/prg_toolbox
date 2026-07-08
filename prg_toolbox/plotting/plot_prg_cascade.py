from .plot_imports import *
from matplotlib.patches import ConnectionPatch
import matplotlib.cm as cm
from ..coarse_graining import CGVariables

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
        idx_in_level = next(idx for idx, clu in enumerate(arr)
                             if marker in np.atleast_1d(clu))
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


def assign_y_positions(root):
    leaf_counter = [0]

    def _assign(node):
        if not node.children:
            node.y = leaf_counter[0]
            leaf_counter[0] += 1
        else:
            for c in node.children:
                _assign(c)
            node.y = np.mean([c.y for c in node.children])

    _assign(root)


def count_internal_nodes(root):
    count = [0]
    def _count(node):
        if node.children:
            count[0] += 1
            for c in node.children:
                _count(c)
    _count(root)
    return count[0]


def assign_colors(root, cmap_name='magma', low=0.25, high=0.70):
    """Each merging pair (two children of the same node) shares one color,
    sampled from the colormap. Root gets a fixed dark anchor color."""
    n_internal = max(count_internal_nodes(root), 1)
    palette = cm.get_cmap(cmap_name)(np.linspace(low, high, n_internal))
    color_iter = iter(palette)

    def _assign(node, color):
        node.color = color
        if node.children:
            pair_color = next(color_iter)
            for c in node.children:
                _assign(c, pair_color)

    _assign(root, cm.get_cmap(cmap_name)(0.92))  # root/final merged trace


def collect_by_level(root):
    levels = {}
    def _collect(node):
        levels.setdefault(node.level, []).append(node)
        for c in node.children:
            _collect(c)
    _collect(root)
    return levels

def make_toy_binary_data(n_neurons=8, n_bins=200, base_rate=0.01,
                          corr_strength=0.7, seed=0):
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
    latents = [rng.random((2 ** lvl, n_bins)) < base_rate for lvl in range(n_levels + 1)]

    spikes = np.zeros((n_neurons, n_bins), dtype=int)
    for neuron in range(n_neurons):
        signal = rng.random(n_bins) < base_rate  # private noise
        for lvl in range(n_levels, -1, -1):
            group_idx = neuron // (n_neurons // (2 ** lvl))
            mask = rng.random(n_bins) < corr_strength
            signal = signal | (latents[lvl][group_idx] & mask)
        spikes[neuron] = signal.astype(int)

    return spikes

def plot_prg_cascade(top_level=4, neuron_tracked=0,
                     time_bins=100, spacing=0.55, trace_scale=0.45,
                     col_width=2.8, row_height=0.84, cmap_name='magma'):

    toy_data = make_toy_binary_data(n_neurons=2**(1+top_level), n_bins=time_bins, corr_strength=0.7, seed=1)
    CG_variables = CGVariables(toy_data, rg_steps=top_level)
    root = build_tree(CG_variables, top_level, neuron_tracked)
    assign_y_positions(root)
    assign_colors(root, cmap_name=cmap_name)
    levels = collect_by_level(root)

    n_levels = top_level + 1
    n_leaves = len(levels[0])

    fig, axes = plt.subplots(
        1, n_levels,
        figsize=(col_width * n_levels, row_height * n_leaves),
        sharey=False
    )
    if n_levels == 1:
        axes = [axes]

    for level in range(n_levels):
        ax = axes[level]
        for node in levels[level]:
            ts = node.timeseries[-time_bins:]
            ts_norm = ts - ts.mean()
            span = np.ptp(ts_norm)
            if span > 0:
                ts_norm = ts_norm / span
            y_trace = node.y * spacing + ts_norm * trace_scale
            ax.plot(y_trace, lw=1.3, color=node.color, solid_capstyle='round')
            ax.axhline(node.y * spacing, color='grey', lw=0.3, alpha=0.25, zorder=0)

        ax.set_title(f'step {level}\n(n={len(levels[level])})', fontsize=10, fontweight='bold')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        if level == 0:
            ax.set_ylabel('neuron / cluster', fontsize=9)

    fig.suptitle(f"Coarse graining timeseries: neuron {neuron_tracked}", y=1.1, fontsize=20)
    plt.subplots_adjust(wspace=0.05)

    # --- curved, colored connectors between levels ---
    for level in range(1, n_levels):
        for node in levels[level]:
            for child in node.children:
                con = ConnectionPatch(
                    xyA=(time_bins - 1, child.y * spacing), coordsA=axes[child.level].transData,
                    xyB=(0, node.y * spacing), coordsB=axes[node.level].transData,
                    axesA=axes[child.level], axesB=axes[node.level],
                    connectionstyle="arc3,rad=0.15",
                    color=child.color, lw=1.6, alpha=0.85,
                    capstyle='round', zorder=5
                )
                fig.add_artist(con)
                # small marker dots at the junction points
                axes[child.level].plot(time_bins - 1, child.y * spacing, 'o',
                                        color=child.color, ms=4, zorder=6,
                                        markeredgecolor='white', markeredgewidth=0.5)
                axes[node.level].plot(0, node.y * spacing, 'o',
                                       color=node.color, ms=4, zorder=6,
                                       markeredgecolor='white', markeredgewidth=0.5)

