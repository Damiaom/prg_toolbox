from .plot_imports import *
def plot_constellation_steps(CG_variables, highest_level=4, neuron_tracked=0, positions=None):
    """
    Draws a frame-by-frame storyboard of the merging block variables.
    
    Parameters:
    -----------
    CG_variables : CG_variables object
        The coarse-grained object
    highest_level : int 
        depth of the tree to plot
    neuron_tracked : int 
        ID of the neuron to anchor the lineage
    positions : dict, optional {neuron_id: (x, y)}. 
        If None, random positions are generated.
    
    Returns
    ----------
        None
    """
    # 1. Safely find the final macroscopic block
    final_idx = next(i for i, clu in enumerate(CG_variables.CG_cluster_idx[highest_level]) if neuron_tracked in np.atleast_1d(clu))
    target_neurons = list(np.atleast_1d(CG_variables.CG_cluster_idx[highest_level][final_idx]))
    
    # 2. Handle positions: use provided or generate random
    if positions is None:
        np.random.seed(42) 
        positions = {n: (np.random.rand(), np.random.rand()) for n in target_neurons}
    
    # Pre-calculate all cluster centers and merge coordinates
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, highest_level + 1))
    cluster_centers = {} 
    
    for i, clu in enumerate(CG_variables.CG_cluster_idx[0]):
        neuron_id = np.atleast_1d(clu)[0]
        if neuron_id in target_neurons: 
            cluster_centers[(0, i)] = positions[neuron_id]
            
    merges = [] 
    for step in range(1, highest_level + 1):
        for c_idx, clu_neurons in enumerate(CG_variables.CG_cluster_idx[step]):
            clu_array = np.atleast_1d(clu_neurons)
            if set(clu_array).issubset(set(target_neurons)):
                cx = np.mean([positions[n][0] for n in clu_array])
                cy = np.mean([positions[n][1] for n in clu_array])
                cluster_centers[(step, c_idx)] = (cx, cy)
                
                clu_size = len(clu_array)
                rep1, rep2 = clu_array[0], clu_array[int(clu_size/2)]
                
                prev_idx1 = next(i for i, c in enumerate(CG_variables.CG_cluster_idx[step-1]) if rep1 in np.atleast_1d(c))
                prev_idx2 = next(i for i, c in enumerate(CG_variables.CG_cluster_idx[step-1]) if rep2 in np.atleast_1d(c))
                
                p1 = cluster_centers[(step-1, prev_idx1)]
                p2 = cluster_centers[(step-1, prev_idx2)]
                merges.append((step, p1, p2, (cx, cy)))

    # --- Frame-by-Frame Plotting ---
    fig, axes = plt.subplots(highest_level + 1, 1, figsize=(8, 8 * (highest_level + 1)))
    if highest_level == 0: axes = [axes]

    for current_step in range(highest_level + 1):
        ax = axes[current_step]
        
        # 1. Plot base neurons and the TRACKED neuron label
        xs = [positions[n][0] for n in target_neurons]
        ys = [positions[n][1] for n in target_neurons]
        ax.scatter(xs, ys, color=colors[0], s=30, zorder=10, alpha=0.3)
        
        # Add label to the tracked neuron
        tx, ty = positions[neuron_tracked]
        if not current_step:
            ax.text(tx, ty + 0.02, f"Neuron {neuron_tracked}", fontsize=8, ha='center', color='black', fontweight='bold')
        
        # 2. Connections and Dots
        for merge in merges:
            m_step, p1, p2, center = merge
            if m_step <= current_step:
                lw = 1.0 + (m_step * 0.5)
                alpha_line = 0.8 if m_step == current_step else 0.15
                ax.plot([p1[0], center[0]], [p1[1], center[1]], color=colors[m_step], lw=lw, zorder=m_step, alpha=alpha_line)
                ax.plot([p2[0], center[0]], [p2[1], center[1]], color=colors[m_step], lw=lw, zorder=m_step, alpha=alpha_line)

        for (node_step, _), (cx, cy) in cluster_centers.items():
            if 0 < node_step <= current_step:
                node_size = 80 * (2.0 ** node_step)
                if node_step == current_step or node_step == current_step+1:
                    alpha_dot = 0.9
                else:
                    alpha_dot = 0.2
                ax.scatter(cx, cy, color=colors[node_step], s=node_size, zorder=10+node_step, alpha=alpha_dot, edgecolor='none')

        if current_step == highest_level:
            ax.scatter(*cluster_centers[(highest_level, final_idx)], color='gold', marker='*', s=600, zorder=30, edgecolor='black')

        ax.set_title(f"Step {current_step}", fontsize=16, color=colors[current_step] if current_step > 0 else 'black')
        ax.axis('off')

    plt.suptitle(f"Coarse graining: tracking neuron {neuron_tracked}", fontsize=18, y=1)
    plt.tight_layout()
    plt.show()
