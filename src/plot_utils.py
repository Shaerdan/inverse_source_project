"""
Plotting utilities with ground truth display.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


def add_sources_to_plot(ax, sources_true, show_legend=True, show_text=False):
    """
    Add true source markers to a plot with proper legend.
    
    Parameters
    ----------
    ax : matplotlib axis
    sources_true : list of ((x, y), intensity) tuples
    show_legend : bool
        Show legend with intensities
    show_text : bool
        Annotate each source with its intensity value
    """
    pos_sources = [(x, y, q) for (x, y), q in sources_true if q > 0]
    neg_sources = [(x, y, q) for (x, y), q in sources_true if q < 0]
    
    # Plot positive sources
    if pos_sources:
        xs, ys, qs = zip(*pos_sources)
        ax.plot(xs, ys, 'o', color='black', markersize=15,
                markerfacecolor='none', markeredgewidth=2.5,
                label=f'True + (q={qs[0]:+.1f})')
        if show_text:
            for x, y, q in pos_sources:
                ax.annotate(f'{q:+.1f}', (x, y), textcoords='offset points',
                           xytext=(8, 8), fontsize=9, fontweight='bold')
    
    # Plot negative sources
    if neg_sources:
        xs, ys, qs = zip(*neg_sources)
        ax.plot(xs, ys, 's', color='black', markersize=12,
                markerfacecolor='none', markeredgewidth=2.5,
                label=f'True − (q={qs[0]:+.1f})')
        if show_text:
            for x, y, q in neg_sources:
                ax.annotate(f'{q:+.1f}', (x, y), textcoords='offset points',
                           xytext=(8, -12), fontsize=9, fontweight='bold')
    
    if show_legend:
        ax.legend(loc='upper right', fontsize=9)


def add_ground_truth_text(fig, sources_true):
    """Add a text box with ground truth info to the figure."""
    text_lines = ["Ground Truth:"]
    for i, ((x, y), q) in enumerate(sources_true):
        text_lines.append(f"  ({x:+.1f}, {y:+.1f}): q={q:+.1f}")
    
    text = "\n".join(text_lines)
    fig.text(0.02, 0.98, text, transform=fig.transFigure,
             fontsize=9, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def plot_source_comparison(q_recovered, interior_coords, mesh, sources_true,
                           u_measured=None, boundary_angles=None, G=None,
                           title='Recovered Sources', save_path=None):
    """
    Standard plot for recovered vs true sources with ground truth display.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get mesh triangulation
    coords = mesh.geometry.x
    mesh.topology.create_connectivity(2, 0)
    cells = mesh.topology.connectivity(2, 0)
    triangles = np.array([cells.links(i) for i in range(cells.num_nodes)])
    tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
    
    # Left plot: source distribution
    ax = axes[0]
    ax.triplot(tri, 'k-', linewidth=0.3, alpha=0.3)
    
    vmax = max(np.max(np.abs(q_recovered)), 0.01)
    scatter = ax.scatter(
        interior_coords[:, 0], interior_coords[:, 1],
        c=q_recovered, cmap='RdBu_r', s=40, vmin=-vmax, vmax=vmax
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f'Recovered q (max={vmax:.3f})')
    
    add_sources_to_plot(ax, sources_true, show_legend=True, show_text=True)
    
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    
    # Right plot: boundary fit
    ax = axes[1]
    if u_measured is not None and boundary_angles is not None:
        ax.plot(boundary_angles, u_measured, 'b-', linewidth=2, label='Measured')
        if G is not None:
            ax.plot(boundary_angles, G @ q_recovered, 'r--', linewidth=2, label='Reconstructed')
            residual = np.linalg.norm(G @ q_recovered - u_measured)
            ax.set_title(f'Boundary Fit (residual={residual:.4f})')
        ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Angle (radians)')
    ax.set_ylabel('Value')
    
    # Add ground truth text box
    add_ground_truth_text(fig, sources_true)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_regularization_comparison(results_dict, solver, u_measured, sources_true,
                                   title='Regularization Comparison', save_path=None):
    """
    Compare multiple regularization results with ground truth.
    
    Parameters
    ----------
    results_dict : dict
        {'method_name': {'q': array, 'alpha': float}, ...}
    """
    n_methods = len(results_dict)
    fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))
    
    if n_methods == 1:
        axes = axes.reshape(2, 1)
    
    # Get mesh triangulation
    coords = solver.mesh.geometry.x
    solver.mesh.topology.create_connectivity(2, 0)
    cells = solver.mesh.topology.connectivity(2, 0)
    triangles = np.array([cells.links(i) for i in range(cells.num_nodes)])
    tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
    
    for col, (name, data) in enumerate(results_dict.items()):
        q = data['q']
        alpha = data.get('alpha', None)
        
        # Top row: source distribution
        ax = axes[0, col]
        ax.triplot(tri, 'k-', linewidth=0.2, alpha=0.3)
        
        vmax = max(np.max(np.abs(q)), 0.01)
        scatter = ax.scatter(
            solver.interior_coords[:, 0], solver.interior_coords[:, 1],
            c=q, cmap='RdBu_r', s=30, vmin=-vmax, vmax=vmax
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(f'q (max={vmax:.3f})')
        
        add_sources_to_plot(ax, sources_true, show_legend=(col == 0), show_text=False)
        
        title_str = f'{name}'
        if alpha is not None:
            title_str += f' (α={alpha:.1e})'
        ax.set_title(title_str)
        ax.set_aspect('equal')
        
        # Bottom row: boundary fit
        ax = axes[1, col]
        ax.plot(solver.boundary_angles, u_measured, 'b-', linewidth=2, label='Measured')
        ax.plot(solver.boundary_angles, solver.G @ q, 'r--', linewidth=2, label='Fit')
        if col == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        residual = np.linalg.norm(solver.G @ q - u_measured)
        ax.set_title(f'Residual = {residual:.4f}')
    
    # Add ground truth text box
    add_ground_truth_text(fig, sources_true)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig
