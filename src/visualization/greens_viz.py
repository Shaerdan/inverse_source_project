"""
Green's Function Visualizations
===============================

Visualizations for understanding the Green's function and Green's matrix
used in inverse source problems.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm
from typing import Tuple, Optional

from .config import COLORS, COLORMAPS, FIGSIZE
from .utils import get_domain_boundary, domain_mask, get_bounding_box, add_domain_boundary


def plot_greens_function(source_location: Tuple[float, float],
                         domain_type: str = 'disk',
                         domain_params: dict = None,
                         resolution: int = 100,
                         clip_singularity: float = None,
                         title: str = None,
                         ax: plt.Axes = None) -> plt.Axes:
    """
    Heatmap of Neumann Green's function G(x, ξ) for fixed source ξ.

    Parameters
    ----------
    source_location : tuple (x, y)
        Source position ξ
    domain_type : str
        Domain type ('disk', 'ellipse', etc.)
    domain_params : dict
        Domain parameters
    resolution : int
        Grid resolution
    clip_singularity : float, optional
        Clip values within this distance of source
    title : str
        Plot title
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    ax : plt.Axes
        The matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE['square'])

    if domain_params is None:
        domain_params = {}

    # Get domain boundary
    boundary = get_domain_boundary(domain_type, domain_params)
    x_min, x_max, y_min, y_max = get_bounding_box(boundary, margin=0.05)

    # Create grid
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(x, y)

    # Evaluate Green's function
    xi_x, xi_y = source_location
    G = np.zeros_like(xx)

    try:
        from ..analytical_solver import greens_function_disk_neumann
    except ImportError:
        try:
            from analytical_solver import greens_function_disk_neumann
        except ImportError:
            # Fallback: simple log approximation
            def greens_function_disk_neumann(pts, xi):
                dx = pts[:, 0] - xi[0]
                dy = pts[:, 1] - xi[1]
                r = np.sqrt(dx**2 + dy**2)
                r = np.maximum(r, 1e-10)
                return -1/(2*np.pi) * np.log(r)

    # Evaluate at grid points
    points = np.column_stack([xx.ravel(), yy.ravel()])
    xi = np.array([xi_x, xi_y])
    G_flat = greens_function_disk_neumann(points, xi)
    G = G_flat.reshape(xx.shape)

    # Mask outside domain
    mask = domain_mask(xx, yy, domain_type, domain_params)
    G = np.ma.masked_where(~mask, G)

    # Clip near singularity
    if clip_singularity is not None:
        dist_to_source = np.sqrt((xx - xi_x)**2 + (yy - xi_y)**2)
        G = np.ma.masked_where(dist_to_source < clip_singularity, G)

    # Determine colormap range (symmetric around zero for diverging)
    vmax = np.nanmax(np.abs(G))
    vmin = -vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Plot heatmap
    im = ax.pcolormesh(xx, yy, G, cmap=COLORMAPS['greens'], norm=norm, shading='auto')

    # Contour lines
    ax.contour(xx, yy, G, levels=15, colors='k', linewidths=0.3, alpha=0.5)

    # Mark source
    ax.scatter(xi_x, xi_y, c='gold', s=200, marker='*', edgecolors='black',
              linewidths=2, zorder=10, label=f'Source ξ=({xi_x:.2f}, {xi_y:.2f})')

    # Domain boundary
    add_domain_boundary(ax, boundary, fill=False, color=COLORS['boundary'], linewidth=2)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('G(x, ξ)')

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right', fontsize=8)

    if title is None:
        title = f'Neumann Green\'s Function (ξ = ({xi_x:.2f}, {xi_y:.2f}))'
    ax.set_title(title)

    return ax


def plot_greens_boundary(source_location: Tuple[float, float],
                         domain_type: str = 'disk',
                         domain_params: dict = None,
                         n_boundary: int = 100,
                         title: str = None,
                         ax: plt.Axes = None) -> plt.Axes:
    """
    Plot G(x_boundary, ξ) as function of boundary angle.

    This shows one column of the Green's matrix.

    Parameters
    ----------
    source_location : tuple
        Source position
    domain_type : str
        Domain type
    domain_params : dict
        Domain parameters
    n_boundary : int
        Number of boundary points
    title : str
        Plot title
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    ax : plt.Axes
        The matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE['wide'])

    if domain_params is None:
        domain_params = {}

    # Get boundary points
    boundary = get_domain_boundary(domain_type, domain_params, n_points=n_boundary)

    # Compute angles
    theta = np.arctan2(boundary[:, 1], boundary[:, 0])

    # Evaluate Green's function on boundary
    xi_x, xi_y = source_location

    try:
        from ..analytical_solver import greens_function_disk_neumann
    except ImportError:
        try:
            from analytical_solver import greens_function_disk_neumann
        except ImportError:
            def greens_function_disk_neumann(pts, xi):
                dx = pts[:, 0] - xi[0]
                dy = pts[:, 1] - xi[1]
                r = np.sqrt(dx**2 + dy**2)
                r = np.maximum(r, 1e-10)
                return -1/(2*np.pi) * np.log(r)

    xi = np.array([xi_x, xi_y])
    G_boundary = greens_function_disk_neumann(boundary, xi)

    # Sort by angle
    sort_idx = np.argsort(theta)
    theta_sorted = theta[sort_idx]
    G_sorted = G_boundary[sort_idx]

    # Normalize theta to [0, 2π]
    theta_sorted = np.mod(theta_sorted, 2*np.pi)

    # Plot
    ax.plot(theta_sorted, G_sorted, color=COLORS['measured'], linewidth=2)

    # Mark source angle
    source_angle = np.arctan2(xi_y, xi_x)
    if source_angle < 0:
        source_angle += 2*np.pi
    ax.axvline(source_angle, color=COLORS['source_positive'], linestyle='--',
              linewidth=1.5, label=f'Source angle: {source_angle:.2f}')

    # Peak location
    peak_idx = np.argmax(G_sorted)
    peak_angle = theta_sorted[peak_idx]
    ax.scatter([peak_angle], [G_sorted[peak_idx]], c=COLORS['highlight'],
              s=100, zorder=5, label=f'Peak: θ = {peak_angle:.2f}')

    from .utils import format_axes_pi
    ax.set_xlim(0, 2*np.pi)
    format_axes_pi(ax, 'x')
    ax.set_xlabel('θ (boundary angle)')
    ax.set_ylabel('G(x_boundary, ξ)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    if title is None:
        title = f'Green\'s Function on Boundary (ξ = ({xi_x:.2f}, {xi_y:.2f}))'
    ax.set_title(title)

    return ax


def plot_greens_matrix(G: np.ndarray,
                       sensor_angles: np.ndarray = None,
                       grid_positions: np.ndarray = None,
                       title: str = "Green's Matrix Analysis") -> plt.Figure:
    """
    Analyze Green's matrix structure.

    Creates 3-panel figure:
    - Matrix heatmap
    - Singular value spectrum
    - Column coherence histogram

    Parameters
    ----------
    G : ndarray, shape (n_sensors, n_grid)
        Green's matrix
    sensor_angles : ndarray, optional
        Angles of sensor locations
    grid_positions : ndarray, optional
        Positions of grid points
    title : str
        Figure title

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    n_sensors, n_grid = G.shape

    # Panel 1: Matrix heatmap
    ax1 = axes[0]
    vmax = np.max(np.abs(G))
    im = ax1.imshow(np.abs(G), aspect='auto', cmap='viridis',
                   vmin=0, vmax=vmax)
    ax1.set_xlabel('Grid point index')
    ax1.set_ylabel('Sensor index')
    ax1.set_title(f'|G| ({n_sensors} × {n_grid})')
    plt.colorbar(im, ax=ax1)

    # Panel 2: Singular value spectrum
    ax2 = axes[1]
    U, s, Vh = np.linalg.svd(G, full_matrices=False)

    ax2.semilogy(s, 'o-', color=COLORS['trajectory'], markersize=4)
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Singular value (log scale)')
    ax2.set_title('Singular Value Spectrum')
    ax2.grid(True, alpha=0.3)

    # Condition number
    cond = s[0] / s[-1] if s[-1] > 0 else np.inf
    ax2.annotate(f'Condition: {cond:.2e}', xy=(0.95, 0.95),
                xycoords='axes fraction', ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Numerical rank (singular values > 1e-10 * max)
    tol = 1e-10 * s[0]
    num_rank = np.sum(s > tol)
    ax2.axhline(tol, color=COLORS['warning'], linestyle='--', linewidth=1,
               label=f'Rank threshold (rank ≈ {num_rank})')
    ax2.legend(fontsize=8)

    # Panel 3: Column coherence histogram
    ax3 = axes[2]

    # Compute pairwise coherence (normalized inner products)
    G_normalized = G / np.linalg.norm(G, axis=0, keepdims=True)

    # Only compute upper triangle (avoid duplicate pairs)
    coherences = []
    for i in range(n_grid):
        for j in range(i+1, n_grid):
            coh = np.abs(np.dot(G_normalized[:, i], G_normalized[:, j]))
            coherences.append(coh)

    coherences = np.array(coherences)

    ax3.hist(coherences, bins=50, color=COLORS['trajectory'], alpha=0.7, edgecolor='white')
    ax3.axvline(0.33, color=COLORS['warning'], linestyle='--', linewidth=2,
               label='Sparse recovery threshold (0.33)')
    ax3.axvline(coherences.max(), color=COLORS['error'], linestyle='-', linewidth=2,
               label=f'Max coherence: {coherences.max():.3f}')

    ax3.set_xlabel('Coherence |⟨g_i, g_j⟩|')
    ax3.set_ylabel('Count')
    ax3.set_title('Column Coherence Distribution')
    ax3.legend(fontsize=8)
    ax3.set_xlim(0, 1)

    plt.tight_layout()
    return fig


def plot_greens_function_comparison(source_locations: list,
                                    domain_type: str = 'disk',
                                    domain_params: dict = None,
                                    n_boundary: int = 100,
                                    title: str = "Green's Functions Comparison") -> plt.Figure:
    """
    Compare Green's functions for multiple source locations.

    Parameters
    ----------
    source_locations : list of (x, y) tuples
        Source positions to compare
    domain_type : str
        Domain type
    domain_params : dict
        Domain parameters
    n_boundary : int
        Number of boundary points
    title : str
        Figure title

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure
    """
    n_sources = len(source_locations)
    fig, axes = plt.subplots(1, n_sources + 1, figsize=(4*(n_sources+1), 4))

    # First n panels: individual heatmaps
    for i, (xi_x, xi_y) in enumerate(source_locations):
        plot_greens_function((xi_x, xi_y), domain_type, domain_params,
                            resolution=50, title=f'ξ{i+1}=({xi_x:.1f},{xi_y:.1f})',
                            ax=axes[i])

    # Last panel: boundary values comparison
    ax_last = axes[-1]
    boundary = get_domain_boundary(domain_type, domain_params, n_points=n_boundary)
    theta = np.arctan2(boundary[:, 1], boundary[:, 0])
    theta = np.mod(theta, 2*np.pi)
    sort_idx = np.argsort(theta)
    theta_sorted = theta[sort_idx]

    try:
        from ..analytical_solver import greens_function_disk_neumann
    except ImportError:
        try:
            from analytical_solver import greens_function_disk_neumann
        except ImportError:
            def greens_function_disk_neumann(pts, xi):
                dx = pts[:, 0] - xi[0]
                dy = pts[:, 1] - xi[1]
                r = np.sqrt(dx**2 + dy**2)
                r = np.maximum(r, 1e-10)
                return -1/(2*np.pi) * np.log(r)

    colors = plt.cm.tab10(np.linspace(0, 1, n_sources))
    for i, ((xi_x, xi_y), color) in enumerate(zip(source_locations, colors)):
        xi = np.array([xi_x, xi_y])
        G = greens_function_disk_neumann(boundary, xi)
        ax_last.plot(theta_sorted, G[sort_idx], color=color, linewidth=1.5,
                    label=f'ξ{i+1}')

    from .utils import format_axes_pi
    ax_last.set_xlim(0, 2*np.pi)
    format_axes_pi(ax_last, 'x')
    ax_last.set_xlabel('θ')
    ax_last.set_ylabel('G(θ, ξ)')
    ax_last.set_title('Boundary Comparison')
    ax_last.legend(fontsize=8)
    ax_last.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    return fig
