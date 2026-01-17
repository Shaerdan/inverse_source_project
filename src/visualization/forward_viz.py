"""
Forward Problem Visualizations
==============================

Visualizations for the forward Poisson problem with point sources.
Includes boundary potential plots, interior solution heatmaps, and source configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from typing import List, Tuple, Optional, Union

from .config import COLORS, COLORMAPS, FIGSIZE, LINESTYLES, get_source_color
from .utils import (
    get_domain_boundary, domain_mask, get_bounding_box, sources_to_arrays,
    add_domain_boundary, add_source_markers, add_sensor_markers,
    add_colorbar, set_domain_axes, format_axes_pi
)


def plot_boundary_values(theta: np.ndarray,
                         u_boundary: np.ndarray,
                         sources: List[Tuple] = None,
                         u_reference: np.ndarray = None,
                         title: str = "Boundary Potential u(θ)",
                         ax: plt.Axes = None) -> plt.Axes:
    """
    Plot boundary potential u(θ) vs θ.

    Parameters
    ----------
    theta : ndarray
        Boundary angles (radians), shape (n,)
    u_boundary : ndarray
        Boundary potential values, shape (n,)
    sources : list, optional
        Source list for marking source angles
    u_reference : ndarray, optional
        Reference values (e.g., true solution) for comparison
    title : str
        Plot title
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    ax : plt.Axes
        The matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE['wide'])

    # Sort by theta for proper plotting
    sort_idx = np.argsort(theta)
    theta_sorted = theta[sort_idx]
    u_sorted = u_boundary[sort_idx]

    # Plot main curve
    ax.plot(theta_sorted, u_sorted, color=COLORS['measured'],
            label='u(θ)', **LINESTYLES['measured'])

    # Plot reference if provided
    if u_reference is not None:
        u_ref_sorted = u_reference[sort_idx]
        ax.plot(theta_sorted, u_ref_sorted, color=COLORS['grid'],
                label='Reference', **LINESTYLES['reference'])
        # Shade difference
        ax.fill_between(theta_sorted, u_sorted, u_ref_sorted,
                       alpha=0.2, color=COLORS['residual'])

    # Mark source angles
    if sources is not None:
        positions, intensities = sources_to_arrays(sources)
        for pos, intensity in zip(positions, intensities):
            source_angle = np.arctan2(pos[1], pos[0])
            # Normalize to [0, 2π]
            if source_angle < 0:
                source_angle += 2 * np.pi
            color = get_source_color(intensity)
            ax.axvline(source_angle, color=color, linestyle='--',
                      alpha=0.7, linewidth=1)

    # Format axes
    ax.set_xlim(0, 2*np.pi)
    format_axes_pi(ax, 'x')
    ax.set_xlabel('θ (radians)')
    ax.set_ylabel('u(θ)')
    ax.set_title(title)

    if u_reference is not None or sources is not None:
        ax.legend(loc='upper right')

    ax.grid(True, alpha=0.3)

    return ax


def plot_interior_solution(forward_solver,
                           sources: List[Tuple],
                           domain_type: str = 'disk',
                           domain_params: dict = None,
                           resolution: int = 50,
                           show_sources: bool = True,
                           show_sensors: bool = False,
                           title: str = "Interior Solution u(x,y)",
                           ax: plt.Axes = None) -> plt.Axes:
    """
    Heatmap of solution u(x,y) on domain interior.

    Parameters
    ----------
    forward_solver : ForwardSolver
        Solver with solve_interior or solve_full method
    sources : list
        Source list [((x,y), q), ...]
    domain_type : str
        Domain type for boundary/masking
    domain_params : dict, optional
        Domain parameters
    resolution : int
        Grid resolution for heatmap
    show_sources : bool
        Whether to mark source positions
    show_sensors : bool
        Whether to show sensor locations
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

    # Get domain boundary
    boundary = get_domain_boundary(domain_type, domain_params)
    x_min, x_max, y_min, y_max = get_bounding_box(boundary, margin=0.05)

    # Create evaluation grid
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(x, y)

    # Get domain mask
    mask = domain_mask(xx, yy, domain_type, domain_params)

    # Evaluate solution at interior points
    points_flat = np.column_stack([xx.ravel(), yy.ravel()])

    # Try different methods to get interior solution
    if hasattr(forward_solver, 'solve_interior'):
        u_flat = forward_solver.solve_interior(sources, points_flat)
    elif hasattr(forward_solver, 'solve_full'):
        # For FEM solvers, need to interpolate from mesh
        u_mesh = forward_solver.solve_full(sources)
        from scipy.interpolate import LinearNDInterpolator
        interp = LinearNDInterpolator(forward_solver.nodes, u_mesh, fill_value=np.nan)
        u_flat = interp(points_flat)
    else:
        # Fallback: compute directly using Green's function
        u_flat = _compute_interior_solution(sources, points_flat, domain_type, domain_params)

    u_grid = u_flat.reshape(xx.shape)

    # Mask outside domain
    u_grid = np.ma.masked_where(~mask, u_grid)

    # Create symmetric colormap
    vmax = np.nanmax(np.abs(u_grid))
    vmin = -vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Plot heatmap
    im = ax.contourf(xx, yy, u_grid, levels=50, cmap=COLORMAPS['solution'],
                     norm=norm, extend='both')

    # Add contour lines
    ax.contour(xx, yy, u_grid, levels=15, colors='k', linewidths=0.3, alpha=0.5)

    # Add boundary
    add_domain_boundary(ax, boundary, fill=False, color=COLORS['boundary'], linewidth=2)

    # Add sources
    if show_sources:
        positions, intensities = sources_to_arrays(sources)
        for pos, intensity in zip(positions, intensities):
            color = get_source_color(intensity)
            size = 100 * np.sqrt(np.abs(intensity))
            marker = '+' if intensity > 0 else '_'
            ax.scatter(pos[0], pos[1], c=color, s=size, marker='o',
                      edgecolors='white', linewidths=2, zorder=5)
            # Add +/- label
            ax.plot(pos[0], pos[1], marker, color='white', markersize=8,
                   markeredgewidth=2, zorder=6)

    # Add sensors
    if show_sensors and hasattr(forward_solver, 'sensor_locations'):
        add_sensor_markers(ax, forward_solver.sensor_locations)

    # Colorbar
    add_colorbar(ax, im, label='u(x,y)')

    set_domain_axes(ax, boundary)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)

    return ax


def _compute_interior_solution(sources: List[Tuple], points: np.ndarray,
                                domain_type: str, domain_params: dict) -> np.ndarray:
    """
    Compute interior solution using analytical Green's function.
    Fallback when solver doesn't have solve_interior method.
    """
    try:
        from ..analytical_solver import greens_function_disk_neumann
    except ImportError:
        try:
            from analytical_solver import greens_function_disk_neumann
        except ImportError:
            # Return zeros if we can't compute
            return np.zeros(len(points))

    u = np.zeros(len(points))
    for (pos_x, pos_y), q in sources:
        xi = np.array([pos_x, pos_y])
        u += q * greens_function_disk_neumann(points, xi)

    return u - np.mean(u)


def plot_source_configuration(sources: List[Tuple],
                              domain_boundary: np.ndarray = None,
                              domain_type: str = 'disk',
                              domain_params: dict = None,
                              sources_recovered: List[Tuple] = None,
                              sensor_locations: np.ndarray = None,
                              show_matching: bool = True,
                              show_legend: bool = True,
                              title: str = "Source Configuration",
                              ax: plt.Axes = None) -> plt.Axes:
    """
    Plot sources on domain.

    Parameters
    ----------
    sources : list
        True source list
    domain_boundary : ndarray, optional
        Pre-computed boundary points
    domain_type : str
        Domain type (used if boundary not provided)
    domain_params : dict
        Domain parameters
    sources_recovered : list, optional
        Recovered sources for comparison
    sensor_locations : ndarray, optional
        Sensor positions to show
    show_matching : bool
        Whether to show matching arrows
    show_legend : bool
        Whether to show legend
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

    # Get boundary
    if domain_boundary is None:
        domain_boundary = get_domain_boundary(domain_type, domain_params)

    # Fill domain
    add_domain_boundary(ax, domain_boundary, fill=True)

    # Add sensors
    if sensor_locations is not None:
        add_sensor_markers(ax, sensor_locations, label='Sensors')

    # Add matching arrows first (so sources are on top)
    if sources_recovered is not None and show_matching:
        from .utils import add_matching_arrows
        add_matching_arrows(ax, sources, sources_recovered)

    # Add true sources
    positions, intensities = sources_to_arrays(sources)

    # Plot positive and negative separately for legend
    pos_mask = intensities > 0
    neg_mask = intensities <= 0

    if np.any(pos_mask):
        ax.scatter(positions[pos_mask, 0], positions[pos_mask, 1],
                  c=COLORS['source_positive'], s=150, marker='o',
                  edgecolors='white', linewidths=2, zorder=10,
                  label='True (+)')

    if np.any(neg_mask):
        ax.scatter(positions[neg_mask, 0], positions[neg_mask, 1],
                  c=COLORS['source_negative'], s=150, marker='o',
                  edgecolors='white', linewidths=2, zorder=10,
                  label='True (-)')

    # Add recovered sources
    if sources_recovered is not None:
        pos_rec, int_rec = sources_to_arrays(sources_recovered)

        # Use squares for recovered
        for pos, intensity in zip(pos_rec, int_rec):
            color = get_source_color(intensity)
            ax.scatter(pos[0], pos[1], c='none', s=200, marker='s',
                      edgecolors=color, linewidths=2.5, zorder=9)

        # Add to legend (just one marker)
        ax.scatter([], [], c='none', s=200, marker='s',
                  edgecolors=COLORS['trajectory'], linewidths=2.5,
                  label='Recovered')

    # Add intensity labels
    for i, (pos, intensity) in enumerate(zip(positions, intensities)):
        ax.annotate(f'q={intensity:.2f}',
                   (pos[0], pos[1] - 0.12),
                   fontsize=8, ha='center', va='top',
                   color=COLORS['text'])

    set_domain_axes(ax, domain_boundary)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)

    if show_legend:
        ax.legend(loc='upper right', fontsize=8)

    return ax


def plot_boundary_comparison(theta: np.ndarray,
                             u_values: List[np.ndarray],
                             labels: List[str],
                             colors: List[str] = None,
                             title: str = "Boundary Values Comparison",
                             ax: plt.Axes = None) -> plt.Axes:
    """
    Compare multiple boundary value solutions.

    Parameters
    ----------
    theta : ndarray
        Boundary angles
    u_values : list of ndarray
        List of boundary value arrays to compare
    labels : list of str
        Labels for each solution
    colors : list of str, optional
        Colors for each curve
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

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(u_values)))

    # Sort by theta
    sort_idx = np.argsort(theta)
    theta_sorted = theta[sort_idx]

    for u, label, color in zip(u_values, labels, colors):
        u_sorted = u[sort_idx]
        ax.plot(theta_sorted, u_sorted, label=label, color=color, linewidth=1.5)

    ax.set_xlim(0, 2*np.pi)
    format_axes_pi(ax, 'x')
    ax.set_xlabel('θ (radians)')
    ax.set_ylabel('u(θ)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_forward_diagnostics(forward_solver,
                             sources: List[Tuple],
                             domain_type: str = 'disk',
                             domain_params: dict = None,
                             title: str = "Forward Problem Diagnostics") -> plt.Figure:
    """
    Combined forward problem diagnostic figure.

    Creates 2x2 panel:
    - Top left: Source configuration
    - Top right: Interior solution
    - Bottom left: Boundary values
    - Bottom right: Boundary values (polar plot)

    Parameters
    ----------
    forward_solver : ForwardSolver
        The forward solver
    sources : list
        Source list
    domain_type : str
        Domain type
    domain_params : dict
        Domain parameters
    title : str
        Figure title

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE['dashboard'])
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Get boundary data
    u_boundary = forward_solver.solve(sources)
    theta = getattr(forward_solver, 'theta', np.linspace(0, 2*np.pi, len(u_boundary), endpoint=False))
    boundary = get_domain_boundary(domain_type, domain_params)
    sensor_locs = getattr(forward_solver, 'sensor_locations', None)

    # Panel 1: Source configuration
    plot_source_configuration(sources, boundary, domain_type, domain_params,
                             sensor_locations=sensor_locs,
                             title='Source Positions', ax=axes[0, 0])

    # Panel 2: Interior solution
    plot_interior_solution(forward_solver, sources, domain_type, domain_params,
                          title='Interior Solution', ax=axes[0, 1])

    # Panel 3: Boundary values (Cartesian)
    plot_boundary_values(theta, u_boundary, sources=sources,
                        title='Boundary Potential', ax=axes[1, 0])

    # Panel 4: Boundary values (Polar)
    ax_polar = fig.add_subplot(2, 2, 4, projection='polar')
    ax_polar.plot(theta, u_boundary, color=COLORS['measured'], linewidth=2)
    ax_polar.set_title('Boundary Potential (Polar)', pad=10)
    axes[1, 1].set_visible(False)  # Hide the Cartesian axes

    plt.tight_layout()
    return fig
