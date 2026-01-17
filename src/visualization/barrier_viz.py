"""
Barrier/Constraint Visualizations
=================================

Visualizations for understanding barrier functions and optimization landscapes.
Critical for debugging nonlinear inverse solver constraint handling.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Circle
from typing import Callable, Tuple, List, Optional

from .config import (COLORS, COLORMAPS, FIGSIZE, BARRIER_MU_DEFAULT,
                     BARRIER_R_MAX_DEFAULT, BARRIER_LEVELS)
from .utils import get_domain_boundary, domain_mask, get_bounding_box, add_domain_boundary


def plot_barrier_landscape(domain_type: str = 'disk',
                           domain_params: dict = None,
                           mu: float = BARRIER_MU_DEFAULT,
                           R_max: float = BARRIER_R_MAX_DEFAULT,
                           resolution: int = 100,
                           show_contours: bool = True,
                           title: str = None,
                           ax: plt.Axes = None) -> plt.Axes:
    """
    Visualize log-barrier function over domain.

    For disk domain:
        barrier(r) = -μ * log(R_max² - r²)  for r < R_max
        barrier(r) = penalty * (r - R_max)²  for r >= R_max

    Parameters
    ----------
    domain_type : str
        Domain type ('disk', 'ellipse', etc.)
    domain_params : dict
        Domain parameters
    mu : float
        Barrier coefficient
    R_max : float
        Feasible region boundary (fraction of domain radius)
    resolution : int
        Grid resolution
    show_contours : bool
        Show contour lines
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
    x_min, x_max, y_min, y_max = get_bounding_box(boundary, margin=0.1)

    # Create grid
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(x, y)

    # Compute barrier function
    barrier = np.zeros_like(xx)

    if domain_type == 'disk':
        radius = domain_params.get('radius', 1.0)
        r = np.sqrt(xx**2 + yy**2)
        r_bound = R_max * radius

        # Interior: log barrier
        interior = r < r_bound
        barrier[interior] = -mu * np.log(np.maximum(r_bound**2 - r[interior]**2, 1e-15))

        # Exterior: quadratic penalty
        exterior = r >= r_bound
        penalty_weight = 1000.0
        barrier[exterior] = penalty_weight * (r[exterior] - r_bound)**2

    elif domain_type == 'ellipse':
        a = domain_params.get('a', 2.0)
        b = domain_params.get('b', 1.0)

        # Normalized ellipse coordinate
        r_norm = np.sqrt((xx/a)**2 + (yy/b)**2)

        interior = r_norm < R_max
        barrier[interior] = -mu * np.log(np.maximum(R_max**2 - r_norm[interior]**2, 1e-15))

        exterior = r_norm >= R_max
        penalty_weight = 1000.0
        barrier[exterior] = penalty_weight * (r_norm[exterior] - R_max)**2

    else:
        # Generic domain: use distance-based barrier
        from scipy.ndimage import distance_transform_edt
        mask = domain_mask(xx, yy, domain_type, domain_params)

        # Distance to boundary (inside positive, outside negative)
        dist_inside = distance_transform_edt(mask) * (x[1] - x[0])
        dist_outside = distance_transform_edt(~mask) * (x[1] - x[0])
        signed_dist = dist_inside - dist_outside

        # Barrier based on signed distance
        margin = R_max * np.max(dist_inside)  # Use fraction of max interior distance

        interior = signed_dist > (1 - R_max) * margin
        barrier[interior] = -mu * np.log(np.maximum(signed_dist[interior], 1e-15))

        exterior = signed_dist <= (1 - R_max) * margin
        penalty_weight = 1000.0
        barrier[exterior] = penalty_weight * np.maximum(0, margin - signed_dist[exterior])**2

    # Mask outside domain for visualization
    outside = ~domain_mask(xx, yy, domain_type, domain_params)
    barrier_display = np.ma.masked_where(outside, barrier)

    # Determine colormap range
    vmin = np.nanmin(barrier_display)
    vmax = min(np.nanmax(barrier_display), vmin + 10)  # Clip for visibility

    # Plot heatmap
    im = ax.pcolormesh(xx, yy, barrier_display, cmap=COLORMAPS['barrier'],
                       vmin=vmin, vmax=vmax, shading='auto')

    # Contour lines
    if show_contours:
        # Use custom levels based on barrier range
        levels = np.linspace(vmin, vmax, 10)
        ax.contour(xx, yy, barrier_display, levels=levels, colors='k',
                  linewidths=0.5, alpha=0.5)

    # Domain boundary (true boundary at r=1)
    add_domain_boundary(ax, boundary, fill=False, color=COLORS['boundary'],
                       linewidth=2, label='Domain boundary')

    # Feasible region boundary at R_max
    if domain_type == 'disk':
        radius = domain_params.get('radius', 1.0)
        circle = Circle((0, 0), R_max * radius, fill=False,
                        edgecolor=COLORS['warning'], linestyle='--',
                        linewidth=2, label=f'R_max = {R_max}')
        ax.add_patch(circle)
    elif domain_type == 'ellipse':
        from matplotlib.patches import Ellipse
        a = domain_params.get('a', 2.0)
        b = domain_params.get('b', 1.0)
        ellipse = plt.matplotlib.patches.Ellipse((0, 0), 2*R_max*a, 2*R_max*b,
                                                   fill=False, edgecolor=COLORS['warning'],
                                                   linestyle='--', linewidth=2)
        ax.add_patch(ellipse)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Barrier Value')

    # Annotations
    ax.annotate(f'μ = {mu:.0e}', xy=(0.02, 0.98), xycoords='axes fraction',
               va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if title is None:
        title = f'Log-Barrier Function (μ={mu:.0e})'
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)

    return ax


def plot_objective_slice(objective_fn: Callable,
                         base_params: np.ndarray,
                         vary_indices: Tuple[int, int],
                         ranges: Tuple[Tuple[float, float], Tuple[float, float]],
                         resolution: int = 50,
                         true_values: Tuple[float, float] = None,
                         current_values: Tuple[float, float] = None,
                         log_scale: bool = True,
                         domain_boundary: np.ndarray = None,
                         title: str = "Objective Function Slice",
                         ax: plt.Axes = None) -> plt.Axes:
    """
    2D slice of objective function.

    Parameters
    ----------
    objective_fn : callable
        Objective function f(params) -> float
    base_params : ndarray
        Base parameter vector (all parameters fixed at these values except two)
    vary_indices : tuple of int
        Indices of the two parameters to vary
    ranges : tuple of (min, max)
        Ranges for each varied parameter
    resolution : int
        Grid resolution
    true_values : tuple, optional
        True values of the varied parameters
    current_values : tuple, optional
        Current optimizer values
    log_scale : bool
        Use log scale for objective values
    domain_boundary : ndarray, optional
        If the slice is through positions, show domain boundary
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

    idx1, idx2 = vary_indices
    (x_min, x_max), (y_min, y_max) = ranges

    # Create grid
    x_vals = np.linspace(x_min, x_max, resolution)
    y_vals = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(x_vals, y_vals)

    # Evaluate objective
    obj_vals = np.zeros_like(xx)
    for i in range(resolution):
        for j in range(resolution):
            params = base_params.copy()
            params[idx1] = xx[i, j]
            params[idx2] = yy[i, j]
            try:
                obj_vals[i, j] = objective_fn(params)
            except:
                obj_vals[i, j] = np.nan

    # Handle invalid values
    obj_vals = np.ma.masked_invalid(obj_vals)

    # Determine color normalization
    if log_scale:
        obj_positive = np.ma.masked_less_equal(obj_vals, 0)
        if obj_positive.count() > 0:
            vmin = max(np.nanmin(obj_positive), 1e-10)
            vmax = np.nanmax(obj_positive)
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None
            log_scale = False
    else:
        norm = None

    # Plot heatmap
    im = ax.pcolormesh(xx, yy, obj_vals, cmap='viridis', norm=norm, shading='auto')

    # Contour lines
    if log_scale and norm is not None:
        levels = np.logspace(np.log10(vmin), np.log10(vmax), 10)
    else:
        levels = 15
    ax.contour(xx, yy, obj_vals, levels=levels, colors='w', linewidths=0.5, alpha=0.5)

    # Find and mark minimum in the slice
    min_idx = np.unravel_index(np.nanargmin(obj_vals), obj_vals.shape)
    ax.scatter(xx[min_idx], yy[min_idx], c='cyan', s=100, marker='v',
              edgecolors='white', linewidths=2, zorder=10, label='Slice minimum')

    # Mark true values
    if true_values is not None:
        ax.scatter(true_values[0], true_values[1], c='lime', s=150, marker='*',
                  edgecolors='white', linewidths=2, zorder=10, label='True')

    # Mark current values
    if current_values is not None:
        ax.scatter(current_values[0], current_values[1], c=COLORS['final'],
                  s=100, marker='o', edgecolors='white', linewidths=2,
                  zorder=10, label='Current')

    # Domain boundary (if applicable - when varying position coordinates)
    if domain_boundary is not None:
        add_domain_boundary(ax, domain_boundary, fill=False,
                           color=COLORS['boundary'], linewidth=2)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    label = 'log(Objective)' if log_scale else 'Objective'
    cbar.set_label(label)

    ax.set_xlabel(f'Parameter {idx1}')
    ax.set_ylabel(f'Parameter {idx2}')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)

    return ax


def plot_barrier_radial_profile(mu_values: List[float] = None,
                                R_max: float = BARRIER_R_MAX_DEFAULT,
                                domain_radius: float = 1.0,
                                title: str = "Barrier Radial Profile",
                                ax: plt.Axes = None) -> plt.Axes:
    """
    Plot barrier value vs radius for different μ values.

    Parameters
    ----------
    mu_values : list of float
        Barrier coefficients to compare
    R_max : float
        Feasible region boundary
    domain_radius : float
        Domain radius
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

    if mu_values is None:
        mu_values = [1e-3, 1e-4, 1e-5]

    r = np.linspace(0, domain_radius, 200)
    r_bound = R_max * domain_radius

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(mu_values)))

    for mu, color in zip(mu_values, colors):
        barrier = np.zeros_like(r)

        # Interior: log barrier
        interior = r < r_bound
        barrier[interior] = -mu * np.log(np.maximum(r_bound**2 - r[interior]**2, 1e-15))

        # Exterior: quadratic penalty
        exterior = r >= r_bound
        penalty_weight = 1000.0
        barrier[exterior] = penalty_weight * (r[exterior] - r_bound)**2

        # Clip for visualization
        barrier = np.clip(barrier, -10, 100)

        ax.plot(r, barrier, color=color, linewidth=2, label=f'μ = {mu:.0e}')

    # Mark boundaries
    ax.axvline(r_bound, color=COLORS['warning'], linestyle='--', linewidth=2,
              label=f'R_max = {R_max}')
    ax.axvline(domain_radius, color=COLORS['boundary'], linestyle='-', linewidth=2,
              label='Domain boundary')

    # Annotation about μ effect
    ax.annotate('Larger μ → stronger barrier\nSmaller μ → closer to hard constraint',
               xy=(0.02, 0.98), xycoords='axes fraction', va='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel('Radius r')
    ax.set_ylabel('Barrier Value')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, domain_radius * 1.1)
    ax.set_ylim(-2, 20)

    return ax


def plot_constraint_violation(trajectory: List[np.ndarray],
                              n_sources: int,
                              domain_type: str = 'disk',
                              domain_params: dict = None,
                              R_max: float = BARRIER_R_MAX_DEFAULT,
                              title: str = "Constraint Violation During Optimization",
                              ax: plt.Axes = None) -> plt.Axes:
    """
    Plot constraint violation (distance to boundary) during optimization.

    Parameters
    ----------
    trajectory : list of ndarray
        Parameter vectors at each iteration
    n_sources : int
        Number of sources
    domain_type : str
        Domain type
    domain_params : dict
        Domain parameters
    R_max : float
        Feasible region boundary
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

    n_iters = len(trajectory)
    iterations = np.arange(n_iters)

    # Compute max normalized distance for each iteration
    max_violations = []

    for params in trajectory:
        violations = []
        for i in range(n_sources):
            x = params[2*i]
            y = params[2*i + 1]

            if domain_type == 'disk':
                radius = domain_params.get('radius', 1.0)
                r_norm = np.sqrt(x**2 + y**2) / radius
            elif domain_type == 'ellipse':
                a = domain_params.get('a', 2.0)
                b = domain_params.get('b', 1.0)
                r_norm = np.sqrt((x/a)**2 + (y/b)**2)
            else:
                # Generic: just use distance from origin
                r_norm = np.sqrt(x**2 + y**2)

            violations.append(r_norm)

        max_violations.append(max(violations))

    max_violations = np.array(max_violations)

    # Plot
    ax.plot(iterations, max_violations, color=COLORS['trajectory'], linewidth=2,
           label='Max normalized position')

    # Feasible region boundary
    ax.axhline(R_max, color=COLORS['warning'], linestyle='--', linewidth=2,
              label=f'Feasible boundary (R_max={R_max})')

    # Domain boundary
    ax.axhline(1.0, color=COLORS['boundary'], linestyle='-', linewidth=2,
              label='Domain boundary')

    # Highlight violations
    violation_mask = max_violations > R_max
    if np.any(violation_mask):
        ax.fill_between(iterations, R_max, max_violations,
                       where=violation_mask, alpha=0.3, color=COLORS['error'],
                       label='Violations')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max Normalized Position')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(1.1, max_violations.max() * 1.1))

    return ax


def plot_barrier_diagnostics(trajectory: List[np.ndarray],
                             history: List[float],
                             n_sources: int,
                             domain_type: str = 'disk',
                             domain_params: dict = None,
                             mu: float = BARRIER_MU_DEFAULT,
                             R_max: float = BARRIER_R_MAX_DEFAULT,
                             title: str = "Barrier Diagnostics") -> plt.Figure:
    """
    Combined barrier function diagnostic figure.

    Panels:
    - Top left: Barrier landscape
    - Top right: Radial profile for different μ
    - Bottom left: Constraint violation during optimization
    - Bottom right: Convergence

    Parameters
    ----------
    trajectory : list of ndarray
        Parameter vectors
    history : list of float
        Objective values
    n_sources : int
        Number of sources
    domain_type : str
        Domain type
    domain_params : dict
        Domain parameters
    mu : float
        Barrier coefficient used
    R_max : float
        Feasible region boundary
    title : str
        Figure title

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE['dashboard'])

    # Panel 1: Barrier landscape with final positions
    plot_barrier_landscape(domain_type, domain_params, mu, R_max,
                          title='Barrier Landscape', ax=axes[0, 0])

    # Add final source positions
    if trajectory:
        final_params = trajectory[-1]
        for i in range(n_sources):
            x, y = final_params[2*i], final_params[2*i + 1]
            axes[0, 0].scatter(x, y, c=COLORS['final'], s=100, marker='o',
                              edgecolors='white', linewidths=2, zorder=10)

    # Panel 2: Radial profile
    plot_barrier_radial_profile(mu_values=[mu, mu*10, mu*0.1], R_max=R_max,
                               title='Radial Barrier Profile', ax=axes[0, 1])

    # Panel 3: Constraint violation
    plot_constraint_violation(trajectory, n_sources, domain_type, domain_params,
                             R_max=R_max, title='Constraint Violation', ax=axes[1, 0])

    # Panel 4: Convergence
    from .optimization_viz import plot_convergence
    plot_convergence(history, title='Convergence', ax=axes[1, 1])

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig
