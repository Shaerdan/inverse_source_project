"""
Conformal Mapping Visualizations
================================

Visualizations for understanding conformal mappings between domains.
Shows how the unit disk maps to physical domains and vice versa.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import Optional

from .config import COLORS, COLORMAPS, FIGSIZE
from .utils import add_domain_boundary


def plot_domain_correspondence(conformal_map,
                               n_circles: int = 5,
                               n_radials: int = 8,
                               n_boundary: int = 100,
                               title: str = "Conformal Map Correspondence") -> plt.Figure:
    """
    Show mapping between canonical (unit disk) and physical domain.

    Creates 2-panel figure showing how circles and radial lines in the
    disk map to curves in the physical domain.

    Parameters
    ----------
    conformal_map : ConformalMap
        The conformal mapping object with to_disk() and from_disk() methods
    n_circles : int
        Number of concentric circles to show
    n_radials : int
        Number of radial lines to show
    n_boundary : int
        Number of points for boundary
    title : str
        Figure title

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE['wide'])

    # Colors for curves
    circle_colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_circles))
    radial_colors = plt.cm.Oranges(np.linspace(0.3, 0.9, n_radials))

    n_pts = 100  # Points per curve

    # =========================================================================
    # Left panel: Canonical domain (unit disk)
    # =========================================================================

    # Draw unit circle
    theta_boundary = np.linspace(0, 2*np.pi, n_boundary)
    ax1.plot(np.cos(theta_boundary), np.sin(theta_boundary),
            color=COLORS['boundary'], linewidth=2)

    # Concentric circles
    radii = np.linspace(0.2, 0.8, n_circles)
    for r, color in zip(radii, circle_colors):
        theta = np.linspace(0, 2*np.pi, n_pts)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax1.plot(x, y, color=color, linewidth=1.5, alpha=0.8)

    # Radial lines
    angles = np.linspace(0, 2*np.pi, n_radials, endpoint=False)
    for angle, color in zip(angles, radial_colors):
        r_line = np.linspace(0, 0.95, n_pts)
        x = r_line * np.cos(angle)
        y = r_line * np.sin(angle)
        ax1.plot(x, y, color=color, linewidth=1.5, alpha=0.8)

    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.set_xlabel('Re(w)')
    ax1.set_ylabel('Im(w)')
    ax1.set_title('Canonical Domain (Unit Disk)')
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Right panel: Physical domain
    # =========================================================================

    # Physical boundary
    boundary = conformal_map.boundary_physical(n_boundary)
    if np.iscomplexobj(boundary):
        boundary_pts = np.column_stack([boundary.real, boundary.imag])
    else:
        boundary_pts = boundary

    ax2.plot(boundary_pts[:, 0], boundary_pts[:, 1],
            color=COLORS['boundary'], linewidth=2)

    # Map circles
    for r, color in zip(radii, circle_colors):
        theta = np.linspace(0, 2*np.pi, n_pts)
        w = r * np.exp(1j * theta)
        z = conformal_map.from_disk(w)

        if np.iscomplexobj(z):
            x, y = z.real, z.imag
        else:
            x, y = z[:, 0], z[:, 1]

        ax2.plot(x, y, color=color, linewidth=1.5, alpha=0.8)

    # Map radial lines
    for angle, color in zip(angles, radial_colors):
        r_line = np.linspace(0.01, 0.95, n_pts)
        w = r_line * np.exp(1j * angle)
        z = conformal_map.from_disk(w)

        if np.iscomplexobj(z):
            x, y = z.real, z.imag
        else:
            x, y = z[:, 0], z[:, 1]

        ax2.plot(x, y, color=color, linewidth=1.5, alpha=0.8)

    ax2.set_aspect('equal')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Physical Domain')
    ax2.grid(True, alpha=0.3)

    # Adjust limits
    x_margin = (boundary_pts[:, 0].max() - boundary_pts[:, 0].min()) * 0.1
    y_margin = (boundary_pts[:, 1].max() - boundary_pts[:, 1].min()) * 0.1
    ax2.set_xlim(boundary_pts[:, 0].min() - x_margin, boundary_pts[:, 0].max() + x_margin)
    ax2.set_ylim(boundary_pts[:, 1].min() - y_margin, boundary_pts[:, 1].max() + y_margin)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_mapping_jacobian(conformal_map,
                          resolution: int = 50,
                          show_physical_inset: bool = True,
                          title: str = "Mapping Jacobian |f'(z)|",
                          ax: plt.Axes = None) -> plt.Axes:
    """
    Heatmap of |f'(z)| (Jacobian/scale factor) over canonical domain.

    Parameters
    ----------
    conformal_map : ConformalMap
        The conformal mapping
    resolution : int
        Grid resolution
    show_physical_inset : bool
        Show physical domain as inset
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

    # Create polar grid in disk
    r = np.linspace(0.01, 0.95, resolution)
    theta = np.linspace(0, 2*np.pi, 2*resolution)
    R, Theta = np.meshgrid(r, theta)
    W = R * np.exp(1j * Theta)

    # Map to physical domain
    Z = conformal_map.from_disk(W.ravel()).reshape(W.shape)

    # Compute Jacobian numerically
    # |f'(w)| = |dz/dw|
    eps = 1e-6
    W_eps = W + eps
    Z_eps = conformal_map.from_disk(W_eps.ravel()).reshape(W.shape)
    jacobian = np.abs(Z_eps - Z) / eps

    # Convert to Cartesian for plotting
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    # Plot heatmap
    im = ax.pcolormesh(X, Y, jacobian, cmap=COLORMAPS['jacobian'], shading='auto')

    # Unit circle boundary
    theta_b = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta_b), np.sin(theta_b), color=COLORS['boundary'], linewidth=2)

    # Contour lines
    ax.contour(X, Y, jacobian, levels=10, colors='white', linewidths=0.5, alpha=0.5)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Local scale factor |f\'(w)|')

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_xlabel('Re(w)')
    ax.set_ylabel('Im(w)')
    ax.set_title(title)

    # Inset with physical domain
    if show_physical_inset:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper right')

        boundary = conformal_map.boundary_physical(100)
        if np.iscomplexobj(boundary):
            ax_inset.plot(boundary.real, boundary.imag, color=COLORS['boundary'], linewidth=1)
        else:
            ax_inset.plot(boundary[:, 0], boundary[:, 1], color=COLORS['boundary'], linewidth=1)

        ax_inset.set_aspect('equal')
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_title('Physical', fontsize=8)

    return ax


def plot_boundary_correspondence(conformal_map,
                                 sensor_locations_physical: np.ndarray = None,
                                 title: str = "Boundary Correspondence",
                                 ax: plt.Axes = None) -> plt.Axes:
    """
    Show how boundary angle maps between domains.

    Plots θ_physical vs θ_canonical showing the boundary correspondence.

    Parameters
    ----------
    conformal_map : ConformalMap
        The conformal mapping
    sensor_locations_physical : ndarray, optional
        Physical sensor locations to mark
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

    # Sample boundary correspondence
    n_pts = 200
    theta_canonical = np.linspace(0, 2*np.pi, n_pts)

    # Points on canonical (disk) boundary
    w_boundary = np.exp(1j * theta_canonical)

    # Map to physical boundary
    z_boundary = conformal_map.from_disk(w_boundary)

    # Get angle in physical domain
    if np.iscomplexobj(z_boundary):
        theta_physical = np.angle(z_boundary)
    else:
        theta_physical = np.arctan2(z_boundary[:, 1], z_boundary[:, 0])

    # Unwrap to avoid discontinuities
    theta_physical = np.unwrap(theta_physical)

    # Normalize to [0, 2π]
    theta_physical = np.mod(theta_physical, 2*np.pi)

    # Sort for clean plotting
    sort_idx = np.argsort(theta_canonical)

    # Main curve
    ax.plot(theta_canonical[sort_idx], theta_physical[sort_idx],
           color=COLORS['trajectory'], linewidth=2, label='Correspondence')

    # Identity line (would be this for disk)
    ax.plot([0, 2*np.pi], [0, 2*np.pi], '--', color=COLORS['grid'],
           linewidth=1, label='Identity (disk)')

    # Mark sensors if provided
    if sensor_locations_physical is not None:
        # Convert to complex
        if sensor_locations_physical.ndim == 2:
            z_sensors = sensor_locations_physical[:, 0] + 1j * sensor_locations_physical[:, 1]
        else:
            z_sensors = sensor_locations_physical

        # Map to canonical
        w_sensors = conformal_map.to_disk(z_sensors)

        theta_sens_physical = np.angle(z_sensors)
        theta_sens_physical = np.mod(theta_sens_physical, 2*np.pi)

        if np.iscomplexobj(w_sensors):
            theta_sens_canonical = np.angle(w_sensors)
        else:
            theta_sens_canonical = np.arctan2(w_sensors[:, 1], w_sensors[:, 0])
        theta_sens_canonical = np.mod(theta_sens_canonical, 2*np.pi)

        ax.scatter(theta_sens_canonical, theta_sens_physical,
                  c=COLORS['sensor'], s=20, zorder=5, label='Sensors')

    from .utils import format_axes_pi
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0, 2*np.pi)
    format_axes_pi(ax, 'x')
    format_axes_pi(ax, 'y')
    ax.set_xlabel('θ_canonical (disk angle)')
    ax.set_ylabel('θ_physical (domain angle)')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return ax


def plot_conformal_distortion(conformal_map,
                              resolution: int = 30,
                              title: str = "Local Distortion Analysis") -> plt.Figure:
    """
    Analyze local distortion of the conformal map.

    Shows:
    - Area distortion (|f'(z)|²)
    - Angle preservation check
    - Grid deformation

    Parameters
    ----------
    conformal_map : ConformalMap
        The conformal mapping
    resolution : int
        Grid resolution
    title : str
        Figure title

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # =========================================================================
    # Panel 1: Area distortion
    # =========================================================================
    ax1 = axes[0]

    r = np.linspace(0.1, 0.9, resolution)
    theta = np.linspace(0, 2*np.pi, 2*resolution)
    R, Theta = np.meshgrid(r, theta)
    W = R * np.exp(1j * Theta)

    # Compute area distortion = |dz/dw|²
    eps = 1e-6
    Z = conformal_map.from_disk(W.ravel()).reshape(W.shape)
    Z_eps = conformal_map.from_disk((W + eps).ravel()).reshape(W.shape)
    area_dist = np.abs((Z_eps - Z) / eps)**2

    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    im = ax1.pcolormesh(X, Y, area_dist, cmap='RdBu_r', shading='auto')
    theta_b = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta_b), np.sin(theta_b), color=COLORS['boundary'], linewidth=2)
    plt.colorbar(im, ax=ax1, label='Area distortion')
    ax1.set_aspect('equal')
    ax1.set_title('Area Distortion |f\'|²')
    ax1.set_xlabel('Re(w)')
    ax1.set_ylabel('Im(w)')

    # =========================================================================
    # Panel 2: Grid in physical domain
    # =========================================================================
    ax2 = axes[1]

    # Physical boundary
    boundary = conformal_map.boundary_physical(100)
    if np.iscomplexobj(boundary):
        ax2.plot(boundary.real, boundary.imag, color=COLORS['boundary'], linewidth=2)
    else:
        ax2.plot(boundary[:, 0], boundary[:, 1], color=COLORS['boundary'], linewidth=2)

    # Map circles
    for r_val in [0.3, 0.5, 0.7, 0.9]:
        theta_c = np.linspace(0, 2*np.pi, 100)
        w = r_val * np.exp(1j * theta_c)
        z = conformal_map.from_disk(w)
        if np.iscomplexobj(z):
            ax2.plot(z.real, z.imag, color=COLORS['measured'], alpha=0.5, linewidth=0.8)
        else:
            ax2.plot(z[:, 0], z[:, 1], color=COLORS['measured'], alpha=0.5, linewidth=0.8)

    # Map radials
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        r_line = np.linspace(0.1, 0.9, 50)
        w = r_line * np.exp(1j * angle)
        z = conformal_map.from_disk(w)
        if np.iscomplexobj(z):
            ax2.plot(z.real, z.imag, color=COLORS['recovered'], alpha=0.5, linewidth=0.8)
        else:
            ax2.plot(z[:, 0], z[:, 1], color=COLORS['recovered'], alpha=0.5, linewidth=0.8)

    ax2.set_aspect('equal')
    ax2.set_title('Mapped Grid')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # =========================================================================
    # Panel 3: Jacobian histogram
    # =========================================================================
    ax3 = axes[2]

    jacobian = np.sqrt(area_dist).ravel()
    jacobian = jacobian[np.isfinite(jacobian)]

    ax3.hist(jacobian, bins=50, color=COLORS['trajectory'], alpha=0.7, edgecolor='white')
    ax3.axvline(1.0, color=COLORS['warning'], linestyle='--', linewidth=2,
               label='No distortion')
    ax3.axvline(np.mean(jacobian), color=COLORS['error'], linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(jacobian):.2f}')

    ax3.set_xlabel('|f\'(w)|')
    ax3.set_ylabel('Count')
    ax3.set_title('Jacobian Distribution')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_source_mapping(conformal_map,
                        sources_physical: list,
                        title: str = "Source Position Mapping") -> plt.Figure:
    """
    Show how sources in physical domain map to canonical domain.

    Parameters
    ----------
    conformal_map : ConformalMap
        The conformal mapping
    sources_physical : list
        Source list in physical domain [((x,y), q), ...]
    title : str
        Figure title

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure
    """
    from .utils import sources_to_arrays
    from .config import get_source_color

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE['wide'])

    pos_physical, intensities = sources_to_arrays(sources_physical)

    # Convert to complex
    z_sources = pos_physical[:, 0] + 1j * pos_physical[:, 1]

    # Map to canonical
    w_sources = conformal_map.to_disk(z_sources)

    # =========================================================================
    # Left: Physical domain
    # =========================================================================
    boundary = conformal_map.boundary_physical(100)
    if np.iscomplexobj(boundary):
        ax1.plot(boundary.real, boundary.imag, color=COLORS['boundary'], linewidth=2)
    else:
        ax1.plot(boundary[:, 0], boundary[:, 1], color=COLORS['boundary'], linewidth=2)

    for i, (pos, intensity) in enumerate(zip(pos_physical, intensities)):
        color = get_source_color(intensity)
        ax1.scatter(pos[0], pos[1], c=color, s=150, marker='o',
                   edgecolors='white', linewidths=2, zorder=5)
        ax1.annotate(f'{i+1}', (pos[0]+0.05, pos[1]+0.05), fontsize=9)

    ax1.set_aspect('equal')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Physical Domain')
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Right: Canonical domain
    # =========================================================================
    theta_b = np.linspace(0, 2*np.pi, 100)
    ax2.plot(np.cos(theta_b), np.sin(theta_b), color=COLORS['boundary'], linewidth=2)

    for i, (w, intensity) in enumerate(zip(w_sources, intensities)):
        color = get_source_color(intensity)
        if np.iscomplexobj(w):
            x, y = w.real, w.imag
        else:
            x, y = w[0], w[1]
        ax2.scatter(x, y, c=color, s=150, marker='o',
                   edgecolors='white', linewidths=2, zorder=5)
        ax2.annotate(f'{i+1}', (x+0.05, y+0.05), fontsize=9)

    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.set_xlabel('Re(w)')
    ax2.set_ylabel('Im(w)')
    ax2.set_title('Canonical Domain (Unit Disk)')
    ax2.grid(True, alpha=0.3)

    # Add arrows connecting corresponding sources
    for pos, w, intensity in zip(pos_physical, w_sources, intensities):
        color = get_source_color(intensity)
        if np.iscomplexobj(w):
            w_x, w_y = w.real, w.imag
        else:
            w_x, w_y = w[0], w[1]

        # Draw arrow in figure coordinates
        con = plt.matplotlib.patches.ConnectionPatch(
            xyA=(pos[0], pos[1]), xyB=(w_x, w_y),
            coordsA="data", coordsB="data",
            axesA=ax1, axesB=ax2,
            arrowstyle="->", color=color, alpha=0.5,
            connectionstyle="arc3,rad=0.2"
        )
        fig.add_artist(con)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig
