"""
Inverse Problem Visualizations
==============================

Visualizations for analyzing inverse source recovery results.
Includes boundary fit comparison, source recovery analysis, and linear solver output.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.table import Table
from typing import List, Tuple, Optional, Union

from .config import COLORS, COLORMAPS, FIGSIZE, LINESTYLES, get_source_color
from .utils import (
    get_domain_boundary, sources_to_arrays, match_sources, compute_source_errors,
    add_domain_boundary, add_source_markers, add_matching_arrows, set_domain_axes,
    format_axes_pi
)


def plot_boundary_fit(theta: np.ndarray,
                      u_measured: np.ndarray,
                      u_recovered: np.ndarray,
                      show_residual: bool = True,
                      title: str = "Boundary Data Fit",
                      fig: plt.Figure = None) -> Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]:
    """
    Compare measured vs recovered boundary data.

    Parameters
    ----------
    theta : ndarray
        Boundary angles
    u_measured : ndarray
        Measured boundary values
    u_recovered : ndarray
        Recovered boundary values
    show_residual : bool
        If True, creates 2-panel figure with residual
    title : str
        Plot title
    fig : plt.Figure, optional
        Existing figure to use

    Returns
    -------
    ax or (ax_main, ax_residual) : Axes
        The matplotlib axes
    """
    # Sort by theta
    sort_idx = np.argsort(theta)
    theta_sorted = theta[sort_idx]
    u_meas_sorted = u_measured[sort_idx]
    u_rec_sorted = u_recovered[sort_idx]

    if show_residual:
        if fig is None:
            fig, (ax_main, ax_res) = plt.subplots(2, 1, figsize=FIGSIZE['wide'],
                                                   height_ratios=[3, 1],
                                                   sharex=True)
        else:
            ax_main, ax_res = fig.subplots(2, 1, height_ratios=[3, 1], sharex=True)

        # Main panel: overlay
        ax_main.plot(theta_sorted, u_meas_sorted, color=COLORS['measured'],
                    label='Measured', **LINESTYLES['measured'])
        ax_main.plot(theta_sorted, u_rec_sorted, color=COLORS['recovered'],
                    label='Recovered', **LINESTYLES['recovered'])

        # Shade difference
        ax_main.fill_between(theta_sorted, u_meas_sorted, u_rec_sorted,
                            alpha=0.2, color=COLORS['residual'])

        ax_main.set_ylabel('u(θ)')
        ax_main.set_title(title)
        ax_main.legend(loc='upper right')
        ax_main.grid(True, alpha=0.3)

        # Residual panel
        residual = u_meas_sorted - u_rec_sorted
        ax_res.plot(theta_sorted, residual, color=COLORS['residual'],
                   linewidth=1.5)
        ax_res.axhline(0, color=COLORS['grid'], linestyle='-', linewidth=1)
        ax_res.fill_between(theta_sorted, 0, residual, alpha=0.3,
                           color=COLORS['residual'])

        # RMS annotation
        rms = np.sqrt(np.mean(residual**2))
        ax_res.annotate(f'RMS = {rms:.2e}', xy=(0.02, 0.95), xycoords='axes fraction',
                       va='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax_res.set_xlabel('θ (radians)')
        ax_res.set_ylabel('Residual')
        ax_res.grid(True, alpha=0.3)

        format_axes_pi(ax_res, 'x')
        ax_main.set_xlim(0, 2*np.pi)

        plt.tight_layout()
        return ax_main, ax_res

    else:
        if fig is None:
            fig, ax = plt.subplots(figsize=FIGSIZE['wide'])
        else:
            ax = fig.add_subplot(111)

        ax.plot(theta_sorted, u_meas_sorted, color=COLORS['measured'],
               label='Measured', **LINESTYLES['measured'])
        ax.plot(theta_sorted, u_rec_sorted, color=COLORS['recovered'],
               label='Recovered', **LINESTYLES['recovered'])
        ax.fill_between(theta_sorted, u_meas_sorted, u_rec_sorted,
                       alpha=0.2, color=COLORS['residual'])

        # RMS annotation
        rms = np.sqrt(np.mean((u_measured - u_recovered)**2))
        ax.annotate(f'RMS = {rms:.2e}', xy=(0.98, 0.98), xycoords='axes fraction',
                   ha='right', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_xlim(0, 2*np.pi)
        format_axes_pi(ax, 'x')
        ax.set_xlabel('θ (radians)')
        ax.set_ylabel('u(θ)')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        return ax


def plot_source_recovery(sources_true: List[Tuple],
                         sources_recovered: List[Tuple],
                         domain_boundary: np.ndarray,
                         title: str = "Source Recovery Analysis") -> plt.Figure:
    """
    Comprehensive source recovery comparison (4-panel figure).

    Parameters
    ----------
    sources_true : list
        True sources
    sources_recovered : list
        Recovered sources
    domain_boundary : ndarray
        Domain boundary points
    title : str
        Figure title

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE['dashboard'])
    fig.suptitle(title, fontsize=14, fontweight='bold')

    pos_true, int_true = sources_to_arrays(sources_true)
    pos_rec, int_rec = sources_to_arrays(sources_recovered)
    errors = compute_source_errors(sources_true, sources_recovered)

    # Panel 1: Spatial comparison
    ax1 = axes[0, 0]
    add_domain_boundary(ax1, domain_boundary, fill=True)

    # True sources
    for pos, intensity in zip(pos_true, int_true):
        color = get_source_color(intensity)
        ax1.scatter(pos[0], pos[1], c=color, s=150, marker='o',
                   edgecolors='white', linewidths=2, zorder=10)

    # Recovered sources
    for pos, intensity in zip(pos_rec, int_rec):
        color = get_source_color(intensity)
        ax1.scatter(pos[0], pos[1], c='none', s=200, marker='s',
                   edgecolors=color, linewidths=2.5, zorder=9)

    # Matching arrows
    add_matching_arrows(ax1, sources_true, sources_recovered,
                       matching=errors['matching'])

    set_domain_axes(ax1, domain_boundary)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Spatial Comparison')

    # Legend
    ax1.scatter([], [], c=COLORS['source_positive'], s=100, marker='o', label='True (+)')
    ax1.scatter([], [], c=COLORS['source_negative'], s=100, marker='o', label='True (-)')
    ax1.scatter([], [], c='none', s=100, marker='s', edgecolors=COLORS['trajectory'],
               linewidths=2, label='Recovered')
    ax1.legend(loc='upper right', fontsize=7)

    # Panel 2: Position error vs radius
    ax2 = axes[0, 1]
    radii = np.linalg.norm(pos_true, axis=1)
    pos_errors = errors['position_errors']

    ax2.scatter(radii, pos_errors, c=COLORS['trajectory'], s=80, edgecolors='white')

    # Trend line
    if len(radii) > 2:
        z = np.polyfit(radii, pos_errors, 1)
        p = np.poly1d(z)
        r_line = np.linspace(0, radii.max(), 50)
        ax2.plot(r_line, p(r_line), '--', color=COLORS['grid'], linewidth=1)

    ax2.set_xlabel('True source radius')
    ax2.set_ylabel('Position error')
    ax2.set_title('Error vs Radius')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Intensity scatter
    ax3 = axes[1, 0]

    # Reorder recovered intensities according to matching
    int_rec_matched = np.zeros_like(int_true)
    for true_idx, rec_idx, _ in errors['matching']:
        int_rec_matched[true_idx] = int_rec[rec_idx]

    colors = [get_source_color(i) for i in int_true]
    ax3.scatter(int_true, int_rec_matched, c=colors, s=100, edgecolors='white', linewidths=2)

    # Perfect recovery line
    lims = [min(int_true.min(), int_rec.min()) - 0.2,
            max(int_true.max(), int_rec.max()) + 0.2]
    ax3.plot(lims, lims, '--', color=COLORS['grid'], linewidth=1, label='Perfect')

    # Correlation coefficient
    if len(int_true) > 1:
        corr = np.corrcoef(int_true, int_rec_matched)[0, 1]
        ax3.annotate(f'Corr = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                    va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax3.set_xlabel('True intensity')
    ax3.set_ylabel('Recovered intensity')
    ax3.set_title('Intensity Comparison')
    ax3.set_xlim(lims)
    ax3.set_ylim(lims)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Error summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create table data
    table_data = [['Metric', 'Value']]
    table_data.append(['Position RMSE', f'{errors["position_rmse"]:.4f}'])
    table_data.append(['Position Max', f'{errors["position_max"]:.4f}'])
    table_data.append(['Intensity RMSE', f'{errors["intensity_rmse"]:.4f}'])
    table_data.append(['Intensity Max', f'{errors["intensity_max"]:.4f}'])
    table_data.append(['', ''])

    # Individual source errors
    table_data.append(['Source', 'Pos Err | Int Err'])
    for i, (true_idx, rec_idx, dist) in enumerate(errors['matching']):
        int_err = errors['intensity_errors'][i]
        table_data.append([f'{true_idx+1}→{rec_idx+1}', f'{dist:.4f} | {int_err:.4f}'])

    # Draw table
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for j in range(2):
        table[(0, j)].set_facecolor(COLORS['trajectory'])
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax4.set_title('Error Summary', pad=20)

    plt.tight_layout()
    return fig


def plot_linear_solution(grid_positions: np.ndarray,
                         grid_intensities: np.ndarray,
                         domain_boundary: np.ndarray,
                         sources_true: List[Tuple] = None,
                         threshold: float = None,
                         show_peaks: bool = True,
                         title: str = "Linear Solver Output",
                         ax: plt.Axes = None) -> plt.Axes:
    """
    Visualize grid-based linear solver output.

    Parameters
    ----------
    grid_positions : ndarray, shape (n, 2)
        Source candidate grid positions
    grid_intensities : ndarray, shape (n,)
        Recovered intensities at grid points
    domain_boundary : ndarray
        Domain boundary
    sources_true : list, optional
        True sources for comparison
    threshold : float, optional
        Threshold for detecting peaks (default: 10% of max |intensity|)
    show_peaks : bool
        Circle detected peaks
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

    # Default threshold
    if threshold is None:
        threshold = 0.1 * np.max(np.abs(grid_intensities))

    # Domain fill
    add_domain_boundary(ax, domain_boundary, fill=True)

    # Normalize intensities for sizing
    abs_int = np.abs(grid_intensities)
    max_int = np.max(abs_int) if np.max(abs_int) > 0 else 1

    # Size: larger for higher |intensity|
    sizes = 20 + 200 * (abs_int / max_int)

    # Color by intensity (diverging colormap)
    vmax = max(np.max(grid_intensities), -np.min(grid_intensities))
    vmin = -vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    scatter = ax.scatter(grid_positions[:, 0], grid_positions[:, 1],
                        c=grid_intensities, s=sizes, cmap=COLORMAPS['intensity'],
                        norm=norm, alpha=0.7, edgecolors='none')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Intensity')

    # Circle peaks
    if show_peaks:
        peaks = np.where(abs_int > threshold)[0]
        for idx in peaks:
            ax.scatter(grid_positions[idx, 0], grid_positions[idx, 1],
                      s=300, facecolors='none', edgecolors=COLORS['highlight'],
                      linewidths=2, zorder=5)

        # Sparsity annotation
        sparsity = len(peaks) / len(grid_intensities)
        ax.annotate(f'Peaks: {len(peaks)}/{len(grid_intensities)} ({sparsity:.1%})',
                   xy=(0.02, 0.98), xycoords='axes fraction', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # True sources
    if sources_true is not None:
        pos_true, int_true = sources_to_arrays(sources_true)
        for pos, intensity in zip(pos_true, int_true):
            ax.scatter(pos[0], pos[1], c='none', s=250, marker='*',
                      edgecolors='gold', linewidths=3, zorder=10)

        ax.scatter([], [], c='none', s=200, marker='*', edgecolors='gold',
                  linewidths=2, label='True sources')
        ax.legend(loc='upper right')

    set_domain_axes(ax, domain_boundary)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)

    return ax


def plot_linear_methods_comparison(grid_positions: np.ndarray,
                                   results: dict,
                                   domain_boundary: np.ndarray,
                                   sources_true: List[Tuple] = None,
                                   title: str = "Linear Methods Comparison") -> plt.Figure:
    """
    Compare multiple linear solver methods (L2, L1, TV).

    Parameters
    ----------
    grid_positions : ndarray
        Source grid positions
    results : dict
        Dictionary mapping method name to intensity array
        e.g., {'L2': q_l2, 'L1': q_l1, 'TV': q_tv}
    domain_boundary : ndarray
        Domain boundary
    sources_true : list, optional
        True sources
    title : str
        Figure title

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure
    """
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))

    if n_methods == 1:
        axes = [axes]

    for ax, (method_name, intensities) in zip(axes, results.items()):
        plot_linear_solution(grid_positions, intensities, domain_boundary,
                            sources_true=sources_true,
                            title=method_name, ax=ax)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_inverse_residual_map(forward_solver,
                              sources_recovered: List[Tuple],
                              u_measured: np.ndarray,
                              domain_type: str = 'disk',
                              domain_params: dict = None,
                              title: str = "Residual Map") -> plt.Figure:
    """
    Spatial analysis of fit residual.

    Creates 2-panel figure:
    - Left: Residual vs angle (polar)
    - Right: Spectral analysis of residual

    Parameters
    ----------
    forward_solver
        Forward solver instance
    sources_recovered : list
        Recovered sources
    u_measured : ndarray
        Measured boundary data
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE['wide'])

    # Compute recovered data
    u_recovered = forward_solver.solve(sources_recovered)
    theta = getattr(forward_solver, 'theta', np.linspace(0, 2*np.pi, len(u_measured), endpoint=False))

    residual = u_measured - u_recovered

    # Sort
    sort_idx = np.argsort(theta)
    theta_sorted = theta[sort_idx]
    residual_sorted = residual[sort_idx]

    # Panel 1: Residual polar plot
    ax1_polar = fig.add_subplot(1, 2, 1, projection='polar')
    ax1.set_visible(False)

    # Color by sign
    colors = np.where(residual_sorted > 0, COLORS['source_positive'],
                     COLORS['source_negative'])
    ax1_polar.scatter(theta_sorted, np.abs(residual_sorted), c=colors, s=30, alpha=0.7)
    ax1_polar.set_title('Residual Magnitude (Polar)', pad=10)

    # Panel 2: Spectral analysis
    # FFT of residual
    n = len(residual_sorted)
    fft_vals = np.fft.fft(residual_sorted)
    freqs = np.fft.fftfreq(n, d=theta_sorted[1] - theta_sorted[0] if len(theta_sorted) > 1 else 1)

    # Plot positive frequencies only
    pos_mask = freqs >= 0
    ax2.stem(freqs[pos_mask], np.abs(fft_vals[pos_mask]), linefmt=COLORS['trajectory'],
            markerfmt='o', basefmt=COLORS['grid'])
    ax2.set_xlabel('Frequency (cycles/radian)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Residual Spectrum')
    ax2.set_xlim(0, min(10, n//2))
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_source_intensity_bar(sources_true: List[Tuple],
                              sources_recovered: List[Tuple],
                              title: str = "Source Intensities",
                              ax: plt.Axes = None) -> plt.Axes:
    """
    Bar chart comparing true vs recovered intensities.

    Parameters
    ----------
    sources_true : list
        True sources
    sources_recovered : list
        Recovered sources
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

    _, int_true = sources_to_arrays(sources_true)
    _, int_rec = sources_to_arrays(sources_recovered)

    # Match sources
    errors = compute_source_errors(sources_true, sources_recovered)

    # Reorder recovered to match true
    int_rec_matched = np.zeros_like(int_true)
    for true_idx, rec_idx, _ in errors['matching']:
        int_rec_matched[true_idx] = int_rec[rec_idx]

    n = len(int_true)
    x = np.arange(n)
    width = 0.35

    bars1 = ax.bar(x - width/2, int_true, width, label='True',
                  color=COLORS['measured'], alpha=0.8)
    bars2 = ax.bar(x + width/2, int_rec_matched, width, label='Recovered',
                  color=COLORS['recovered'], alpha=0.8)

    ax.axhline(0, color=COLORS['grid'], linewidth=1)
    ax.set_xlabel('Source Index')
    ax.set_ylabel('Intensity')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in range(n)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    return ax
