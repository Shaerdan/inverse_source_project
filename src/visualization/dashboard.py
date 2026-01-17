"""
Combined Diagnostic Dashboards
==============================

Comprehensive dashboard visualizations that combine multiple plots
for complete solver diagnostics at a glance.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Dict, Optional

from .config import COLORS, FIGSIZE, DASHBOARD_GRID_SPEC
from .utils import (
    get_domain_boundary, sources_to_arrays, compute_source_errors,
    add_domain_boundary, set_domain_axes, format_axes_pi
)
from .forward_viz import plot_source_configuration, plot_boundary_values
from .inverse_viz import plot_boundary_fit, plot_linear_solution
from .optimization_viz import plot_convergence


def diagnostic_dashboard(sources_true: List[Tuple],
                         sources_recovered: List[Tuple],
                         u_measured: np.ndarray,
                         u_recovered: np.ndarray,
                         theta: np.ndarray,
                         domain_boundary: np.ndarray,
                         history: List[float] = None,
                         sensor_locations: np.ndarray = None,
                         title: str = "Solver Diagnostic") -> plt.Figure:
    """
    Comprehensive single-page diagnostic dashboard.

    Layout (3x3 grid):
    [Source Config   ] [Interior/Boundary] [Boundary Fit    ]
    [Convergence     ] [Position Scatter ] [Intensity Scatter]
    [Residual Detail ] [Error Bar Chart  ] [Summary Stats   ]

    Parameters
    ----------
    sources_true : list
        True source configuration
    sources_recovered : list
        Recovered sources
    u_measured : ndarray
        Measured boundary data
    u_recovered : ndarray
        Recovered boundary data
    theta : ndarray
        Boundary angles
    domain_boundary : ndarray
        Domain boundary points
    history : list, optional
        Convergence history (objective values)
    sensor_locations : ndarray, optional
        Sensor positions
    title : str
        Dashboard title

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure
    """
    fig = plt.figure(figsize=FIGSIZE['dashboard'])
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # Compute errors
    errors = compute_source_errors(sources_true, sources_recovered)
    pos_true, int_true = sources_to_arrays(sources_true)
    pos_rec, int_rec = sources_to_arrays(sources_recovered)

    # Match intensities
    int_rec_matched = np.zeros_like(int_true)
    for true_idx, rec_idx, _ in errors['matching']:
        if rec_idx < len(int_rec):
            int_rec_matched[true_idx] = int_rec[rec_idx]

    # =========================================================================
    # Row 1
    # =========================================================================

    # Panel 1: Source Configuration
    ax1 = fig.add_subplot(gs[0, 0])
    plot_source_configuration(sources_true, domain_boundary,
                             sources_recovered=sources_recovered,
                             sensor_locations=sensor_locations,
                             show_matching=True, show_legend=True,
                             title='Source Configuration', ax=ax1)

    # Panel 2: Boundary values comparison
    ax2 = fig.add_subplot(gs[0, 1])
    sort_idx = np.argsort(theta)
    ax2.plot(theta[sort_idx], u_measured[sort_idx], color=COLORS['measured'],
            linewidth=2, label='Measured')
    ax2.plot(theta[sort_idx], u_recovered[sort_idx], color=COLORS['recovered'],
            linewidth=2, linestyle='--', label='Recovered')
    ax2.fill_between(theta[sort_idx], u_measured[sort_idx], u_recovered[sort_idx],
                    alpha=0.2, color=COLORS['residual'])
    ax2.set_xlim(0, 2*np.pi)
    format_axes_pi(ax2, 'x')
    ax2.set_xlabel('θ')
    ax2.set_ylabel('u(θ)')
    ax2.set_title('Boundary Data')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Boundary fit residual
    ax3 = fig.add_subplot(gs[0, 2])
    residual = u_measured - u_recovered
    ax3.plot(theta[sort_idx], residual[sort_idx], color=COLORS['residual'], linewidth=1.5)
    ax3.axhline(0, color=COLORS['grid'], linestyle='-', linewidth=1)
    ax3.fill_between(theta[sort_idx], 0, residual[sort_idx], alpha=0.3, color=COLORS['residual'])
    rms = np.sqrt(np.mean(residual**2))
    ax3.annotate(f'RMS = {rms:.2e}', xy=(0.02, 0.98), xycoords='axes fraction',
                va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax3.set_xlim(0, 2*np.pi)
    format_axes_pi(ax3, 'x')
    ax3.set_xlabel('θ')
    ax3.set_ylabel('Residual')
    ax3.set_title('Fit Residual')
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # Row 2
    # =========================================================================

    # Panel 4: Convergence
    ax4 = fig.add_subplot(gs[1, 0])
    if history is not None and len(history) > 0:
        plot_convergence(history, title='Convergence', ax=ax4)
    else:
        ax4.text(0.5, 0.5, 'No convergence\nhistory available',
                ha='center', va='center', transform=ax4.transAxes,
                fontsize=12, color=COLORS['grid'])
        ax4.set_title('Convergence')

    # Panel 5: Position scatter
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(pos_true[:, 0], pos_true[:, 1], c=[COLORS['measured']]*len(pos_true),
               s=100, marker='o', label='True', edgecolors='white', linewidths=2)

    pos_rec_matched = np.zeros_like(pos_true)
    for true_idx, rec_idx, _ in errors['matching']:
        if rec_idx < len(pos_rec):
            pos_rec_matched[true_idx] = pos_rec[rec_idx]

    ax5.scatter(pos_rec_matched[:, 0], pos_rec_matched[:, 1], c=[COLORS['recovered']]*len(pos_rec_matched),
               s=100, marker='s', label='Recovered', edgecolors='white', linewidths=2)

    # Arrows from recovered to true
    for i in range(len(pos_true)):
        ax5.annotate('', xy=pos_true[i], xytext=pos_rec_matched[i],
                    arrowprops=dict(arrowstyle='->', color=COLORS['grid'], lw=1))

    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_title('Position Comparison')
    ax5.legend(fontsize=8)
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)

    # Panel 6: Intensity scatter
    ax6 = fig.add_subplot(gs[1, 2])
    colors = [COLORS['source_positive'] if i > 0 else COLORS['source_negative'] for i in int_true]
    ax6.scatter(int_true, int_rec_matched, c=colors, s=100, edgecolors='white', linewidths=2)

    # Perfect line
    lims = [min(int_true.min(), int_rec_matched.min()) - 0.2,
            max(int_true.max(), int_rec_matched.max()) + 0.2]
    ax6.plot(lims, lims, '--', color=COLORS['grid'], linewidth=1)

    if len(int_true) > 1:
        corr = np.corrcoef(int_true, int_rec_matched)[0, 1]
        ax6.annotate(f'r = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                    va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax6.set_xlabel('True intensity')
    ax6.set_ylabel('Recovered intensity')
    ax6.set_title('Intensity Comparison')
    ax6.set_xlim(lims)
    ax6.set_ylim(lims)
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)

    # =========================================================================
    # Row 3
    # =========================================================================

    # Panel 7: Position error distribution
    ax7 = fig.add_subplot(gs[2, 0])
    pos_errors = errors['position_errors']
    n_sources = len(pos_errors)
    x = np.arange(n_sources)
    ax7.bar(x, pos_errors, color=COLORS['trajectory'], alpha=0.8)
    ax7.axhline(errors['position_rmse'], color=COLORS['warning'], linestyle='--',
               linewidth=2, label=f'RMSE = {errors["position_rmse"]:.4f}')
    ax7.set_xlabel('Source pair')
    ax7.set_ylabel('Position error')
    ax7.set_title('Position Errors')
    ax7.set_xticks(x)
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3, axis='y')

    # Panel 8: Intensity error distribution
    ax8 = fig.add_subplot(gs[2, 1])
    int_errors = errors['intensity_errors']
    ax8.bar(x, int_errors, color=COLORS['recovered'], alpha=0.8)
    ax8.axhline(errors['intensity_rmse'], color=COLORS['warning'], linestyle='--',
               linewidth=2, label=f'RMSE = {errors["intensity_rmse"]:.4f}')
    ax8.set_xlabel('Source pair')
    ax8.set_ylabel('Intensity error')
    ax8.set_title('Intensity Errors')
    ax8.set_xticks(x)
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3, axis='y')

    # Panel 9: Summary statistics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    summary_text = f"""
    === Summary Statistics ===

    Sources: {len(sources_true)} true, {len(sources_recovered)} recovered

    Position Errors:
      RMSE:  {errors['position_rmse']:.4f}
      Max:   {errors['position_max']:.4f}

    Intensity Errors:
      RMSE:  {errors['intensity_rmse']:.4f}
      Max:   {errors['intensity_max']:.4f}

    Boundary Fit:
      RMS Residual: {rms:.2e}
      Max Residual: {np.max(np.abs(residual)):.2e}
    """

    if history is not None and len(history) > 0:
        summary_text += f"""
    Optimization:
      Initial obj: {history[0]:.2e}
      Final obj:   {history[-1]:.2e}
      Iterations:  {len(history)}
    """

    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['grid']))

    return fig


def solver_comparison_dashboard(results: List[Dict],
                                sources_true: List[Tuple],
                                u_measured: np.ndarray,
                                theta: np.ndarray,
                                domain_boundary: np.ndarray,
                                title: str = "Solver Comparison") -> plt.Figure:
    """
    Compare multiple solvers side-by-side.

    Parameters
    ----------
    results : list of dict
        Each dict contains:
        - 'name': str, solver name
        - 'sources': list, recovered sources
        - 'u_recovered': ndarray, recovered boundary data
        - 'history': list, optional convergence history
        - 'time': float, optional computation time
    sources_true : list
        True sources
    u_measured : ndarray
        Measured boundary data
    theta : ndarray
        Boundary angles
    domain_boundary : ndarray
        Domain boundary
    title : str
        Dashboard title

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure
    """
    n_solvers = len(results)
    fig = plt.figure(figsize=(5*n_solvers, 12))
    gs = GridSpec(3, n_solvers, figure=fig, hspace=0.3, wspace=0.25)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    pos_true, int_true = sources_to_arrays(sources_true)

    # Collect metrics for comparison
    all_pos_rmse = []
    all_int_rmse = []
    all_rms = []

    for col, result in enumerate(results):
        name = result['name']
        sources_rec = result['sources']
        u_rec = result['u_recovered']
        history = result.get('history', None)
        time_taken = result.get('time', None)

        errors = compute_source_errors(sources_true, sources_rec)
        residual = u_measured - u_rec
        rms = np.sqrt(np.mean(residual**2))

        all_pos_rmse.append(errors['position_rmse'])
        all_int_rmse.append(errors['intensity_rmse'])
        all_rms.append(rms)

        # Row 1: Source recovery
        ax1 = fig.add_subplot(gs[0, col])
        plot_source_configuration(sources_true, domain_boundary,
                                 sources_recovered=sources_rec,
                                 show_matching=True, show_legend=(col == 0),
                                 title=name, ax=ax1)

        # Row 2: Boundary fit
        ax2 = fig.add_subplot(gs[1, col])
        sort_idx = np.argsort(theta)
        ax2.plot(theta[sort_idx], u_measured[sort_idx], color=COLORS['measured'],
                linewidth=1.5, label='Measured')
        ax2.plot(theta[sort_idx], u_rec[sort_idx], color=COLORS['recovered'],
                linewidth=1.5, linestyle='--', label='Recovered')
        ax2.set_xlim(0, 2*np.pi)
        ax2.set_xlabel('θ')
        ax2.set_ylabel('u(θ)')
        ax2.set_title(f'RMS = {rms:.2e}')
        if col == 0:
            ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        # Row 3: Metrics
        ax3 = fig.add_subplot(gs[2, col])
        ax3.axis('off')

        metrics_text = f"""
        Position RMSE: {errors['position_rmse']:.4f}
        Intensity RMSE: {errors['intensity_rmse']:.4f}
        Boundary RMS: {rms:.2e}
        """

        if history:
            metrics_text += f"\nIterations: {len(history)}"
        if time_taken:
            metrics_text += f"\nTime: {time_taken:.2f}s"

        ax3.text(0.5, 0.5, metrics_text, transform=ax3.transAxes,
                fontsize=10, ha='center', va='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['grid']))

    # Add comparison bar chart at bottom (spanning all columns)
    # Actually let's add a small summary
    fig.text(0.5, 0.02, f"Best Position RMSE: {min(all_pos_rmse):.4f} ({results[np.argmin(all_pos_rmse)]['name']})",
            ha='center', fontsize=10, fontweight='bold')

    return fig


def domain_comparison_dashboard(domain_results: Dict[str, Dict],
                                domain_boundaries: Dict[str, np.ndarray],
                                sources_true_dict: Dict[str, List[Tuple]],
                                title: str = "Domain Comparison") -> plt.Figure:
    """
    Compare solver performance across different domains.

    Parameters
    ----------
    domain_results : dict
        Maps domain_type -> result dict with keys:
        - 'sources': recovered sources
        - 'u_measured': measured data
        - 'u_recovered': recovered data
        - 'theta': boundary angles
    domain_boundaries : dict
        Maps domain_type -> boundary array
    sources_true_dict : dict
        Maps domain_type -> true sources
    title : str
        Dashboard title

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure
    """
    domains = list(domain_results.keys())
    n_domains = len(domains)

    fig = plt.figure(figsize=(5*n_domains, 10))
    gs = GridSpec(3, n_domains, figure=fig, hspace=0.35, wspace=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    for col, domain in enumerate(domains):
        result = domain_results[domain]
        boundary = domain_boundaries[domain]
        sources_true = sources_true_dict[domain]

        sources_rec = result['sources']
        u_meas = result['u_measured']
        u_rec = result['u_recovered']
        theta = result['theta']

        errors = compute_source_errors(sources_true, sources_rec)
        rms = np.sqrt(np.mean((u_meas - u_rec)**2))

        # Row 1: Source configuration
        ax1 = fig.add_subplot(gs[0, col])
        plot_source_configuration(sources_true, boundary,
                                 sources_recovered=sources_rec,
                                 show_matching=True, show_legend=False,
                                 title=f'{domain.capitalize()} Domain', ax=ax1)

        # Row 2: Boundary fit
        ax2 = fig.add_subplot(gs[1, col])
        sort_idx = np.argsort(theta)
        ax2.plot(theta[sort_idx], u_meas[sort_idx], color=COLORS['measured'], linewidth=1.5)
        ax2.plot(theta[sort_idx], u_rec[sort_idx], color=COLORS['recovered'],
                linewidth=1.5, linestyle='--')
        ax2.set_xlabel('θ')
        ax2.set_ylabel('u(θ)')
        ax2.set_title(f'Boundary Fit (RMS={rms:.2e})')
        ax2.grid(True, alpha=0.3)

        # Row 3: Metrics summary
        ax3 = fig.add_subplot(gs[2, col])
        metrics = ['Pos RMSE', 'Int RMSE', 'Boundary RMS']
        values = [errors['position_rmse'], errors['intensity_rmse'], rms]

        bars = ax3.barh(metrics, values, color=[COLORS['trajectory'],
                                                 COLORS['recovered'],
                                                 COLORS['residual']])
        ax3.set_xlabel('Error')
        ax3.set_title('Error Metrics')

        # Add value labels
        for bar, val in zip(bars, values):
            ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=8)

    return fig


def quick_diagnostic(forward_solver,
                     inverse_result,
                     sources_true: List[Tuple],
                     domain_type: str = 'disk',
                     domain_params: dict = None,
                     title: str = "Quick Diagnostic") -> plt.Figure:
    """
    Simplified diagnostic for quick checks.

    Creates 2x2 figure with essential information.

    Parameters
    ----------
    forward_solver
        Forward solver used
    inverse_result
        Result from inverse solver (InverseResult object or similar)
    sources_true : list
        True sources
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
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE['comparison'])

    # Extract data from inverse result
    if hasattr(inverse_result, 'sources'):
        sources_rec = [(s.position, s.intensity) if hasattr(s, 'position')
                      else ((s.x, s.y), s.intensity) for s in inverse_result.sources]
    else:
        sources_rec = inverse_result

    if hasattr(inverse_result, 'history'):
        history = inverse_result.history
    else:
        history = None

    # Get boundary data
    u_measured = forward_solver.solve(sources_true)
    u_recovered = forward_solver.solve(sources_rec)
    theta = getattr(forward_solver, 'theta', np.linspace(0, 2*np.pi, len(u_measured), endpoint=False))
    boundary = get_domain_boundary(domain_type, domain_params)

    # Panel 1: Source positions
    plot_source_configuration(sources_true, boundary,
                             sources_recovered=sources_rec,
                             show_matching=True,
                             title='Sources', ax=axes[0, 0])

    # Panel 2: Boundary fit
    sort_idx = np.argsort(theta)
    axes[0, 1].plot(theta[sort_idx], u_measured[sort_idx], color=COLORS['measured'],
                   linewidth=2, label='Measured')
    axes[0, 1].plot(theta[sort_idx], u_recovered[sort_idx], color=COLORS['recovered'],
                   linewidth=2, linestyle='--', label='Recovered')
    rms = np.sqrt(np.mean((u_measured - u_recovered)**2))
    axes[0, 1].set_title(f'Boundary (RMS={rms:.2e})')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Convergence
    if history and len(history) > 0:
        plot_convergence(history, title='Convergence', ax=axes[1, 0])
    else:
        axes[1, 0].text(0.5, 0.5, 'No history', ha='center', va='center',
                       transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Convergence')

    # Panel 4: Summary
    axes[1, 1].axis('off')
    errors = compute_source_errors(sources_true, sources_rec)

    summary = f"""
    Position RMSE: {errors['position_rmse']:.4f}
    Intensity RMSE: {errors['intensity_rmse']:.4f}
    Boundary RMS: {rms:.2e}
    """

    if history:
        summary += f"\n    Final objective: {history[-1]:.2e}"

    axes[1, 1].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1, 1].transAxes, fontsize=11,
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['grid']))
    axes[1, 1].set_title('Summary')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig
