"""
Optimization Diagnostics Visualization
======================================

Visualizations for understanding optimization behavior in nonlinear inverse problems.
Includes convergence plots, source trajectories, and parameter evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
from typing import List, Tuple, Optional, Union

from .config import (COLORS, COLORMAPS, FIGSIZE, LINESTYLES,
                     CONVERGENCE_LOG_THRESHOLD, get_trajectory_color)
from .utils import (get_domain_boundary, sources_to_arrays, add_domain_boundary,
                    add_source_markers, set_domain_axes, get_bounding_box)


def plot_convergence(history: List[float],
                     true_minimum: float = None,
                     log_scale: bool = True,
                     milestones: List[int] = None,
                     show_rate: bool = True,
                     title: str = "Optimization Convergence",
                     ax: plt.Axes = None) -> plt.Axes:
    """
    Plot objective value vs iteration.

    Parameters
    ----------
    history : list of float
        Objective values at each iteration
    true_minimum : float, optional
        True minimum value (if known)
    log_scale : bool
        Use log scale for y-axis
    milestones : list of int, optional
        Iteration numbers to mark (e.g., restart points)
    show_rate : bool
        Annotate convergence rate
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

    history = np.array(history)
    iterations = np.arange(len(history))

    # Clip very small values for log scale
    if log_scale:
        history_plot = np.maximum(history, CONVERGENCE_LOG_THRESHOLD)
    else:
        history_plot = history

    # Main convergence curve
    ax.plot(iterations, history_plot, color=COLORS['trajectory'],
            linewidth=2, label='Objective')

    # Mark initial and final
    ax.scatter([0], [history_plot[0]], color=COLORS['initial'],
              s=100, zorder=5, label=f'Initial: {history[0]:.2e}')
    ax.scatter([len(history)-1], [history_plot[-1]], color=COLORS['final'],
              s=100, zorder=5, label=f'Final: {history[-1]:.2e}')

    # True minimum
    if true_minimum is not None:
        ax.axhline(true_minimum, color=COLORS['success'], linestyle='--',
                  linewidth=1.5, label=f'True min: {true_minimum:.2e}')

    # Milestones
    if milestones:
        for m in milestones:
            if 0 <= m < len(history):
                ax.axvline(m, color=COLORS['grid'], linestyle=':',
                          linewidth=1, alpha=0.7)

    # Convergence rate annotation
    if show_rate and len(history) > 20:
        # Compute rate from last 20% of iterations
        start_idx = int(0.8 * len(history))
        if history[-1] > CONVERGENCE_LOG_THRESHOLD:
            log_vals = np.log(history[start_idx:])
            iters = np.arange(start_idx, len(history))
            if len(iters) > 2:
                slope, _ = np.polyfit(iters, log_vals, 1)
                rate_text = f'Rate: {slope:.2e}/iter'
                ax.annotate(rate_text, xy=(0.95, 0.95), xycoords='axes fraction',
                           ha='right', va='top', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    return ax


def plot_multistart_convergence(histories: List[List[float]],
                                labels: List[str] = None,
                                highlight_best: bool = True,
                                log_scale: bool = True,
                                title: str = "Multi-Start Convergence",
                                ax: plt.Axes = None) -> plt.Axes:
    """
    Compare convergence across multiple optimization runs.

    Parameters
    ----------
    histories : list of list of float
        Objective histories for each run
    labels : list of str, optional
        Labels for each run
    highlight_best : bool
        Highlight the run with lowest final value
    log_scale : bool
        Use log scale
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

    n_runs = len(histories)
    colors = plt.cm.tab10(np.linspace(0, 1, n_runs))

    final_values = [h[-1] for h in histories]
    best_idx = np.argmin(final_values)

    if labels is None:
        labels = [f'Run {i+1}' for i in range(n_runs)]

    for i, (history, label, color) in enumerate(zip(histories, labels, colors)):
        history = np.array(history)
        if log_scale:
            history = np.maximum(history, CONVERGENCE_LOG_THRESHOLD)

        lw = 3 if (highlight_best and i == best_idx) else 1
        alpha = 1.0 if (highlight_best and i == best_idx) else 0.5

        ax.plot(history, color=color, linewidth=lw, alpha=alpha, label=label)
        ax.scatter([len(history)-1], [history[-1]], color=color, s=50, zorder=5)

    # Summary annotation
    summary = f'Best: {min(final_values):.2e}\nWorst: {max(final_values):.2e}\nMedian: {np.median(final_values):.2e}'
    ax.annotate(summary, xy=(0.02, 0.02), xycoords='axes fraction',
               fontsize=8, va='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    return ax


def plot_source_trajectory(trajectory: List[np.ndarray],
                           n_sources: int,
                           domain_boundary: np.ndarray = None,
                           domain_type: str = 'disk',
                           domain_params: dict = None,
                           sources_true: List[Tuple] = None,
                           source_indices: List[int] = None,
                           show_arrows: bool = True,
                           title: str = "Source Position Trajectories",
                           ax: plt.Axes = None) -> plt.Axes:
    """
    Plot source position trajectories during optimization.

    Parameters
    ----------
    trajectory : list of ndarray
        Parameter vectors at each iteration.
        Layout: [x0, y0, x1, y1, ..., q0, q1, ...]
    n_sources : int
        Number of sources
    domain_boundary : ndarray, optional
        Pre-computed boundary
    domain_type : str
        Domain type
    domain_params : dict
        Domain parameters
    sources_true : list, optional
        True source positions for comparison
    source_indices : list of int, optional
        Which sources to show (default: all)
    show_arrows : bool
        Show direction arrows on trajectories
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

    if domain_boundary is None:
        domain_boundary = get_domain_boundary(domain_type, domain_params)

    if source_indices is None:
        source_indices = list(range(n_sources))

    n_iters = len(trajectory)

    # Fill and draw boundary
    add_domain_boundary(ax, domain_boundary, fill=True)

    # Extract positions for each source
    for src_idx in source_indices:
        x_traj = np.array([params[2*src_idx] for params in trajectory])
        y_traj = np.array([params[2*src_idx + 1] for params in trajectory])

        # Create line segments colored by iteration
        points = np.column_stack([x_traj, y_traj]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Color by iteration (light to dark)
        colors = np.linspace(0.2, 1.0, len(segments))
        cmap = plt.cm.get_cmap(COLORMAPS['convergence'])

        lc = LineCollection(segments, cmap=cmap, linewidth=1.5, alpha=0.8)
        lc.set_array(colors)
        ax.add_collection(lc)

        # Mark initial position
        ax.scatter(x_traj[0], y_traj[0], c=COLORS['initial'],
                  s=80, marker='o', zorder=5, edgecolors='white', linewidths=1)

        # Mark final position
        ax.scatter(x_traj[-1], y_traj[-1], c=COLORS['final'],
                  s=100, marker='o', zorder=6, edgecolors='white', linewidths=2)

        # Direction arrows
        if show_arrows and n_iters > 10:
            arrow_indices = np.linspace(0, n_iters-2, min(5, n_iters//2), dtype=int)
            for idx in arrow_indices:
                dx = x_traj[idx+1] - x_traj[idx]
                dy = y_traj[idx+1] - y_traj[idx]
                if np.sqrt(dx**2 + dy**2) > 0.01:  # Only if movement is significant
                    ax.annotate('', xy=(x_traj[idx+1], y_traj[idx+1]),
                               xytext=(x_traj[idx], y_traj[idx]),
                               arrowprops=dict(arrowstyle='->', color=cmap(colors[idx]),
                                             lw=1, alpha=0.6))

    # Add true sources
    if sources_true is not None:
        pos_true, int_true = sources_to_arrays(sources_true)
        for i, (pos, intensity) in enumerate(zip(pos_true, int_true)):
            if i in source_indices:
                ax.scatter(pos[0], pos[1], c='none', s=200, marker='*',
                          edgecolors='gold', linewidths=2, zorder=10)

    # Legend
    ax.scatter([], [], c=COLORS['initial'], s=80, marker='o', label='Initial')
    ax.scatter([], [], c=COLORS['final'], s=100, marker='o', label='Final')
    if sources_true:
        ax.scatter([], [], c='none', s=200, marker='*', edgecolors='gold',
                  linewidths=2, label='True')
    ax.legend(loc='upper right', fontsize=8)

    set_domain_axes(ax, domain_boundary)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)

    return ax


def plot_parameter_evolution(trajectory: List[np.ndarray],
                             n_sources: int,
                             sources_true: List[Tuple] = None,
                             param_type: str = 'positions',
                             title: str = None) -> plt.Figure:
    """
    Plot individual parameter values vs iteration.

    Parameters
    ----------
    trajectory : list of ndarray
        Parameter vectors at each iteration
    n_sources : int
        Number of sources
    sources_true : list, optional
        True sources for reference
    param_type : str
        'positions' (x, y for each source) or 'intensities' (q for each)
    title : str, optional
        Figure title

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure
    """
    n_iters = len(trajectory)
    iterations = np.arange(n_iters)

    if param_type == 'positions':
        n_panels = 2 * n_sources
        fig, axes = plt.subplots(n_sources, 2, figsize=(12, 3*n_sources), squeeze=False)

        if sources_true:
            pos_true, _ = sources_to_arrays(sources_true)

        for i in range(n_sources):
            # X coordinate
            x_vals = [params[2*i] for params in trajectory]
            axes[i, 0].plot(iterations, x_vals, color=COLORS['trajectory'], linewidth=2)
            axes[i, 0].set_ylabel(f'Source {i+1}: x')
            if sources_true and i < len(pos_true):
                axes[i, 0].axhline(pos_true[i, 0], color=COLORS['success'],
                                  linestyle='--', label='True')
            axes[i, 0].grid(True, alpha=0.3)

            # Y coordinate
            y_vals = [params[2*i + 1] for params in trajectory]
            axes[i, 1].plot(iterations, y_vals, color=COLORS['trajectory'], linewidth=2)
            axes[i, 1].set_ylabel(f'Source {i+1}: y')
            if sources_true and i < len(pos_true):
                axes[i, 1].axhline(pos_true[i, 1], color=COLORS['success'],
                                  linestyle='--', label='True')
            axes[i, 1].grid(True, alpha=0.3)

        axes[-1, 0].set_xlabel('Iteration')
        axes[-1, 1].set_xlabel('Iteration')

        if title is None:
            title = "Position Evolution"

    else:  # intensities
        fig, axes = plt.subplots(n_sources, 1, figsize=(10, 3*n_sources), squeeze=False)

        if sources_true:
            _, int_true = sources_to_arrays(sources_true)

        for i in range(n_sources):
            q_vals = [params[2*n_sources + i] for params in trajectory]
            axes[i, 0].plot(iterations, q_vals, color=COLORS['trajectory'], linewidth=2)
            axes[i, 0].set_ylabel(f'Source {i+1}: q')
            if sources_true and i < len(int_true):
                axes[i, 0].axhline(int_true[i], color=COLORS['success'],
                                  linestyle='--', label='True')
            axes[i, 0].axhline(0, color=COLORS['grid'], linestyle='-', alpha=0.5)
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].legend(loc='upper right')

        axes[-1, 0].set_xlabel('Iteration')

        if title is None:
            title = "Intensity Evolution"

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_optimization_summary(trajectory: List[np.ndarray],
                              history: List[float],
                              n_sources: int,
                              domain_boundary: np.ndarray,
                              sources_true: List[Tuple] = None,
                              title: str = "Optimization Summary") -> plt.Figure:
    """
    Combined optimization summary figure (2x2 layout).

    Panels:
    - Top left: Convergence
    - Top right: Source trajectories
    - Bottom left: Position evolution
    - Bottom right: Intensity evolution

    Parameters
    ----------
    trajectory : list of ndarray
        Parameter vectors at each iteration
    history : list of float
        Objective values
    n_sources : int
        Number of sources
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
    fig = plt.figure(figsize=FIGSIZE['dashboard'])

    # Panel 1: Convergence
    ax1 = fig.add_subplot(2, 2, 1)
    plot_convergence(history, title='Convergence', ax=ax1)

    # Panel 2: Source trajectories
    ax2 = fig.add_subplot(2, 2, 2)
    plot_source_trajectory(trajectory, n_sources, domain_boundary,
                          sources_true=sources_true,
                          title='Source Trajectories', ax=ax2)

    # Panel 3: Position evolution (simplified - just first 2 sources)
    ax3 = fig.add_subplot(2, 2, 3)
    n_show = min(2, n_sources)
    colors = plt.cm.tab10(np.linspace(0, 1, n_show*2))
    iterations = np.arange(len(trajectory))

    for i in range(n_show):
        x_vals = [params[2*i] for params in trajectory]
        y_vals = [params[2*i + 1] for params in trajectory]
        ax3.plot(iterations, x_vals, color=colors[2*i], label=f'x{i+1}', linewidth=1.5)
        ax3.plot(iterations, y_vals, color=colors[2*i+1], linestyle='--',
                label=f'y{i+1}', linewidth=1.5)

    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Position')
    ax3.set_title(f'Position Evolution (Sources 1-{n_show})')
    ax3.legend(loc='upper right', fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Intensity evolution
    ax4 = fig.add_subplot(2, 2, 4)
    colors = plt.cm.tab10(np.linspace(0, 1, n_sources))

    for i in range(n_sources):
        q_vals = [params[2*n_sources + i] for params in trajectory]
        ax4.plot(iterations, q_vals, color=colors[i], label=f'q{i+1}', linewidth=1.5)

    ax4.axhline(0, color=COLORS['grid'], linestyle='-', alpha=0.5)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Intensity')
    ax4.set_title('Intensity Evolution')
    ax4.legend(loc='upper right', fontsize=7, ncol=2)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_gradient_norms(gradients: List[np.ndarray],
                        labels: List[str] = None,
                        log_scale: bool = True,
                        title: str = "Gradient Norms",
                        ax: plt.Axes = None) -> plt.Axes:
    """
    Plot gradient norms vs iteration.

    Useful for diagnosing convergence issues.

    Parameters
    ----------
    gradients : list of ndarray
        Gradient vectors at each iteration
    labels : list of str, optional
        Component labels (e.g., ['positions', 'intensities'])
    log_scale : bool
        Use log scale
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

    iterations = np.arange(len(gradients))
    norms = np.array([np.linalg.norm(g) for g in gradients])

    ax.plot(iterations, norms, color=COLORS['trajectory'], linewidth=2, label='Total')

    if labels is not None and len(gradients) > 0:
        # Assume gradients can be split into components
        n_params = len(gradients[0])
        n_components = len(labels)
        params_per_component = n_params // n_components

        colors = plt.cm.tab10(np.linspace(0, 1, n_components))
        for i, (label, color) in enumerate(zip(labels, colors)):
            start = i * params_per_component
            end = (i + 1) * params_per_component
            component_norms = np.array([np.linalg.norm(g[start:end]) for g in gradients])
            ax.plot(iterations, component_norms, color=color, linewidth=1,
                   alpha=0.7, label=label)

    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient Norm')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax
