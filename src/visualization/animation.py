"""
Animated Visualizations
=======================

Animated visualizations for optimization trajectories and source movement.
Useful for understanding optimization dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from typing import List, Tuple, Optional

from .config import (COLORS, FIGSIZE, ANIMATION_INTERVAL_MS, ANIMATION_TRAIL_LENGTH,
                     get_source_color, get_trajectory_color)
from .utils import get_domain_boundary, sources_to_arrays, add_domain_boundary, format_axes_pi


def animate_optimization(trajectory: List[np.ndarray],
                         objective_history: List[float],
                         n_sources: int,
                         domain_boundary: np.ndarray,
                         sources_true: List[Tuple] = None,
                         u_measured: np.ndarray = None,
                         theta: np.ndarray = None,
                         forward_solver=None,
                         interval: int = ANIMATION_INTERVAL_MS,
                         save_path: str = None) -> FuncAnimation:
    """
    Full optimization animation (2x2 layout).

    Panels:
    - Top left: Source positions on domain
    - Top right: Boundary fit
    - Bottom left: Convergence curve
    - Bottom right: Info panel

    Parameters
    ----------
    trajectory : list of ndarray
        Parameter vectors at each iteration
    objective_history : list of float
        Objective values at each iteration
    n_sources : int
        Number of sources
    domain_boundary : ndarray
        Domain boundary points
    sources_true : list, optional
        True sources for comparison
    u_measured : ndarray, optional
        Measured boundary data
    theta : ndarray, optional
        Boundary angles
    forward_solver : optional
        Forward solver for computing boundary fit
    interval : int
        Milliseconds between frames
    save_path : str, optional
        Path to save animation (.gif or .mp4)

    Returns
    -------
    anim : FuncAnimation
        The animation object
    """
    n_frames = len(trajectory)

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE['animation'])

    ax_sources = axes[0, 0]
    ax_fit = axes[0, 1]
    ax_conv = axes[1, 0]
    ax_info = axes[1, 1]

    # Setup source positions panel
    add_domain_boundary(ax_sources, domain_boundary, fill=True)

    if sources_true is not None:
        pos_true, int_true = sources_to_arrays(sources_true)
        for pos, intensity in zip(pos_true, int_true):
            ax_sources.scatter(pos[0], pos[1], c='none', s=200, marker='*',
                              edgecolors='gold', linewidths=2, zorder=10)

    # Initialize source markers
    source_scatters = []
    trail_lines = []
    for i in range(n_sources):
        scatter = ax_sources.scatter([], [], s=100, zorder=5)
        source_scatters.append(scatter)
        line, = ax_sources.plot([], [], '-', linewidth=1, alpha=0.5)
        trail_lines.append(line)

    x_range = [domain_boundary[:, 0].min(), domain_boundary[:, 0].max()]
    y_range = [domain_boundary[:, 1].min(), domain_boundary[:, 1].max()]
    ax_sources.set_xlim(x_range[0] - 0.1, x_range[1] + 0.1)
    ax_sources.set_ylim(y_range[0] - 0.1, y_range[1] + 0.1)
    ax_sources.set_aspect('equal')
    ax_sources.set_title('Source Positions')

    # Setup boundary fit panel
    if theta is not None and u_measured is not None:
        sort_idx = np.argsort(theta)
        ax_fit.plot(theta[sort_idx], u_measured[sort_idx], color=COLORS['measured'],
                   linewidth=2, label='Measured')
        fit_line, = ax_fit.plot([], [], color=COLORS['recovered'],
                               linewidth=2, linestyle='--', label='Current')
        ax_fit.set_xlim(0, 2*np.pi)
        format_axes_pi(ax_fit, 'x')
        ax_fit.legend(fontsize=8)
    else:
        fit_line = None
    ax_fit.set_title('Boundary Fit')
    ax_fit.grid(True, alpha=0.3)

    # Setup convergence panel
    conv_line, = ax_conv.plot([], [], color=COLORS['trajectory'], linewidth=2)
    ax_conv.set_xlim(0, n_frames)
    ax_conv.set_ylim(min(objective_history) * 0.9, objective_history[0] * 1.1)
    ax_conv.set_yscale('log')
    ax_conv.set_xlabel('Iteration')
    ax_conv.set_ylabel('Objective')
    ax_conv.set_title('Convergence')
    ax_conv.grid(True, alpha=0.3)
    conv_marker = ax_conv.scatter([], [], c=COLORS['final'], s=50, zorder=5)

    # Setup info panel
    ax_info.axis('off')
    info_text = ax_info.text(0.5, 0.5, '', transform=ax_info.transAxes,
                             ha='center', va='center', fontsize=11,
                             fontfamily='monospace',
                             bbox=dict(boxstyle='round', facecolor='white',
                                      edgecolor=COLORS['grid']))

    def init():
        """Initialize animation."""
        for scatter in source_scatters:
            scatter.set_offsets(np.empty((0, 2)))
        for line in trail_lines:
            line.set_data([], [])
        if fit_line is not None:
            fit_line.set_data([], [])
        conv_line.set_data([], [])
        conv_marker.set_offsets(np.empty((0, 2)))
        info_text.set_text('')
        return source_scatters + trail_lines + [fit_line, conv_line, conv_marker, info_text]

    def update(frame):
        """Update animation for given frame."""
        params = trajectory[frame]

        # Update source positions
        for i in range(n_sources):
            x, y = params[2*i], params[2*i + 1]
            q = params[2*n_sources + i] if len(params) > 2*n_sources + i else 1.0
            color = get_source_color(q)

            source_scatters[i].set_offsets([[x, y]])
            source_scatters[i].set_color(color)

            # Trail
            trail_start = max(0, frame - ANIMATION_TRAIL_LENGTH)
            trail_x = [trajectory[f][2*i] for f in range(trail_start, frame + 1)]
            trail_y = [trajectory[f][2*i + 1] for f in range(trail_start, frame + 1)]
            trail_lines[i].set_data(trail_x, trail_y)
            trail_lines[i].set_color(color)

        # Update boundary fit
        if fit_line is not None and forward_solver is not None and theta is not None:
            sources_current = [((params[2*i], params[2*i + 1]),
                               params[2*n_sources + i] if len(params) > 2*n_sources + i else 1.0)
                              for i in range(n_sources)]
            try:
                u_current = forward_solver.solve(sources_current)
                sort_idx = np.argsort(theta)
                fit_line.set_data(theta[sort_idx], u_current[sort_idx])
            except:
                pass

        # Update convergence
        conv_line.set_data(range(frame + 1), objective_history[:frame + 1])
        conv_marker.set_offsets([[frame, objective_history[frame]]])

        # Update info
        info = f"Iteration: {frame}/{n_frames-1}\n"
        info += f"Objective: {objective_history[frame]:.2e}\n"
        if sources_true is not None:
            # Compute position error
            pos_current = np.array([[params[2*i], params[2*i + 1]] for i in range(n_sources)])
            pos_true_arr, _ = sources_to_arrays(sources_true)
            if len(pos_true_arr) == len(pos_current):
                dists = np.linalg.norm(pos_current - pos_true_arr, axis=1)
                info += f"Mean pos error: {np.mean(dists):.4f}"
        info_text.set_text(info)

        return source_scatters + trail_lines + [fit_line, conv_line, conv_marker, info_text]

    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames,
                        interval=interval, blit=True)

    plt.tight_layout()

    if save_path is not None:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000//interval)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=1000//interval)
        else:
            anim.save(save_path + '.gif', writer='pillow', fps=1000//interval)

    return anim


def animate_source_movement(trajectory: List[np.ndarray],
                            n_sources: int,
                            domain_boundary: np.ndarray,
                            sources_true: List[Tuple] = None,
                            trail_length: int = ANIMATION_TRAIL_LENGTH,
                            interval: int = 50,
                            save_path: str = None) -> FuncAnimation:
    """
    Simple animation of source positions only.

    Parameters
    ----------
    trajectory : list of ndarray
        Parameter vectors at each iteration
    n_sources : int
        Number of sources
    domain_boundary : ndarray
        Domain boundary
    sources_true : list, optional
        True sources for comparison
    trail_length : int
        Length of fading trail
    interval : int
        Milliseconds between frames
    save_path : str, optional
        Path to save animation

    Returns
    -------
    anim : FuncAnimation
        The animation object
    """
    n_frames = len(trajectory)

    fig, ax = plt.subplots(figsize=FIGSIZE['square'])

    # Domain
    add_domain_boundary(ax, domain_boundary, fill=True)

    # True sources
    if sources_true is not None:
        pos_true, int_true = sources_to_arrays(sources_true)
        for pos, intensity in zip(pos_true, int_true):
            ax.scatter(pos[0], pos[1], c='none', s=200, marker='*',
                      edgecolors='gold', linewidths=2, zorder=10)

    # Initialize
    source_scatters = []
    trail_lines = []
    colors_default = plt.cm.tab10(np.linspace(0, 1, n_sources))

    for i in range(n_sources):
        scatter = ax.scatter([], [], s=120, zorder=5, edgecolors='white', linewidths=2)
        source_scatters.append(scatter)

        line, = ax.plot([], [], '-', linewidth=2, alpha=0.6)
        trail_lines.append(line)

    # Axis limits
    margin = 0.15
    x_range = domain_boundary[:, 0].max() - domain_boundary[:, 0].min()
    y_range = domain_boundary[:, 1].max() - domain_boundary[:, 1].min()
    ax.set_xlim(domain_boundary[:, 0].min() - margin * x_range,
               domain_boundary[:, 0].max() + margin * x_range)
    ax.set_ylim(domain_boundary[:, 1].min() - margin * y_range,
               domain_boundary[:, 1].max() + margin * y_range)
    ax.set_aspect('equal')

    # Iteration counter
    iter_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top',
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    def init():
        for scatter in source_scatters:
            scatter.set_offsets(np.empty((0, 2)))
        for line in trail_lines:
            line.set_data([], [])
        iter_text.set_text('')
        return source_scatters + trail_lines + [iter_text]

    def update(frame):
        params = trajectory[frame]

        for i in range(n_sources):
            x, y = params[2*i], params[2*i + 1]

            # Get intensity for color (if available)
            if len(params) > 2*n_sources + i:
                q = params[2*n_sources + i]
                color = get_source_color(q)
            else:
                color = colors_default[i]

            source_scatters[i].set_offsets([[x, y]])
            source_scatters[i].set_color(color)

            # Trail
            trail_start = max(0, frame - trail_length)
            trail_x = [trajectory[f][2*i] for f in range(trail_start, frame + 1)]
            trail_y = [trajectory[f][2*i + 1] for f in range(trail_start, frame + 1)]
            trail_lines[i].set_data(trail_x, trail_y)
            trail_lines[i].set_color(color)

        iter_text.set_text(f'Iteration: {frame}')

        return source_scatters + trail_lines + [iter_text]

    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames,
                        interval=interval, blit=True)

    ax.set_title('Source Movement')

    if save_path is not None:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000//interval)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=1000//interval)
        else:
            anim.save(save_path + '.gif', writer='pillow', fps=1000//interval)

    return anim


def animate_boundary_fit(trajectory: List[np.ndarray],
                         n_sources: int,
                         forward_solver,
                         u_measured: np.ndarray,
                         theta: np.ndarray,
                         interval: int = 100,
                         save_path: str = None) -> FuncAnimation:
    """
    Animate how the boundary fit improves during optimization.

    Parameters
    ----------
    trajectory : list of ndarray
        Parameter vectors
    n_sources : int
        Number of sources
    forward_solver
        Forward solver to compute boundary values
    u_measured : ndarray
        Measured boundary data
    theta : ndarray
        Boundary angles
    interval : int
        Milliseconds between frames
    save_path : str, optional
        Path to save animation

    Returns
    -------
    anim : FuncAnimation
        The animation object
    """
    n_frames = len(trajectory)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])

    sort_idx = np.argsort(theta)
    theta_sorted = theta[sort_idx]
    u_meas_sorted = u_measured[sort_idx]

    # Main panel
    ax1.plot(theta_sorted, u_meas_sorted, color=COLORS['measured'],
            linewidth=2, label='Measured')
    fit_line, = ax1.plot([], [], color=COLORS['recovered'],
                        linewidth=2, linestyle='--', label='Current fit')
    ax1.fill_between(theta_sorted, u_meas_sorted, u_meas_sorted,
                    alpha=0.2, color=COLORS['residual'])
    fill_collection = ax1.collections[-1]

    ax1.set_xlim(0, 2*np.pi)
    format_axes_pi(ax1, 'x')
    ax1.set_ylabel('u(θ)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Residual panel
    res_line, = ax2.plot([], [], color=COLORS['residual'], linewidth=1.5)
    ax2.axhline(0, color=COLORS['grid'], linewidth=1)
    ax2.set_xlim(0, 2*np.pi)
    format_axes_pi(ax2, 'x')
    ax2.set_xlabel('θ')
    ax2.set_ylabel('Residual')
    ax2.grid(True, alpha=0.3)

    # RMS text
    rms_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, va='top',
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    def init():
        fit_line.set_data([], [])
        res_line.set_data([], [])
        rms_text.set_text('')
        return [fit_line, res_line, rms_text]

    def update(frame):
        params = trajectory[frame]

        # Build sources
        sources = [((params[2*i], params[2*i + 1]),
                   params[2*n_sources + i] if len(params) > 2*n_sources + i else 1.0)
                  for i in range(n_sources)]

        try:
            u_current = forward_solver.solve(sources)
            u_curr_sorted = u_current[sort_idx]

            fit_line.set_data(theta_sorted, u_curr_sorted)

            residual = u_meas_sorted - u_curr_sorted
            res_line.set_data(theta_sorted, residual)

            rms = np.sqrt(np.mean(residual**2))
            rms_text.set_text(f'Iter {frame} | RMS = {rms:.2e}')

            # Update y limits for residual
            max_res = max(abs(residual.min()), abs(residual.max())) * 1.2
            if max_res > 0:
                ax2.set_ylim(-max_res, max_res)

        except Exception as e:
            rms_text.set_text(f'Iter {frame} | Error')

        return [fit_line, res_line, rms_text]

    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames,
                        interval=interval, blit=True)

    ax1.set_title('Boundary Fit Animation')
    plt.tight_layout()

    if save_path is not None:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000//interval)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=1000//interval)

    return anim


def create_optimization_video(trajectory: List[np.ndarray],
                              objective_history: List[float],
                              n_sources: int,
                              domain_boundary: np.ndarray,
                              sources_true: List[Tuple] = None,
                              save_path: str = 'optimization.gif',
                              fps: int = 10,
                              subsample: int = 1) -> None:
    """
    Create and save optimization video.

    Convenience function that creates animation and saves it.

    Parameters
    ----------
    trajectory : list of ndarray
        Parameter vectors
    objective_history : list of float
        Objective values
    n_sources : int
        Number of sources
    domain_boundary : ndarray
        Domain boundary
    sources_true : list, optional
        True sources
    save_path : str
        Output path (.gif or .mp4)
    fps : int
        Frames per second
    subsample : int
        Use every nth frame (to reduce file size)
    """
    # Subsample if requested
    if subsample > 1:
        trajectory = trajectory[::subsample]
        objective_history = objective_history[::subsample]

    interval = 1000 // fps

    anim = animate_optimization(
        trajectory=trajectory,
        objective_history=objective_history,
        n_sources=n_sources,
        domain_boundary=domain_boundary,
        sources_true=sources_true,
        interval=interval,
        save_path=save_path
    )

    plt.close()
    print(f"Animation saved to {save_path}")
