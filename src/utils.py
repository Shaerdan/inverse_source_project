"""
Utilities for Inverse Source Localization
==========================================

Plotting, analysis, and helper functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import List, Tuple, Optional
from dataclasses import dataclass


def plot_sources(sources_true: List[Tuple[Tuple[float, float], float]],
                 sources_recovered: Optional[List] = None,
                 ax: Optional[plt.Axes] = None,
                 title: str = "Source Locations",
                 domain_radius: float = 1.0) -> plt.Axes:
    """
    Plot source locations on the domain.
    
    Parameters
    ----------
    sources_true : list of ((x, y), q) tuples
        True source locations and intensities
    sources_recovered : list, optional
        Recovered sources (Source objects or tuples)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    title : str
        Plot title
    domain_radius : float
        Radius of the domain circle to draw
        
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw domain boundary
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(domain_radius * np.cos(theta), domain_radius * np.sin(theta), 
            'k-', linewidth=2, label='Domain')
    
    # Plot true sources
    for i, ((x, y), q) in enumerate(sources_true):
        color = 'red' if q > 0 else 'blue'
        marker = 'o'
        ax.plot(x, y, marker, color=color, markersize=15, 
                markerfacecolor='none', markeredgewidth=2,
                label=f'True: q={q:+.2f}' if i < 4 else None)
    
    # Plot recovered sources if provided
    if sources_recovered is not None:
        for i, s in enumerate(sources_recovered):
            # Handle both Source objects and tuples
            if hasattr(s, 'x'):
                x, y, q = s.x, s.y, s.intensity
            else:
                (x, y), q = s
            
            color = 'red' if q > 0 else 'blue'
            ax.plot(x, y, '+', color=color, markersize=15, markeredgewidth=3,
                    label=f'Rec: q={q:+.2f}' if i < 4 else None)
    
    ax.set_aspect('equal')
    ax.set_xlim(-domain_radius * 1.2, domain_radius * 1.2)
    ax.set_ylim(-domain_radius * 1.2, domain_radius * 1.2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_boundary_data(theta: np.ndarray, 
                       u_measured: np.ndarray,
                       u_recovered: Optional[np.ndarray] = None,
                       ax: Optional[plt.Axes] = None,
                       title: str = "Boundary Data") -> plt.Axes:
    """
    Plot boundary measurements and fit.
    
    Parameters
    ----------
    theta : array
        Angular positions
    u_measured : array
        Measured boundary values
    u_recovered : array, optional
        Recovered/fitted boundary values
    ax : matplotlib.axes.Axes, optional
    title : str
        
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(theta, u_measured, 'b-', linewidth=2, label='Measured')
    
    if u_recovered is not None:
        ax.plot(theta, u_recovered, 'r--', linewidth=2, label='Recovered')
        residual = np.linalg.norm(u_measured - u_recovered)
        ax.set_title(f'{title} (residual = {residual:.4e})')
    else:
        ax.set_title(title)
    
    ax.set_xlabel('θ')
    ax.set_ylabel('u')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_recovery_comparison(sources_true, sources_recovered, 
                             theta, u_measured, u_recovered,
                             figsize: Tuple[float, float] = (14, 5)):
    """
    Create a combined plot showing source locations and boundary fit.
    
    Parameters
    ----------
    sources_true : list of ((x, y), q) tuples
    sources_recovered : list of Source objects or tuples
    theta : array
        Boundary angles
    u_measured : array
        Measured boundary values
    u_recovered : array
        Recovered boundary values
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    plot_sources(sources_true, sources_recovered, ax=axes[0], 
                title='Source Locations: True (○) vs Recovered (+)')
    plot_boundary_data(theta, u_measured, u_recovered, ax=axes[1],
                      title='Boundary Fit')
    
    plt.tight_layout()
    return fig


def compute_source_error(sources_true: List[Tuple[Tuple[float, float], float]],
                         sources_recovered: List) -> dict:
    """
    Compute error metrics between true and recovered sources.
    
    Uses Hungarian algorithm to match sources optimally.
    
    Parameters
    ----------
    sources_true : list of ((x, y), q) tuples
    sources_recovered : list of Source objects or tuples
    
    Returns
    -------
    metrics : dict
        'position_rmse': RMS position error
        'intensity_rmse': RMS intensity error
        'position_max': Maximum position error
        'intensity_max': Maximum intensity error
    """
    from scipy.optimize import linear_sum_assignment
    
    n_true = len(sources_true)
    n_rec = len(sources_recovered)
    n = max(n_true, n_rec)
    
    # Build cost matrix (position distances)
    cost = np.full((n, n), 1e10)
    
    for i, ((x_t, y_t), q_t) in enumerate(sources_true):
        for j, s in enumerate(sources_recovered):
            if hasattr(s, 'x'):
                x_r, y_r = s.x, s.y
            else:
                (x_r, y_r), _ = s
            cost[i, j] = np.sqrt((x_t - x_r)**2 + (y_t - y_r)**2)
    
    # Optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Compute errors for matched pairs
    pos_errors = []
    int_errors = []
    
    for i, j in zip(row_ind, col_ind):
        if i < n_true and j < n_rec:
            (x_t, y_t), q_t = sources_true[i]
            s = sources_recovered[j]
            if hasattr(s, 'x'):
                x_r, y_r, q_r = s.x, s.y, s.intensity
            else:
                (x_r, y_r), q_r = s
            
            pos_errors.append(np.sqrt((x_t - x_r)**2 + (y_t - y_r)**2))
            int_errors.append(abs(q_t - q_r))
    
    return {
        'position_rmse': np.sqrt(np.mean(np.array(pos_errors)**2)) if pos_errors else np.inf,
        'intensity_rmse': np.sqrt(np.mean(np.array(int_errors)**2)) if int_errors else np.inf,
        'position_max': np.max(pos_errors) if pos_errors else np.inf,
        'intensity_max': np.max(int_errors) if int_errors else np.inf,
        'n_matched': len(pos_errors),
    }


def l_curve_analysis(solver, u_measured: np.ndarray, 
                     alphas: Optional[np.ndarray] = None,
                     method: str = 'l1') -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Perform L-curve analysis to select optimal regularization parameter.
    
    Parameters
    ----------
    solver : BEMLinearInverseSolver or similar
        Linear inverse solver with solve_l1/solve_l2 methods
    u_measured : array
        Measured boundary data
    alphas : array, optional
        Regularization parameters to test. If None, uses logspace.
    method : str
        'l1' or 'l2'
        
    Returns
    -------
    alphas : array
        Tested alpha values
    residuals : array
        ||Gq - u|| for each alpha
    reg_norms : array
        ||q|| or ||q||_1 for each alpha
    alpha_opt : float
        Optimal alpha (corner of L-curve)
    """
    if alphas is None:
        alphas = np.logspace(-6, -1, 30)
    
    residuals = []
    reg_norms = []
    
    for alpha in alphas:
        if method == 'l1':
            q = solver.solve_l1(u_measured, alpha=alpha)
            reg_norm = np.sum(np.abs(q))
        else:
            q = solver.solve_l2(u_measured, alpha=alpha)
            reg_norm = np.linalg.norm(q)
        
        residual = solver.compute_residual(q, u_measured) if hasattr(solver, 'compute_residual') else np.linalg.norm(solver.G @ q - (u_measured - np.mean(u_measured)))
        
        residuals.append(residual)
        reg_norms.append(reg_norm)
    
    residuals = np.array(residuals)
    reg_norms = np.array(reg_norms)
    
    # Find corner using curvature
    log_res = np.log(residuals + 1e-14)
    log_reg = np.log(reg_norms + 1e-14)
    
    # Compute curvature (simplified)
    dx = np.gradient(log_res)
    dy = np.gradient(log_reg)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-14)**1.5
    
    # Find maximum curvature (exclude endpoints)
    corner_idx = np.argmax(curvature[2:-2]) + 2
    alpha_opt = alphas[corner_idx]
    
    return alphas, residuals, reg_norms, alpha_opt


def plot_l_curve(alphas: np.ndarray, residuals: np.ndarray, 
                 reg_norms: np.ndarray, alpha_opt: float,
                 ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot L-curve with optimal alpha marked.
    
    Parameters
    ----------
    alphas : array
    residuals : array
    reg_norms : array
    alpha_opt : float
    ax : matplotlib.axes.Axes, optional
    
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.loglog(residuals, reg_norms, 'b.-', markersize=8)
    
    # Mark optimal point
    opt_idx = np.argmin(np.abs(alphas - alpha_opt))
    ax.loglog(residuals[opt_idx], reg_norms[opt_idx], 'ro', markersize=12,
              label=f'α* = {alpha_opt:.2e}')
    
    ax.set_xlabel('Residual ||Gq - u||')
    ax.set_ylabel('Regularization norm')
    ax.set_title('L-Curve')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    return ax


def create_test_sources(n_sources: int = 4, 
                        radius_range: Tuple[float, float] = (0.3, 0.7),
                        seed: Optional[int] = None) -> List[Tuple[Tuple[float, float], float]]:
    """
    Create random test sources satisfying compatibility Σqₖ = 0.
    
    Parameters
    ----------
    n_sources : int
        Number of sources (must be even for automatic compatibility)
    radius_range : tuple
        (min, max) radius for source positions
    seed : int, optional
        Random seed
        
    Returns
    -------
    sources : list of ((x, y), q) tuples
    """
    if seed is not None:
        np.random.seed(seed)
    
    sources = []
    r_min, r_max = radius_range
    
    for i in range(n_sources):
        angle = 2 * np.pi * i / n_sources + np.random.uniform(-0.2, 0.2)
        r = np.random.uniform(r_min, r_max)
        x, y = r * np.cos(angle), r * np.sin(angle)
        
        # Alternate signs for compatibility
        q = 1.0 if i % 2 == 0 else -1.0
        q *= np.random.uniform(0.8, 1.2)  # Add some variation
        
        sources.append(((x, y), q))
    
    # Adjust last source for exact compatibility
    total = sum(q for _, q in sources[:-1])
    (x, y), _ = sources[-1]
    sources[-1] = ((x, y), -total)
    
    return sources
