"""
Utility Functions for Inverse Source Localization
==================================================

Plotting, error computation, and analysis utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def plot_sources(sources: List[Tuple[Tuple[float, float], float]], 
                 ax: plt.Axes = None, 
                 title: str = "Point Sources",
                 show_boundary: bool = True,
                 domain: str = 'disk') -> plt.Axes:
    """
    Plot point sources on the domain.
    
    Parameters
    ----------
    sources : list of ((x, y), intensity)
        Point sources
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str
        Plot title
    show_boundary : bool
        Draw domain boundary
    domain : str
        'disk' or 'custom'
        
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Draw boundary
    if show_boundary:
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)
    
    # Plot sources
    for (x, y), q in sources:
        color = 'red' if q > 0 else 'blue'
        size = 100 * np.abs(q)
        ax.scatter(x, y, c=color, s=size, alpha=0.7, edgecolors='black')
        ax.annotate(f'{q:.2f}', (x, y), textcoords="offset points", 
                   xytext=(5, 5), fontsize=8)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_boundary_data(theta: np.ndarray, u: np.ndarray,
                       ax: plt.Axes = None,
                       title: str = "Boundary Data",
                       label: str = None) -> plt.Axes:
    """
    Plot boundary measurements as a function of angle.
    
    Parameters
    ----------
    theta : array
        Boundary angles
    u : array
        Solution values on boundary
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str
        Plot title
    label : str, optional
        Legend label
        
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(theta, u, '-', label=label)
    ax.set_xlabel('θ (radians)')
    ax.set_ylabel('u(θ)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if label:
        ax.legend()
    
    return ax


def plot_recovery_comparison(true_sources: List[Tuple[Tuple[float, float], float]],
                             positions: np.ndarray,
                             q_recovered: np.ndarray,
                             threshold: float = None,
                             ax: plt.Axes = None,
                             title: str = "Source Recovery") -> plt.Axes:
    """
    Compare true sources with recovered source distribution.
    
    Parameters
    ----------
    true_sources : list
        True point sources
    positions : array, shape (n, 2)
        Candidate positions
    q_recovered : array, shape (n,)
        Recovered intensities
    threshold : float, optional
        Threshold for significant sources (auto if None)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str
        Plot title
        
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    if threshold is None:
        threshold = 0.1 * np.abs(q_recovered).max()
    
    # Draw boundary
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)
    
    # Plot recovered distribution
    significant = np.abs(q_recovered) > threshold
    
    # Background: all candidates (faint)
    ax.scatter(positions[:, 0], positions[:, 1], c='gray', s=5, alpha=0.2)
    
    # Significant recovered sources
    for i in np.where(significant)[0]:
        color = 'red' if q_recovered[i] > 0 else 'blue'
        size = 50 * np.abs(q_recovered[i]) / np.abs(q_recovered).max()
        ax.scatter(positions[i, 0], positions[i, 1], c=color, s=size, 
                  alpha=0.6, marker='o')
    
    # True sources (stars)
    for (x, y), q in true_sources:
        color = 'darkred' if q > 0 else 'darkblue'
        ax.scatter(x, y, c=color, s=200, marker='*', edgecolors='black', linewidths=1)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Legend
    ax.scatter([], [], c='darkred', s=100, marker='*', label='True (+)')
    ax.scatter([], [], c='darkblue', s=100, marker='*', label='True (-)')
    ax.scatter([], [], c='red', s=50, marker='o', alpha=0.6, label='Recovered (+)')
    ax.scatter([], [], c='blue', s=50, marker='o', alpha=0.6, label='Recovered (-)')
    ax.legend(loc='upper right')
    
    return ax


def compute_source_error(true_sources: List[Tuple[Tuple[float, float], float]],
                         positions: np.ndarray,
                         q_recovered: np.ndarray,
                         threshold: float = None) -> dict:
    """
    Compute error metrics for source recovery.
    
    Parameters
    ----------
    true_sources : list
        True point sources
    positions : array
        Candidate positions
    q_recovered : array
        Recovered intensities
    threshold : float, optional
        Threshold for significant sources
        
    Returns
    -------
    metrics : dict
        Error metrics including:
        - n_true: number of true sources
        - n_recovered: number of significant recovered sources
        - position_errors: distances to nearest true source
        - intensity_errors: intensity differences
        - total_intensity_error: |Σq_true - Σq_recovered|
    """
    if threshold is None:
        threshold = 0.1 * np.abs(q_recovered).max()
    
    significant = np.where(np.abs(q_recovered) > threshold)[0]
    
    # Extract true source info
    true_pos = np.array([s[0] for s in true_sources])
    true_q = np.array([s[1] for s in true_sources])
    
    # For each recovered source, find nearest true source
    position_errors = []
    intensity_errors = []
    
    for i in significant:
        pos = positions[i]
        q = q_recovered[i]
        
        # Distance to each true source
        dists = np.sqrt(np.sum((true_pos - pos)**2, axis=1))
        nearest = np.argmin(dists)
        
        position_errors.append(dists[nearest])
        intensity_errors.append(np.abs(q - true_q[nearest]))
    
    return {
        'n_true': len(true_sources),
        'n_recovered': len(significant),
        'position_errors': np.array(position_errors),
        'intensity_errors': np.array(intensity_errors),
        'mean_position_error': np.mean(position_errors) if position_errors else np.inf,
        'mean_intensity_error': np.mean(intensity_errors) if intensity_errors else np.inf,
        'total_intensity_error': np.abs(true_q.sum() - q_recovered.sum()),
    }


def l_curve_analysis(G: np.ndarray, u: np.ndarray,
                     alpha_range: np.ndarray = None,
                     method: str = 'l1',
                     **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute L-curve for regularization parameter selection.
    
    The L-curve plots ||Gq - u|| vs ||q|| (or regularization term)
    for varying α. The optimal α is at the "corner" of the L.
    
    Parameters
    ----------
    G : array
        Forward operator
    u : array
        Measurements
    alpha_range : array, optional
        Regularization parameters to try
    method : str
        'l1' or 'l2'
    **kwargs
        Passed to solver
        
    Returns
    -------
    alphas : array
        Regularization parameters
    residuals : array
        ||Gq - u||
    regularization : array
        ||q|| or ||q||₁
    """
    if alpha_range is None:
        alpha_range = np.logspace(-6, 0, 20)
    
    try:
        from .regularization import solve_l1, solve_l2
    except ImportError:
        from regularization import solve_l1, solve_l2
    
    residuals = []
    regularization = []
    
    for alpha in alpha_range:
        if method == 'l1':
            q = solve_l1(G, u, alpha, **kwargs)
            reg = np.sum(np.abs(q))
        else:
            q = solve_l2(G, u, alpha)
            reg = np.linalg.norm(q)
        
        residuals.append(np.linalg.norm(G @ q - u))
        regularization.append(reg)
    
    return alpha_range, np.array(residuals), np.array(regularization)


def plot_l_curve(residuals: np.ndarray, regularization: np.ndarray,
                 alphas: np.ndarray = None,
                 ax: plt.Axes = None,
                 title: str = "L-Curve") -> plt.Axes:
    """
    Plot L-curve and optionally mark the corner.
    
    Parameters
    ----------
    residuals : array
        ||Gq - u||
    regularization : array
        Regularization term
    alphas : array, optional
        Regularization parameters (for annotation)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str
        Plot title
        
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.loglog(residuals, regularization, 'b.-', markersize=8)
    
    if alphas is not None:
        # Annotate some points
        for i in [0, len(alphas)//4, len(alphas)//2, 3*len(alphas)//4, -1]:
            ax.annotate(f'α={alphas[i]:.1e}', 
                       (residuals[i], regularization[i]),
                       textcoords="offset points", xytext=(5, 5), fontsize=7)
    
    ax.set_xlabel('Residual ||Gq - u||')
    ax.set_ylabel('Regularization ||q||')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax


def create_test_sources(n_sources: int = 4, seed: int = 42,
                        radius_range: Tuple[float, float] = (0.2, 0.7)) -> List[Tuple[Tuple[float, float], float]]:
    """
    Create random test sources satisfying compatibility condition.
    
    Parameters
    ----------
    n_sources : int
        Number of sources
    seed : int
        Random seed
    radius_range : tuple
        (min_radius, max_radius) for source positions
        
    Returns
    -------
    sources : list of ((x, y), intensity)
        Test sources with Σq = 0
    """
    np.random.seed(seed)
    
    sources = []
    total_q = 0
    
    for i in range(n_sources - 1):
        r = radius_range[0] + (radius_range[1] - radius_range[0]) * np.random.rand()
        theta = 2 * np.pi * np.random.rand()
        x, y = r * np.cos(theta), r * np.sin(theta)
        q = np.random.randn()
        sources.append(((x, y), q))
        total_q += q
    
    # Last source ensures Σq = 0
    r = radius_range[0] + (radius_range[1] - radius_range[0]) * np.random.rand()
    theta = 2 * np.pi * np.random.rand()
    x, y = r * np.cos(theta), r * np.sin(theta)
    sources.append(((x, y), -total_q))
    
    return sources
