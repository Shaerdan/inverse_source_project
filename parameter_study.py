"""
Parameter Study Tools for Inverse Source Localization
======================================================

Tools for analyzing regularization parameter selection,
comparing methods, and studying noise sensitivity.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable
import matplotlib.pyplot as plt


@dataclass
class SweepResult:
    """Result from parameter sweep."""
    alphas: np.ndarray
    residuals: np.ndarray
    regularization: np.ndarray
    solutions: List[np.ndarray]
    optimal_alpha: float
    optimal_index: int


def parameter_sweep(G: np.ndarray, u: np.ndarray,
                    solver_func: Callable,
                    alpha_range: np.ndarray = None,
                    points: np.ndarray = None) -> SweepResult:
    """
    Sweep over regularization parameters.
    
    Parameters
    ----------
    G : array
        Forward operator
    u : array
        Measurements
    solver_func : callable
        Function(G, u, alpha) -> q
    alpha_range : array, optional
        Regularization parameters
    points : array, optional
        Source positions (for computing regularization term)
        
    Returns
    -------
    result : SweepResult
    """
    if alpha_range is None:
        alpha_range = np.logspace(-6, 0, 30)
    
    residuals = []
    regularization = []
    solutions = []
    
    u_centered = u - np.mean(u)
    
    for alpha in alpha_range:
        q = solver_func(G, u_centered, alpha)
        solutions.append(q)
        
        residuals.append(np.linalg.norm(G @ q - u_centered))
        regularization.append(np.sum(np.abs(q)))
    
    # Find L-curve corner
    optimal_idx = find_l_curve_corner(np.array(residuals), np.array(regularization))
    
    return SweepResult(
        alphas=alpha_range,
        residuals=np.array(residuals),
        regularization=np.array(regularization),
        solutions=solutions,
        optimal_alpha=alpha_range[optimal_idx],
        optimal_index=optimal_idx
    )


def find_l_curve_corner(residuals: np.ndarray, 
                        regularization: np.ndarray) -> int:
    """
    Find the corner of the L-curve using maximum curvature.
    
    Parameters
    ----------
    residuals : array
        Residual norms
    regularization : array
        Regularization terms
        
    Returns
    -------
    idx : int
        Index of optimal point (corner)
    """
    # Use log-log coordinates
    x = np.log(residuals + 1e-15)
    y = np.log(regularization + 1e-15)
    
    # Compute curvature using finite differences
    n = len(x)
    if n < 3:
        return 0
    
    # First derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    # Second derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Curvature: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-15)**1.5
    
    # Find maximum curvature (exclude endpoints)
    if n > 4:
        idx = np.argmax(curvature[2:-2]) + 2
    else:
        idx = np.argmax(curvature)
    
    return idx


def compare_methods(G: np.ndarray, u: np.ndarray,
                    points: np.ndarray,
                    methods: List[str] = None,
                    alpha: float = 1e-4) -> Dict[str, np.ndarray]:
    """
    Compare different regularization methods.
    
    Parameters
    ----------
    G : array
        Forward operator
    u : array
        Measurements
    points : array
        Source positions
    methods : list of str
        Methods to compare ('l1', 'l2', 'tv_admm')
    alpha : float
        Regularization parameter
        
    Returns
    -------
    results : dict
        Method name -> recovered q
    """
    try:
        from .regularization import solve_l1, solve_l2, solve_tv_admm, build_gradient_operator
    except ImportError:
        from regularization import solve_l1, solve_l2, solve_tv_admm, build_gradient_operator
    
    if methods is None:
        methods = ['l1', 'l2', 'tv_admm']
    
    u_centered = u - np.mean(u)
    results = {}
    
    for method in methods:
        if method == 'l1':
            results['l1'] = solve_l1(G, u_centered, alpha)
        elif method == 'l2':
            results['l2'] = solve_l2(G, u_centered, alpha)
        elif method == 'tv_admm':
            D, _ = build_gradient_operator(points)
            result = solve_tv_admm(G, u_centered, D, alpha)
            results['tv_admm'] = result.q
    
    return results


def plot_l_curve_comparison(sweeps: Dict[str, SweepResult],
                            ax: plt.Axes = None) -> plt.Axes:
    """
    Plot L-curves for multiple methods.
    
    Parameters
    ----------
    sweeps : dict
        Method name -> SweepResult
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.tab10.colors
    
    for i, (name, sweep) in enumerate(sweeps.items()):
        ax.loglog(sweep.residuals, sweep.regularization, 
                 '.-', color=colors[i], label=name, markersize=4)
        
        # Mark optimal point
        ax.scatter(sweep.residuals[sweep.optimal_index],
                  sweep.regularization[sweep.optimal_index],
                  s=100, color=colors[i], marker='*', zorder=10)
    
    ax.set_xlabel('Residual ||Gq - u||')
    ax.set_ylabel('Regularization ||q||₁')
    ax.set_title('L-Curve Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_method_comparison(positions: np.ndarray,
                           results: Dict[str, np.ndarray],
                           true_sources: List = None,
                           threshold_frac: float = 0.1) -> plt.Figure:
    """
    Plot recovered sources for different methods side by side.
    
    Parameters
    ----------
    positions : array
        Source candidate positions
    results : dict
        Method name -> recovered q
    true_sources : list, optional
        True sources for comparison
    threshold_frac : float
        Fraction of max for significance threshold
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
    
    if n_methods == 1:
        axes = [axes]
    
    for ax, (name, q) in zip(axes, results.items()):
        threshold = threshold_frac * np.abs(q).max()
        
        # Draw boundary
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)
        
        # Plot all candidates (faint)
        ax.scatter(positions[:, 0], positions[:, 1], c='gray', s=2, alpha=0.2)
        
        # Plot significant sources
        significant = np.abs(q) > threshold
        for i in np.where(significant)[0]:
            color = 'red' if q[i] > 0 else 'blue'
            size = 50 * np.abs(q[i]) / np.abs(q).max()
            ax.scatter(positions[i, 0], positions[i, 1], c=color, s=size, alpha=0.6)
        
        # Plot true sources
        if true_sources:
            for (x, y), intensity in true_sources:
                color = 'darkred' if intensity > 0 else 'darkblue'
                ax.scatter(x, y, c=color, s=150, marker='*', edgecolors='black')
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_title(f'{name}\n({np.sum(significant)} significant)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def noise_study(forward_func: Callable,
                inverse_func: Callable,
                true_sources: List,
                noise_levels: np.ndarray = None,
                n_trials: int = 10,
                n_boundary: int = 100) -> Dict:
    """
    Study effect of noise on source recovery.
    
    Parameters
    ----------
    forward_func : callable
        Forward solver
    inverse_func : callable
        Inverse solver (takes u, returns q)
    true_sources : list
        True source configuration
    noise_levels : array
        Noise standard deviations to test
    n_trials : int
        Number of random trials per noise level
    n_boundary : int
        Number of boundary points
        
    Returns
    -------
    results : dict
        - noise_levels: tested levels
        - mean_residual: mean residual at each level
        - std_residual: std of residual
        - mean_n_sources: mean number of recovered sources
    """
    if noise_levels is None:
        noise_levels = np.logspace(-4, -1, 10)
    
    # Generate clean data
    u_clean = forward_func(true_sources)
    u_norm = np.linalg.norm(u_clean)
    
    mean_residual = []
    std_residual = []
    mean_n_sources = []
    
    for noise in noise_levels:
        residuals = []
        n_sources = []
        
        for trial in range(n_trials):
            np.random.seed(42 + trial)
            u_noisy = u_clean + noise * u_norm * np.random.randn(len(u_clean))
            
            q = inverse_func(u_noisy)
            
            # Compute metrics
            res = np.linalg.norm(forward_func(q) - u_noisy) if callable(forward_func) else 0
            residuals.append(res)
            
            threshold = 0.1 * np.abs(q).max()
            n_sources.append(np.sum(np.abs(q) > threshold))
        
        mean_residual.append(np.mean(residuals))
        std_residual.append(np.std(residuals))
        mean_n_sources.append(np.mean(n_sources))
    
    return {
        'noise_levels': noise_levels,
        'mean_residual': np.array(mean_residual),
        'std_residual': np.array(std_residual),
        'mean_n_sources': np.array(mean_n_sources),
        'true_n_sources': len(true_sources),
    }


def plot_noise_study(results: Dict, ax: plt.Axes = None) -> plt.Axes:
    """
    Plot noise study results.
    
    Parameters
    ----------
    results : dict
        Output from noise_study()
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    noise = results['noise_levels']
    n_sources = results['mean_n_sources']
    true_n = results['true_n_sources']
    
    ax.semilogx(noise, n_sources, 'b.-', markersize=8, label='Recovered')
    ax.axhline(true_n, color='r', linestyle='--', label=f'True ({true_n})')
    
    ax.set_xlabel('Noise Level (fraction of signal)')
    ax.set_ylabel('Number of Significant Sources')
    ax.set_title('Noise Sensitivity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax
