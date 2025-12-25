"""
Parameter Study Tools for Inverse Source Localization
======================================================

Tools for systematic parameter sweeps, L-curve analysis, and method comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime


@dataclass
class SweepResult:
    """Container for parameter sweep results."""
    alphas: np.ndarray
    residuals: np.ndarray
    reg_norms: np.ndarray
    solutions: List[np.ndarray]
    alpha_optimal: float
    method: str
    metadata: Dict[str, Any]


# =============================================================================
# PARAMETER SWEEPS
# =============================================================================

def parameter_sweep(solver, u_measured: np.ndarray,
                    alphas: Optional[np.ndarray] = None,
                    method: str = 'l1',
                    verbose: bool = True) -> SweepResult:
    """
    Sweep over regularization parameters.
    
    Parameters
    ----------
    solver : LinearInverseSolver
        Solver with solve_l1, solve_l2, or solve_tv methods
    u_measured : array
        Measured boundary data
    alphas : array, optional
        Regularization parameters. If None, uses logspace(-6, -1, 30)
    method : str
        'l1', 'l2', or 'tv'
    verbose : bool
        Print progress
        
    Returns
    -------
    result : SweepResult
    """
    if alphas is None:
        alphas = np.logspace(-6, -1, 30)
    
    residuals = []
    reg_norms = []
    solutions = []
    
    u_centered = u_measured - np.mean(u_measured)
    
    for i, alpha in enumerate(alphas):
        if verbose:
            print(f"  [{i+1}/{len(alphas)}] α = {alpha:.2e}", end='\r')
        
        # Solve based on method
        if method.lower() == 'l1':
            q = solver.solve_l1(u_centered, alpha=alpha)
            reg_norm = np.sum(np.abs(q))
        elif method.lower() == 'l2':
            q = solver.solve_l2(u_centered, alpha=alpha)
            reg_norm = np.linalg.norm(q)
        elif method.lower() == 'tv':
            q = solver.solve_tv(u_centered, alpha=alpha)
            reg_norm = np.sum(np.abs(np.diff(q)))  # Approximate TV norm
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Compute residual
        if hasattr(solver, 'G'):
            residual = np.linalg.norm(solver.G @ q - u_centered)
        else:
            residual = np.nan
        
        residuals.append(residual)
        reg_norms.append(reg_norm)
        solutions.append(q.copy())
    
    if verbose:
        print()
    
    residuals = np.array(residuals)
    reg_norms = np.array(reg_norms)
    
    # Find optimal alpha using L-curve criterion
    alpha_opt = find_l_curve_corner(alphas, residuals, reg_norms)
    
    return SweepResult(
        alphas=alphas,
        residuals=residuals,
        reg_norms=reg_norms,
        solutions=solutions,
        alpha_optimal=alpha_opt,
        method=method,
        metadata={'timestamp': datetime.now().isoformat()},
    )


def find_l_curve_corner(alphas: np.ndarray, residuals: np.ndarray, 
                        reg_norms: np.ndarray) -> float:
    """
    Find the corner of the L-curve using curvature.
    
    Parameters
    ----------
    alphas : array
        Regularization parameters
    residuals : array
        Residual norms ||Gq - u||
    reg_norms : array
        Regularization norms ||q|| or ||q||₁
        
    Returns
    -------
    alpha_opt : float
        Optimal regularization parameter
    """
    # Work in log space
    log_res = np.log(residuals + 1e-14)
    log_reg = np.log(reg_norms + 1e-14)
    
    # Compute curvature using finite differences
    # κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
    dx = np.gradient(log_res)
    dy = np.gradient(log_reg)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-14)**1.5
    
    # Find maximum curvature (excluding boundary points)
    margin = max(2, len(alphas) // 10)
    corner_idx = np.argmax(curvature[margin:-margin]) + margin
    
    return alphas[corner_idx]


# =============================================================================
# L-CURVE ANALYSIS
# =============================================================================

def l_curve_analysis(solver, u_measured: np.ndarray,
                     methods: List[str] = ['l1', 'l2'],
                     alphas: Optional[np.ndarray] = None,
                     verbose: bool = True) -> Dict[str, SweepResult]:
    """
    Perform L-curve analysis for multiple regularization methods.
    
    Parameters
    ----------
    solver : LinearInverseSolver
    u_measured : array
    methods : list of str
        Methods to compare
    alphas : array, optional
    verbose : bool
        
    Returns
    -------
    results : dict
        {method: SweepResult}
    """
    results = {}
    
    for method in methods:
        if verbose:
            print(f"\nAnalyzing {method.upper()} regularization...")
        results[method] = parameter_sweep(solver, u_measured, alphas, method, verbose)
        if verbose:
            print(f"  Optimal α = {results[method].alpha_optimal:.2e}")
    
    return results


def plot_l_curve_comparison(results: Dict[str, SweepResult],
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot L-curves for multiple methods.
    
    Parameters
    ----------
    results : dict
        {method: SweepResult}
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    # L-curve plot
    ax = axes[0]
    for (method, result), color in zip(results.items(), colors):
        ax.loglog(result.residuals, result.reg_norms, '.-', 
                  color=color, label=method.upper(), markersize=4)
        
        # Mark optimal point
        opt_idx = np.argmin(np.abs(result.alphas - result.alpha_optimal))
        ax.plot(result.residuals[opt_idx], result.reg_norms[opt_idx], 
                'o', color=color, markersize=12)
    
    ax.set_xlabel('Residual ||Gq - u||')
    ax.set_ylabel('Regularization norm')
    ax.set_title('L-Curve Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # Optimal alpha bar chart
    ax = axes[1]
    methods = list(results.keys())
    opt_alphas = [results[m].alpha_optimal for m in methods]
    bars = ax.bar(range(len(methods)), opt_alphas, color=colors[:len(methods)])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_ylabel('Optimal α')
    ax.set_title('Optimal Regularization Parameters')
    ax.set_yscale('log')
    
    # Add value labels
    for bar, alpha in zip(bars, opt_alphas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{alpha:.2e}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# METHOD COMPARISON
# =============================================================================

def compare_methods(solver, u_measured: np.ndarray, sources_true: List,
                    methods: List[str] = ['l1', 'l2'],
                    alpha: Optional[float] = None,
                    verbose: bool = True) -> Dict[str, Dict]:
    """
    Compare different regularization methods at optimal or given alpha.
    
    Parameters
    ----------
    solver : LinearInverseSolver
    u_measured : array
    sources_true : list
        Ground truth sources for error computation
    methods : list of str
    alpha : float, optional
        If None, uses L-curve optimal for each method
    verbose : bool
        
    Returns
    -------
    comparison : dict
        {method: {'q': solution, 'residual': ..., 'metrics': ...}}
    """
    from .utils import compute_source_error
    
    comparison = {}
    u_centered = u_measured - np.mean(u_measured)
    
    for method in methods:
        if verbose:
            print(f"\n{method.upper()}:")
        
        # Find optimal alpha if not given
        if alpha is None:
            result = parameter_sweep(solver, u_measured, method=method, verbose=False)
            method_alpha = result.alpha_optimal
            if verbose:
                print(f"  Optimal α = {method_alpha:.2e}")
        else:
            method_alpha = alpha
        
        # Solve
        if method.lower() == 'l1':
            q = solver.solve_l1(u_centered, alpha=method_alpha)
        elif method.lower() == 'l2':
            q = solver.solve_l2(u_centered, alpha=method_alpha)
        elif method.lower() == 'tv':
            q = solver.solve_tv(u_centered, alpha=method_alpha)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        residual = np.linalg.norm(solver.G @ q - u_centered)
        
        # Extract recovered sources
        threshold = 0.1 * np.max(np.abs(q))
        significant_idx = np.where(np.abs(q) > threshold)[0]
        
        sources_recovered = []
        for idx in significant_idx:
            pos = solver.interior_points[idx]
            if hasattr(pos, '__len__') and len(pos) == 2:
                sources_recovered.append(((pos[0], pos[1]), q[idx]))
            else:
                # Complex number representation
                sources_recovered.append(((pos.real, pos.imag), q[idx]))
        
        # Compute metrics
        metrics = compute_source_error(sources_true, sources_recovered)
        
        if verbose:
            print(f"  Residual: {residual:.4e}")
            print(f"  Position RMSE: {metrics['position_rmse']:.4f}")
            print(f"  Intensity RMSE: {metrics['intensity_rmse']:.4f}")
            print(f"  Recovered {len(sources_recovered)} sources")
        
        comparison[method] = {
            'q': q,
            'alpha': method_alpha,
            'residual': residual,
            'sources_recovered': sources_recovered,
            'metrics': metrics,
        }
    
    return comparison


def plot_method_comparison(comparison: Dict[str, Dict], solver,
                           sources_true: List, u_measured: np.ndarray,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot visual comparison of different methods.
    
    Parameters
    ----------
    comparison : dict
        Output from compare_methods
    solver : LinearInverseSolver
    sources_true : list
    u_measured : array
    save_path : str, optional
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    n_methods = len(comparison)
    fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))
    
    theta_circle = np.linspace(0, 2*np.pi, 100)
    u_centered = u_measured - np.mean(u_measured)
    
    for i, (method, data) in enumerate(comparison.items()):
        # Top row: source recovery
        ax = axes[0, i]
        ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'k-', linewidth=2)
        
        # Plot intensity distribution
        vmax = max(np.max(np.abs(data['q'])), 0.01)
        if hasattr(solver, 'interior_points'):
            pts = solver.interior_points
            if isinstance(pts[0], complex):
                x = [p.real for p in pts]
                y = [p.imag for p in pts]
            else:
                x = [p[0] for p in pts]
                y = [p[1] for p in pts]
            scatter = ax.scatter(x, y, c=data['q'], cmap='RdBu_r', 
                               s=30, vmin=-vmax, vmax=vmax)
            plt.colorbar(scatter, ax=ax, label='q')
        
        # Mark true sources
        for (x, y), q in sources_true:
            ax.plot(x, y, 'ko', markersize=12, markerfacecolor='none', 
                    markeredgewidth=2)
        
        ax.set_aspect('equal')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_title(f'{method.upper()}\nPos RMSE: {data["metrics"]["position_rmse"]:.3f}')
        
        # Bottom row: boundary fit
        ax = axes[1, i]
        theta = np.linspace(0, 2*np.pi, len(u_measured), endpoint=False)
        ax.plot(theta, u_centered, 'b-', linewidth=2, label='Measured')
        u_fit = solver.G @ data['q']
        ax.plot(theta, u_fit, 'r--', linewidth=2, label='Fit')
        ax.set_xlabel('θ')
        ax.set_ylabel('u')
        ax.set_title(f'Residual: {data["residual"]:.4e}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# NOISE STUDY
# =============================================================================

def noise_study(solver, sources_true: List, 
                noise_levels: Optional[np.ndarray] = None,
                method: str = 'l1',
                n_trials: int = 10,
                verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Study the effect of noise on recovery quality.
    
    Parameters
    ----------
    solver : LinearInverseSolver
    sources_true : list
        True source configuration
    noise_levels : array, optional
        Noise standard deviations to test
    method : str
        Regularization method
    n_trials : int
        Number of trials per noise level
    verbose : bool
        
    Returns
    -------
    results : dict
        'noise_levels': array
        'position_rmse_mean': array
        'position_rmse_std': array
        'intensity_rmse_mean': array
        'intensity_rmse_std': array
    """
    from .utils import compute_source_error
    
    if noise_levels is None:
        noise_levels = np.logspace(-4, -1, 10)
    
    position_rmse = np.zeros((len(noise_levels), n_trials))
    intensity_rmse = np.zeros((len(noise_levels), n_trials))
    
    # Generate clean data
    u_clean = solver.solve_forward(sources_true)
    
    for i, sigma in enumerate(noise_levels):
        if verbose:
            print(f"Noise level {i+1}/{len(noise_levels)}: σ = {sigma:.2e}")
        
        for j in range(n_trials):
            # Add noise
            u_noisy = u_clean + sigma * np.random.randn(len(u_clean))
            
            # Find optimal alpha and solve
            sweep = parameter_sweep(solver, u_noisy, method=method, verbose=False)
            opt_idx = np.argmin(np.abs(sweep.alphas - sweep.alpha_optimal))
            q = sweep.solutions[opt_idx]
            
            # Extract sources
            threshold = 0.1 * np.max(np.abs(q))
            sources_rec = []
            for idx in np.where(np.abs(q) > threshold)[0]:
                pos = solver.interior_points[idx]
                if isinstance(pos, complex):
                    sources_rec.append(((pos.real, pos.imag), q[idx]))
                else:
                    sources_rec.append(((pos[0], pos[1]), q[idx]))
            
            # Compute errors
            metrics = compute_source_error(sources_true, sources_rec)
            position_rmse[i, j] = metrics['position_rmse']
            intensity_rmse[i, j] = metrics['intensity_rmse']
    
    return {
        'noise_levels': noise_levels,
        'position_rmse_mean': np.mean(position_rmse, axis=1),
        'position_rmse_std': np.std(position_rmse, axis=1),
        'intensity_rmse_mean': np.mean(intensity_rmse, axis=1),
        'intensity_rmse_std': np.std(intensity_rmse, axis=1),
    }


def plot_noise_study(results: Dict[str, np.ndarray],
                     save_path: Optional[str] = None) -> plt.Figure:
    """Plot noise study results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    noise = results['noise_levels']
    
    ax = axes[0]
    ax.errorbar(noise, results['position_rmse_mean'], 
                yerr=results['position_rmse_std'],
                fmt='o-', capsize=4)
    ax.set_xlabel('Noise Level σ')
    ax.set_ylabel('Position RMSE')
    ax.set_xscale('log')
    ax.set_title('Position Recovery vs Noise')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.errorbar(noise, results['intensity_rmse_mean'], 
                yerr=results['intensity_rmse_std'],
                fmt='o-', capsize=4)
    ax.set_xlabel('Noise Level σ')
    ax.set_ylabel('Intensity RMSE')
    ax.set_xscale('log')
    ax.set_title('Intensity Recovery vs Noise')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# SAVE/LOAD RESULTS
# =============================================================================

def save_results(results: Dict, path: str):
    """Save parameter study results to JSON."""
    # Convert numpy arrays to lists
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, SweepResult):
            return {
                'alphas': obj.alphas.tolist(),
                'residuals': obj.residuals.tolist(),
                'reg_norms': obj.reg_norms.tolist(),
                'alpha_optimal': obj.alpha_optimal,
                'method': obj.method,
            }
        return obj
    
    with open(path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"Results saved to {path}")


def load_results(path: str) -> Dict:
    """Load parameter study results from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Convert lists back to numpy arrays
    def convert(obj):
        if isinstance(obj, dict):
            if 'alphas' in obj and 'residuals' in obj:
                return SweepResult(
                    alphas=np.array(obj['alphas']),
                    residuals=np.array(obj['residuals']),
                    reg_norms=np.array(obj['reg_norms']),
                    solutions=[],
                    alpha_optimal=obj['alpha_optimal'],
                    method=obj['method'],
                    metadata={},
                )
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    return convert(data)


if __name__ == "__main__":
    print("Parameter study module - run demos through examples/")
