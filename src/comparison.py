#!/usr/bin/env python
"""
Comprehensive Solver Comparison
===============================

Compare all solver combinations:
  - BEM Linear vs BEM Nonlinear
  - FEM Linear vs FEM Nonlinear
  - BEM vs FEM
  - L1 vs L2 vs TV regularization

Usage:
    python -m inverse_source.comparison
    python -m inverse_source.comparison --quick
    python -m inverse_source.comparison --method all --save
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from time import time


@dataclass
class ComparisonResult:
    """Results from a solver comparison."""
    solver_name: str
    method_type: str  # 'linear' or 'nonlinear'
    forward_type: str  # 'bem' or 'fem'
    
    # Accuracy metrics
    position_rmse: float
    intensity_rmse: float
    boundary_residual: float
    
    # Computational cost
    time_seconds: float
    iterations: Optional[int] = None
    
    # Recovered sources
    sources_recovered: List[Tuple[Tuple[float, float], float]] = None


def compute_metrics(sources_true, sources_recovered, u_true, u_recovered) -> Dict:
    """Compute comparison metrics between true and recovered sources."""
    from scipy.optimize import linear_sum_assignment
    
    # Convert to arrays
    true_pos = np.array([s[0] for s in sources_true])
    true_int = np.array([s[1] for s in sources_true])
    
    rec_pos = np.array([s[0] for s in sources_recovered])
    rec_int = np.array([s[1] for s in sources_recovered])
    
    # Match sources using Hungarian algorithm (optimal assignment)
    n_true = len(sources_true)
    n_rec = len(sources_recovered)
    
    if n_rec == 0:
        return {
            'position_rmse': np.inf,
            'intensity_rmse': np.inf,
            'boundary_residual': np.linalg.norm(u_true - u_recovered) / np.linalg.norm(u_true)
        }
    
    # Build cost matrix (distance between all pairs)
    cost = np.zeros((n_true, n_rec))
    for i in range(n_true):
        for j in range(n_rec):
            cost[i, j] = np.linalg.norm(true_pos[i] - rec_pos[j])
    
    # Optimal matching
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Compute errors for matched pairs
    pos_errors = [cost[i, j] for i, j in zip(row_ind, col_ind)]
    int_errors = [abs(true_int[i] - rec_int[j]) for i, j in zip(row_ind, col_ind)]
    
    return {
        'position_rmse': np.sqrt(np.mean(np.array(pos_errors)**2)),
        'intensity_rmse': np.sqrt(np.mean(np.array(int_errors)**2)),
        'boundary_residual': np.linalg.norm(u_true - u_recovered) / np.linalg.norm(u_true)
    }


def run_bem_linear(u_measured, sources_true, alpha=1e-4, method='l1') -> ComparisonResult:
    """Run BEM linear inverse solver."""
    from .bem_solver import BEMLinearInverseSolver, BEMForwardSolver
    
    t0 = time()
    
    linear = BEMLinearInverseSolver(n_boundary=len(u_measured))
    linear.build_greens_matrix()
    
    if method == 'l1':
        q = linear.solve_l1(u_measured, alpha=alpha)
    elif method == 'l2':
        q = linear.solve_l2(u_measured, alpha=alpha)
    elif method == 'tv':
        q = linear.solve_tv(u_measured, alpha=alpha)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    elapsed = time() - t0
    
    # Extract significant sources
    threshold = 0.1 * np.max(np.abs(q))
    significant_idx = np.where(np.abs(q) > threshold)[0]
    
    sources_rec = []
    for idx in significant_idx:
        pos = linear.interior_points[idx]
        sources_rec.append(((pos[0], pos[1]), q[idx]))
    
    # Compute recovered boundary data
    forward = BEMForwardSolver(n_boundary_points=len(u_measured))
    u_rec = linear.G @ q
    u_rec = u_rec - np.mean(u_rec)
    
    u_true = u_measured - np.mean(u_measured)
    metrics = compute_metrics(sources_true, sources_rec, u_true, u_rec)
    
    return ComparisonResult(
        solver_name=f"BEM Linear ({method.upper()})",
        method_type='linear',
        forward_type='bem',
        position_rmse=metrics['position_rmse'],
        intensity_rmse=metrics['intensity_rmse'],
        boundary_residual=metrics['boundary_residual'],
        time_seconds=elapsed,
        sources_recovered=sources_rec
    )


def run_bem_nonlinear(u_measured, sources_true, n_sources=4, 
                       optimizer='L-BFGS-B') -> ComparisonResult:
    """Run BEM nonlinear inverse solver."""
    from .bem_solver import BEMNonlinearInverseSolver, BEMForwardSolver
    
    t0 = time()
    
    inverse = BEMNonlinearInverseSolver(n_sources=n_sources, n_boundary=len(u_measured))
    inverse.set_measured_data(u_measured)
    result = inverse.solve(method=optimizer, maxiter=200)
    
    elapsed = time() - t0
    
    # Convert sources
    sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
    
    # Compute recovered boundary data
    forward = BEMForwardSolver(n_boundary_points=len(u_measured))
    u_rec = forward.solve(sources_rec)
    
    u_true = u_measured - np.mean(u_measured)
    metrics = compute_metrics(sources_true, sources_rec, u_true, u_rec)
    
    return ComparisonResult(
        solver_name=f"BEM Nonlinear ({optimizer})",
        method_type='nonlinear',
        forward_type='bem',
        position_rmse=metrics['position_rmse'],
        intensity_rmse=metrics['intensity_rmse'],
        boundary_residual=metrics['boundary_residual'],
        time_seconds=elapsed,
        iterations=result.iterations,
        sources_recovered=sources_rec
    )


def run_fem_linear(u_measured, sources_true, alpha=1e-3, method='l1') -> ComparisonResult:
    """Run FEM linear inverse solver."""
    from .fem_solver import FEMLinearInverseSolver, FEMForwardSolver
    
    t0 = time()
    
    linear = FEMLinearInverseSolver(n_radial=10, n_angular=20)
    linear.build_greens_matrix()
    
    # Interpolate u_measured to FEM boundary if needed
    if len(u_measured) != linear.n_boundary:
        from scipy.interpolate import interp1d
        theta_meas = np.linspace(0, 2*np.pi, len(u_measured), endpoint=False)
        interp = interp1d(theta_meas, u_measured, kind='linear', fill_value='extrapolate')
        u_fem = interp(linear.theta)
    else:
        u_fem = u_measured
    
    if method == 'l1':
        q = linear.solve_l1(u_fem, alpha=alpha)
    elif method == 'l2':
        q = linear.solve_l2(u_fem, alpha=alpha)
    elif method == 'tv':
        q = linear.solve_tv(u_fem, alpha=alpha)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    elapsed = time() - t0
    
    # Extract significant sources
    threshold = 0.1 * np.max(np.abs(q))
    significant_idx = np.where(np.abs(q) > threshold)[0]
    
    sources_rec = []
    for idx in significant_idx:
        pos = linear.interior_points[idx]
        sources_rec.append(((pos[0], pos[1]), q[idx]))
    
    # Compute recovered boundary data
    u_rec = linear.G @ q
    u_rec = u_rec - np.mean(u_rec)
    
    u_true = u_fem - np.mean(u_fem)
    metrics = compute_metrics(sources_true, sources_rec, u_true, u_rec)
    
    return ComparisonResult(
        solver_name=f"FEM Linear ({method.upper()})",
        method_type='linear',
        forward_type='fem',
        position_rmse=metrics['position_rmse'],
        intensity_rmse=metrics['intensity_rmse'],
        boundary_residual=metrics['boundary_residual'],
        time_seconds=elapsed,
        sources_recovered=sources_rec
    )


def run_fem_nonlinear(u_measured, sources_true, n_sources=4,
                       optimizer='L-BFGS-B') -> ComparisonResult:
    """Run FEM nonlinear inverse solver."""
    from .fem_solver import FEMNonlinearInverseSolver, FEMForwardSolver
    
    t0 = time()
    
    inverse = FEMNonlinearInverseSolver(n_sources=n_sources, n_radial=10, n_angular=20)
    inverse.set_measured_data(u_measured)
    result = inverse.solve(method=optimizer, maxiter=100)
    
    elapsed = time() - t0
    
    # Convert sources
    sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
    
    # Compute recovered boundary data
    forward = FEMForwardSolver(n_radial=10, n_angular=20)
    u_rec = forward.solve(sources_rec)
    
    # Interpolate to match
    if len(u_rec) != len(u_measured):
        from scipy.interpolate import interp1d
        interp = interp1d(forward.theta, u_rec, kind='linear', fill_value='extrapolate')
        theta_meas = np.linspace(0, 2*np.pi, len(u_measured), endpoint=False)
        u_rec = interp(theta_meas)
    
    u_true = u_measured - np.mean(u_measured)
    u_rec = u_rec - np.mean(u_rec)
    metrics = compute_metrics(sources_true, sources_rec, u_true, u_rec)
    
    return ComparisonResult(
        solver_name=f"FEM Nonlinear ({optimizer})",
        method_type='nonlinear',
        forward_type='fem',
        position_rmse=metrics['position_rmse'],
        intensity_rmse=metrics['intensity_rmse'],
        boundary_residual=metrics['boundary_residual'],
        time_seconds=elapsed,
        iterations=result.iterations,
        sources_recovered=sources_rec
    )


def compare_all_solvers(sources_true: List[Tuple[Tuple[float, float], float]],
                        noise_level: float = 0.001,
                        alpha_linear: float = 1e-4,
                        quick: bool = False,
                        verbose: bool = True) -> List[ComparisonResult]:
    """
    Compare all solver combinations.
    
    Parameters
    ----------
    sources_true : list
        True source configuration
    noise_level : float
        Noise standard deviation
    alpha_linear : float
        Regularization parameter for linear solvers
    quick : bool
        If True, skip slower solvers
    verbose : bool
        Print progress
        
    Returns
    -------
    results : list of ComparisonResult
    """
    from .bem_solver import BEMForwardSolver
    
    # Generate synthetic data using BEM
    if verbose:
        print("Generating synthetic data...")
    forward = BEMForwardSolver(n_boundary_points=100)
    u_clean = forward.solve(sources_true)
    np.random.seed(42)
    u_measured = u_clean + noise_level * np.random.randn(len(u_clean))
    
    n_sources = len(sources_true)
    results = []
    
    # BEM Linear solvers
    if verbose:
        print("\n" + "="*60)
        print("BEM LINEAR SOLVERS")
        print("="*60)
    
    for method in ['l1', 'l2', 'tv']:
        if verbose:
            print(f"\nRunning BEM Linear ({method.upper()})...")
        try:
            result = run_bem_linear(u_measured, sources_true, alpha=alpha_linear, method=method)
            results.append(result)
            if verbose:
                print(f"  Position RMSE: {result.position_rmse:.4f}")
                print(f"  Time: {result.time_seconds:.2f}s")
        except Exception as e:
            if verbose:
                print(f"  Failed: {e}")
    
    # BEM Nonlinear
    if verbose:
        print("\n" + "="*60)
        print("BEM NONLINEAR SOLVERS")
        print("="*60)
    
    optimizers = ['L-BFGS-B'] if quick else ['L-BFGS-B', 'differential_evolution']
    for opt in optimizers:
        if verbose:
            print(f"\nRunning BEM Nonlinear ({opt})...")
        try:
            result = run_bem_nonlinear(u_measured, sources_true, n_sources=n_sources, optimizer=opt)
            results.append(result)
            if verbose:
                print(f"  Position RMSE: {result.position_rmse:.4f}")
                print(f"  Time: {result.time_seconds:.2f}s")
        except Exception as e:
            if verbose:
                print(f"  Failed: {e}")
    
    # FEM Linear solvers
    if verbose:
        print("\n" + "="*60)
        print("FEM LINEAR SOLVERS")
        print("="*60)
    
    fem_methods = ['l1'] if quick else ['l1', 'l2', 'tv']
    for method in fem_methods:
        if verbose:
            print(f"\nRunning FEM Linear ({method.upper()})...")
        try:
            result = run_fem_linear(u_measured, sources_true, alpha=alpha_linear*10, method=method)
            results.append(result)
            if verbose:
                print(f"  Position RMSE: {result.position_rmse:.4f}")
                print(f"  Time: {result.time_seconds:.2f}s")
        except Exception as e:
            if verbose:
                print(f"  Failed: {e}")
    
    # FEM Nonlinear
    if not quick:
        if verbose:
            print("\n" + "="*60)
            print("FEM NONLINEAR SOLVERS")
            print("="*60)
            print("\nRunning FEM Nonlinear (L-BFGS-B)...")
        try:
            result = run_fem_nonlinear(u_measured, sources_true, n_sources=n_sources)
            results.append(result)
            if verbose:
                print(f"  Position RMSE: {result.position_rmse:.4f}")
                print(f"  Time: {result.time_seconds:.2f}s")
        except Exception as e:
            if verbose:
                print(f"  Failed: {e}")
    
    return results


def print_comparison_table(results: List[ComparisonResult]):
    """Print a formatted comparison table."""
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    # Header
    print(f"\n{'Solver':<35} {'Pos RMSE':>10} {'Int RMSE':>10} {'Residual':>10} {'Time':>8}")
    print("-"*80)
    
    # Sort by position RMSE
    sorted_results = sorted(results, key=lambda r: r.position_rmse)
    
    for r in sorted_results:
        print(f"{r.solver_name:<35} {r.position_rmse:>10.4f} {r.intensity_rmse:>10.4f} "
              f"{r.boundary_residual:>10.4f} {r.time_seconds:>7.2f}s")
    
    print("-"*80)
    
    # Best results
    best_pos = min(results, key=lambda r: r.position_rmse)
    best_time = min(results, key=lambda r: r.time_seconds)
    
    print(f"\nBest position accuracy: {best_pos.solver_name}")
    print(f"Fastest solver: {best_time.solver_name}")


def plot_comparison(results: List[ComparisonResult], sources_true, save_path=None):
    """Create comparison visualization."""
    n_results = len(results)
    n_cols = min(4, n_results)
    n_rows = (n_results + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_results == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Domain boundary
    theta = np.linspace(0, 2*np.pi, 100)
    boundary_x = np.cos(theta)
    boundary_y = np.sin(theta)
    
    for idx, result in enumerate(results):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        # Plot boundary
        ax.plot(boundary_x, boundary_y, 'k-', linewidth=1)
        
        # Plot true sources
        for (x, y), q in sources_true:
            color = 'red' if q > 0 else 'blue'
            ax.plot(x, y, 'o', color=color, markersize=12, 
                   markerfacecolor='none', markeredgewidth=2, label='True' if idx == 0 else '')
        
        # Plot recovered sources
        if result.sources_recovered:
            for (x, y), q in result.sources_recovered:
                color = 'red' if q > 0 else 'blue'
                ax.plot(x, y, '+', color=color, markersize=10, markeredgewidth=2)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_title(f"{result.solver_name}\nRMSE={result.position_rmse:.3f}, t={result.time_seconds:.1f}s",
                    fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_results, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to {save_path}")
    
    return fig


def main():
    """Main entry point for comparison script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare all inverse solvers')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer solvers)')
    parser.add_argument('--noise', type=float, default=0.001, help='Noise level')
    parser.add_argument('--alpha', type=float, default=1e-4, help='Regularization parameter')
    parser.add_argument('--save', type=str, default=None, help='Save figure to path')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    args = parser.parse_args()
    
    # Test sources
    sources_true = [
        ((-0.3, 0.4), 1.0),
        ((0.5, 0.3), 1.0),
        ((-0.4, -0.4), -1.0),
        ((0.3, -0.5), -1.0),
    ]
    
    print("="*60)
    print("INVERSE SOURCE LOCALIZATION - SOLVER COMPARISON")
    print("="*60)
    print(f"\nTrue sources: {len(sources_true)}")
    print(f"Noise level: {args.noise}")
    print(f"Alpha (linear): {args.alpha}")
    
    # Run comparison
    results = compare_all_solvers(
        sources_true,
        noise_level=args.noise,
        alpha_linear=args.alpha,
        quick=args.quick
    )
    
    # Print table
    print_comparison_table(results)
    
    # Plot
    if not args.no_plot:
        fig = plot_comparison(results, sources_true, save_path=args.save or 'results/comparison.png')
        plt.show()


if __name__ == '__main__':
    main()
