#!/usr/bin/env python
"""
Comprehensive Solver Comparison
===============================

Compare all solver combinations:
  - Analytical Linear vs Analytical Nonlinear
  - FEM Linear vs FEM Nonlinear
  - Analytical vs FEM
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
    
    # Recovered sources (for nonlinear solvers)
    sources_recovered: Optional[List[Tuple[Tuple[float, float], float]]] = None
    
    # Full intensity field (for linear solvers)
    grid_positions: Optional[np.ndarray] = None  # (M, 2) grid point positions
    grid_intensities: Optional[np.ndarray] = None  # (M,) intensity at each grid point


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
    """Run analytical linear inverse solver (formerly called BEM)."""
    try:
        from .analytical_solver import AnalyticalLinearInverseSolver as BEMLinearInverseSolver
        from .analytical_solver import AnalyticalForwardSolver as BEMForwardSolver
    except ImportError:
        from analytical_solver import AnalyticalLinearInverseSolver as BEMLinearInverseSolver
        from analytical_solver import AnalyticalForwardSolver as BEMForwardSolver
    
    t0 = time()
    
    linear = BEMLinearInverseSolver(n_boundary=len(u_measured), source_resolution=0.15, verbose=False)
    linear.build_greens_matrix(verbose=False)
    
    if method == 'l1':
        q = linear.solve_l1(u_measured, alpha=alpha)
    elif method == 'l2':
        q = linear.solve_l2(u_measured, alpha=alpha)
    elif method == 'tv':
        q = linear.solve_tv(u_measured, alpha=alpha)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    elapsed = time() - t0
    
    # Extract significant sources for metrics computation
    threshold = 0.1 * np.max(np.abs(q))
    significant_idx = np.where(np.abs(q) > threshold)[0]
    
    sources_rec = []
    for idx in significant_idx:
        pos = linear.interior_points[idx]
        sources_rec.append(((pos[0], pos[1]), q[idx]))
    
    # Compute recovered boundary data
    u_rec = linear.G @ q
    u_rec = u_rec - np.mean(u_rec)
    
    u_true = u_measured - np.mean(u_measured)
    metrics = compute_metrics(sources_true, sources_rec, u_true, u_rec)
    
    return ComparisonResult(
        solver_name=f"Analytical Linear ({method.upper()})",
        method_type='linear',
        forward_type='bem',
        position_rmse=metrics['position_rmse'],
        intensity_rmse=metrics['intensity_rmse'],
        boundary_residual=metrics['boundary_residual'],
        time_seconds=elapsed,
        sources_recovered=sources_rec,
        grid_positions=linear.interior_points.copy(),
        grid_intensities=q.copy()
    )


def run_bem_nonlinear(u_measured, sources_true, n_sources=4, 
                       optimizer='L-BFGS-B', n_restarts=1, seed=42) -> ComparisonResult:
    """Run analytical nonlinear inverse solver (formerly called BEM).
    
    Parameters
    ----------
    seed : int
        Random seed for differential_evolution. Critical for reproducibility.
    """
    try:
        from .analytical_solver import AnalyticalNonlinearInverseSolver as BEMNonlinearInverseSolver
        from .analytical_solver import Source, InverseResult
    except ImportError:
        from analytical_solver import AnalyticalNonlinearInverseSolver as BEMNonlinearInverseSolver
        from analytical_solver import Source, InverseResult
    from scipy.optimize import differential_evolution
    
    t0 = time()
    
    inverse = BEMNonlinearInverseSolver(n_sources=n_sources, n_boundary=len(u_measured))
    inverse.set_measured_data(u_measured)
    
    if optimizer == 'differential_evolution':
        # Use explicit seed for reproducibility
        n = n_sources
        bounds = []
        for i in range(n):
            bounds.extend([(-0.85, 0.85), (-0.85, 0.85)])
            if i < n - 1: 
                bounds.append((-5.0, 5.0))
        
        inverse.history = []
        result_de = differential_evolution(inverse._objective, bounds, 
                                           maxiter=200, seed=seed, polish=True, workers=1)
        sources = [Source(x, y, q) for (x, y), q in inverse._params_to_sources(result_de.x)]
        result = InverseResult(sources, np.sqrt(result_de.fun), True, '', 
                               result_de.nit, inverse.history.copy())
    else:
        result = inverse.solve(method=optimizer, maxiter=200, n_restarts=n_restarts)
    
    elapsed = time() - t0
    
    # Convert sources
    sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
    
    # Compute recovered boundary data
    u_rec = inverse.forward.solve(sources_rec)
    
    u_true = u_measured - np.mean(u_measured)
    metrics = compute_metrics(sources_true, sources_rec, u_true, u_rec)
    
    name = f"Analytical Nonlinear ({optimizer})"
    if n_restarts > 1:
        name += f" x{n_restarts}"
    
    return ComparisonResult(
        solver_name=name,
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
    from .fem_solver import FEMLinearInverseSolver
    
    t0 = time()
    
    linear = FEMLinearInverseSolver(forward_resolution=0.1, source_resolution=0.15, verbose=False)
    linear.build_greens_matrix(verbose=False)
    
    # Interpolate u_measured to FEM boundary - handle theta wraparound!
    # BEM uses theta in [0, 2π], FEM might use [-π, π]
    theta_meas = np.linspace(0, 2*np.pi, len(u_measured), endpoint=False)
    
    # Convert FEM theta to [0, 2π] range for consistent interpolation
    fem_theta_normalized = linear.theta.copy()
    fem_theta_normalized[fem_theta_normalized < 0] += 2*np.pi
    
    # Sort by normalized theta for proper interpolation
    sort_idx = np.argsort(fem_theta_normalized)
    fem_theta_sorted = fem_theta_normalized[sort_idx]
    
    # Extend data for periodic interpolation
    theta_extended = np.concatenate([theta_meas - 2*np.pi, theta_meas, theta_meas + 2*np.pi])
    u_extended = np.concatenate([u_measured, u_measured, u_measured])
    
    from scipy.interpolate import interp1d
    interp = interp1d(theta_extended, u_extended, kind='linear')
    u_fem_sorted = interp(fem_theta_sorted)
    
    # Unsort to match original FEM ordering
    u_fem = np.zeros_like(u_fem_sorted)
    u_fem[sort_idx] = u_fem_sorted
    
    if method == 'l1':
        q = linear.solve_l1(u_fem, alpha=alpha)
    elif method == 'l2':
        q = linear.solve_l2(u_fem, alpha=alpha)
    elif method == 'tv':
        q = linear.solve_tv(u_fem, alpha=alpha)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    elapsed = time() - t0
    
    # Extract significant sources for metrics
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
        sources_recovered=sources_rec,
        grid_positions=linear.interior_points.copy(),
        grid_intensities=q.copy()
    )


def run_fem_nonlinear(u_measured, sources_true, n_sources=4,
                       optimizer='L-BFGS-B', n_restarts=1, seed=42) -> ComparisonResult:
    """Run FEM nonlinear inverse solver.
    
    Parameters
    ----------
    seed : int
        Random seed for differential_evolution. Critical for reproducibility.
    """
    from .fem_solver import FEMNonlinearInverseSolver, Source, InverseResult
    from scipy.optimize import differential_evolution
    from scipy.interpolate import interp1d
    
    t0 = time()
    
    inverse = FEMNonlinearInverseSolver(n_sources=n_sources, resolution=0.1, verbose=False)
    
    # Fix theta range: convert measurement to FEM theta range
    theta_meas = np.linspace(0, 2*np.pi, len(u_measured), endpoint=False)
    fem_theta = inverse.forward.theta.copy()
    fem_theta[fem_theta < 0] += 2*np.pi
    sort_idx = np.argsort(fem_theta)
    fem_theta_sorted = fem_theta[sort_idx]
    
    # Periodic interpolation
    theta_ext = np.concatenate([theta_meas - 2*np.pi, theta_meas, theta_meas + 2*np.pi])
    u_ext = np.concatenate([u_measured, u_measured, u_measured])
    interp = interp1d(theta_ext, u_ext, kind='linear')
    u_fem_sorted = interp(fem_theta_sorted)
    
    # Unsort
    u_fem = np.zeros_like(u_fem_sorted)
    u_fem[sort_idx] = u_fem_sorted
    
    inverse.set_measured_data(u_fem)
    
    if optimizer == 'differential_evolution':
        # Use explicit seed for reproducibility
        n = n_sources
        bounds = []
        for i in range(n):
            bounds.extend([(-0.8, 0.8), (-0.8, 0.8)])
            if i < n - 1:
                bounds.append((-5.0, 5.0))
        
        inverse.history = []
        result_de = differential_evolution(inverse._objective, bounds, 
                                           maxiter=100, seed=seed, polish=True, workers=1)
        sources = [Source(x, y, q) for (x, y), q in inverse._params_to_sources(result_de.x)]
        result = InverseResult(sources, np.sqrt(result_de.fun), True, '', 
                               result_de.nit, inverse.history.copy())
    else:
        result = inverse.solve(method=optimizer, maxiter=100, n_restarts=n_restarts)
    
    elapsed = time() - t0
    
    # Convert sources
    sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
    
    # Compute recovered boundary data
    u_rec = inverse.forward.solve(sources_rec)
    
    u_true = u_fem - np.mean(u_fem)
    u_rec = u_rec - np.mean(u_rec)
    metrics = compute_metrics(sources_true, sources_rec, u_true, u_rec)
    
    name = f"FEM Nonlinear ({optimizer})"
    if n_restarts > 1:
        name += f" x{n_restarts}"
    
    return ComparisonResult(
        solver_name=name,
        method_type='nonlinear',
        forward_type='fem',
        position_rmse=metrics['position_rmse'],
        intensity_rmse=metrics['intensity_rmse'],
        boundary_residual=metrics['boundary_residual'],
        time_seconds=elapsed,
        iterations=result.iterations,
        sources_recovered=sources_rec
    )


def run_bem_numerical_linear(u_measured, sources_true, alpha=1e-4, method='l1') -> ComparisonResult:
    """Run BEM numerical linear inverse solver."""
    try:
        from .bem_solver import BEMLinearInverseSolver, BEMForwardSolver
    except ImportError:
        from bem_solver import BEMLinearInverseSolver, BEMForwardSolver
    
    t0 = time()
    
    linear = BEMLinearInverseSolver(n_boundary=len(u_measured), source_resolution=0.15, verbose=False)
    linear.build_greens_matrix(verbose=False)
    
    if method == 'l1':
        q = linear.solve_l1(u_measured, alpha=alpha)
    elif method == 'l2':
        q = linear.solve_l2(u_measured, alpha=alpha)
    elif method in ('tv', 'tv_admm'):
        q = linear.solve_tv(u_measured, alpha=alpha, method='admm')
    else:
        raise ValueError(f"Unknown method: {method}")
    
    elapsed = time() - t0
    
    # Extract significant sources
    threshold = 0.1 * np.max(np.abs(q))
    significant = np.where(np.abs(q) > threshold)[0]
    
    sources_rec = []
    for idx in significant:
        pos = linear.interior_points[idx]
        sources_rec.append(((pos[0], pos[1]), q[idx]))
    
    # Compute metrics
    u_rec = linear.G @ q
    u_true = u_measured - np.mean(u_measured)
    u_rec = u_rec - np.mean(u_rec)
    metrics = compute_metrics(sources_true, sources_rec, u_true, u_rec)
    
    return ComparisonResult(
        solver_name=f"BEM Numerical Linear ({method.upper()})",
        method_type='linear',
        forward_type='bem_numerical',
        position_rmse=metrics['position_rmse'],
        intensity_rmse=metrics['intensity_rmse'],
        boundary_residual=metrics['boundary_residual'],
        time_seconds=elapsed,
        grid_positions=linear.interior_points,
        grid_intensities=q
    )


def run_bem_numerical_nonlinear(u_measured, sources_true, n_sources=4, 
                                 optimizer='L-BFGS-B', n_restarts=1, seed=42) -> ComparisonResult:
    """Run BEM numerical nonlinear inverse solver."""
    try:
        from .bem_solver import BEMNonlinearInverseSolver, BEMForwardSolver
    except ImportError:
        from bem_solver import BEMNonlinearInverseSolver, BEMForwardSolver
    
    t0 = time()
    
    inverse = BEMNonlinearInverseSolver(n_sources=n_sources, n_boundary=len(u_measured))
    inverse.set_measured_data(u_measured)
    
    if optimizer == 'differential_evolution':
        result = inverse.solve(method='differential_evolution', maxiter=100)
    else:
        result = inverse.solve(method=optimizer, maxiter=100, n_restarts=n_restarts)
    
    elapsed = time() - t0
    
    # Convert sources
    sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
    
    # Compute recovered boundary data using same-sized forward solver
    forward = BEMForwardSolver(n_elements=len(u_measured))
    u_rec = forward.solve(sources_rec)
    
    u_true = u_measured - np.mean(u_measured)
    u_rec = u_rec - np.mean(u_rec)
    metrics = compute_metrics(sources_true, sources_rec, u_true, u_rec)
    
    name = f"BEM Numerical Nonlinear ({optimizer})"
    if n_restarts > 1:
        name += f" x{n_restarts}"
    
    return ComparisonResult(
        solver_name=name,
        method_type='nonlinear',
        forward_type='bem_numerical',
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
                        seed: int = 42,
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
        If True, skip slower solvers (differential_evolution)
    seed : int
        Random seed for reproducibility
    verbose : bool
        Print progress
        
    Returns
    -------
    results : list of ComparisonResult
    """
    try:
        from .analytical_solver import AnalyticalForwardSolver as BEMForwardSolver
    except ImportError:
        from analytical_solver import AnalyticalForwardSolver as BEMForwardSolver
    
    # Generate synthetic data using analytical solver
    if verbose:
        print(f"Generating synthetic data (seed={seed})...")
    forward = BEMForwardSolver(n_boundary_points=100)
    u_clean = forward.solve(sources_true)
    np.random.seed(seed)
    u_measured = u_clean + noise_level * np.random.randn(len(u_clean))
    
    n_sources = len(sources_true)
    results = []
    
    # =========================================================================
    # ANALYTICAL LINEAR SOLVERS (L1, L2, TV)
    # =========================================================================
    if verbose:
        print("\n" + "="*60)
        print("ANALYTICAL LINEAR SOLVERS")
        print("="*60)
    
    for method in ['l1', 'l2', 'tv']:
        if verbose:
            print(f"\nRunning Analytical Linear ({method.upper()})...")
        try:
            result = run_bem_linear(u_measured, sources_true, alpha=alpha_linear, method=method)
            results.append(result)
            if verbose:
                print(f"  Position RMSE: {result.position_rmse:.4f}")
                print(f"  Time: {result.time_seconds:.2f}s")
        except Exception as e:
            if verbose:
                print(f"  Failed: {e}")
    
    # =========================================================================
    # ANALYTICAL NONLINEAR SOLVERS
    # =========================================================================
    if verbose:
        print("\n" + "="*60)
        print("ANALYTICAL NONLINEAR SOLVERS")
        print("="*60)
    
    if quick:
        # Quick mode: L-BFGS-B with 5 restarts
        if verbose:
            print(f"\nRunning Analytical Nonlinear (L-BFGS-B x5)...")
        try:
            result = run_bem_nonlinear(u_measured, sources_true, n_sources=n_sources, 
                                       optimizer='L-BFGS-B', n_restarts=5, seed=seed)
            results.append(result)
            if verbose:
                print(f"  Position RMSE: {result.position_rmse:.4f}")
                print(f"  Time: {result.time_seconds:.2f}s")
        except Exception as e:
            if verbose:
                print(f"  Failed: {e}")
    else:
        # Full mode: L-BFGS-B x5 + differential_evolution
        if verbose:
            print(f"\nRunning Analytical Nonlinear (L-BFGS-B x5)...")
        try:
            result = run_bem_nonlinear(u_measured, sources_true, n_sources=n_sources, 
                                       optimizer='L-BFGS-B', n_restarts=5, seed=seed)
            results.append(result)
            if verbose:
                print(f"  Position RMSE: {result.position_rmse:.4f}")
                print(f"  Time: {result.time_seconds:.2f}s")
        except Exception as e:
            if verbose:
                print(f"  Failed: {e}")
        
        if verbose:
            print(f"\nRunning Analytical Nonlinear (differential_evolution, seed={seed})...")
        try:
            result = run_bem_nonlinear(u_measured, sources_true, n_sources=n_sources, 
                                       optimizer='differential_evolution', seed=seed)
            results.append(result)
            if verbose:
                print(f"  Position RMSE: {result.position_rmse:.4f}")
                print(f"  Time: {result.time_seconds:.2f}s")
        except Exception as e:
            if verbose:
                print(f"  Failed: {e}")
    
    # =========================================================================
    # BEM NUMERICAL LINEAR SOLVERS (L1, L2, TV)
    # =========================================================================
    if verbose:
        print("\n" + "="*60)
        print("BEM NUMERICAL LINEAR SOLVERS")
        print("="*60)
    
    for method in ['l1', 'l2', 'tv']:
        if verbose:
            print(f"\nRunning BEM Numerical Linear ({method.upper()})...")
        try:
            result = run_bem_numerical_linear(u_measured, sources_true, alpha=alpha_linear, method=method)
            results.append(result)
            if verbose:
                print(f"  Position RMSE: {result.position_rmse:.4f}")
                print(f"  Time: {result.time_seconds:.2f}s")
        except Exception as e:
            if verbose:
                print(f"  Failed: {e}")
    
    # =========================================================================
    # BEM NUMERICAL NONLINEAR SOLVERS
    # =========================================================================
    if verbose:
        print("\n" + "="*60)
        print("BEM NUMERICAL NONLINEAR SOLVERS")
        print("="*60)
    
    if quick:
        # Quick mode: L-BFGS-B with 5 restarts
        if verbose:
            print(f"\nRunning BEM Numerical Nonlinear (L-BFGS-B x5)...")
        try:
            result = run_bem_numerical_nonlinear(u_measured, sources_true, n_sources=n_sources, 
                                                  optimizer='L-BFGS-B', n_restarts=5, seed=seed)
            results.append(result)
            if verbose:
                print(f"  Position RMSE: {result.position_rmse:.4f}")
                print(f"  Time: {result.time_seconds:.2f}s")
        except Exception as e:
            if verbose:
                print(f"  Failed: {e}")
    else:
        # Full mode: L-BFGS-B x5 + differential_evolution
        if verbose:
            print(f"\nRunning BEM Numerical Nonlinear (L-BFGS-B x5)...")
        try:
            result = run_bem_numerical_nonlinear(u_measured, sources_true, n_sources=n_sources, 
                                                  optimizer='L-BFGS-B', n_restarts=5, seed=seed)
            results.append(result)
            if verbose:
                print(f"  Position RMSE: {result.position_rmse:.4f}")
                print(f"  Time: {result.time_seconds:.2f}s")
        except Exception as e:
            if verbose:
                print(f"  Failed: {e}")
        
        if verbose:
            print(f"\nRunning BEM Numerical Nonlinear (differential_evolution, seed={seed})...")
        try:
            result = run_bem_numerical_nonlinear(u_measured, sources_true, n_sources=n_sources, 
                                                  optimizer='differential_evolution', seed=seed)
            results.append(result)
            if verbose:
                print(f"  Position RMSE: {result.position_rmse:.4f}")
                print(f"  Time: {result.time_seconds:.2f}s")
        except Exception as e:
            if verbose:
                print(f"  Failed: {e}")
    
    # =========================================================================
    # FEM LINEAR SOLVERS (L1, L2, TV)
    # =========================================================================
    if verbose:
        print("\n" + "="*60)
        print("FEM LINEAR SOLVERS")
        print("="*60)
    
    for method in ['l1', 'l2', 'tv']:
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
    
    # =========================================================================
    # FEM NONLINEAR SOLVERS
    # =========================================================================
    if verbose:
        print("\n" + "="*60)
        print("FEM NONLINEAR SOLVERS")
        print("="*60)
    
    if quick:
        # Quick mode: L-BFGS-B with 5 restarts
        if verbose:
            print(f"\nRunning FEM Nonlinear (L-BFGS-B x5)...")
        try:
            result = run_fem_nonlinear(u_measured, sources_true, n_sources=n_sources, 
                                       optimizer='L-BFGS-B', n_restarts=5, seed=seed)
            results.append(result)
            if verbose:
                print(f"  Position RMSE: {result.position_rmse:.4f}")
                print(f"  Time: {result.time_seconds:.2f}s")
        except Exception as e:
            if verbose:
                print(f"  Failed: {e}")
    else:
        # Full mode: L-BFGS-B x5 + differential_evolution
        if verbose:
            print(f"\nRunning FEM Nonlinear (L-BFGS-B x5)...")
        try:
            result = run_fem_nonlinear(u_measured, sources_true, n_sources=n_sources, 
                                       optimizer='L-BFGS-B', n_restarts=5, seed=seed)
            results.append(result)
            if verbose:
                print(f"  Position RMSE: {result.position_rmse:.4f}")
                print(f"  Time: {result.time_seconds:.2f}s")
        except Exception as e:
            if verbose:
                print(f"  Failed: {e}")
        
        if verbose:
            print(f"\nRunning FEM Nonlinear (differential_evolution, seed={seed})...")
        try:
            result = run_fem_nonlinear(u_measured, sources_true, n_sources=n_sources, 
                                       optimizer='differential_evolution', seed=seed)
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
    print("\n" + "="*90)
    print("COMPARISON SUMMARY")
    print("="*90)
    
    # Header
    print(f"\n{'Solver':<35} {'Pos RMSE':>10} {'Int RMSE':>10} {'Residual':>10} {'Time':>8}")
    print("-"*90)
    
    # Sort by position RMSE
    sorted_results = sorted(results, key=lambda r: r.position_rmse)
    
    for r in sorted_results:
        print(f"{r.solver_name:<35} {r.position_rmse:>10.4f} {r.intensity_rmse:>10.4f} "
              f"{r.boundary_residual:>10.6f} {r.time_seconds:>7.2f}s")
    
    print("-"*90)
    
    # Best results
    best_pos = min(results, key=lambda r: r.position_rmse)
    best_int = min(results, key=lambda r: r.intensity_rmse)
    best_time = min(results, key=lambda r: r.time_seconds)
    
    print(f"\nBest position accuracy: {best_pos.solver_name} (RMSE={best_pos.position_rmse:.4f})")
    print(f"Best intensity accuracy: {best_int.solver_name} (RMSE={best_int.intensity_rmse:.4f})")
    print(f"Fastest solver: {best_time.solver_name} ({best_time.time_seconds:.2f}s)")


def plot_comparison(results: List[ComparisonResult], sources_true, save_path=None):
    """Create comparison visualization with intensity information."""
    n_results = len(results)
    n_cols = min(4, n_results)
    n_rows = (n_results + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, 4.5*n_rows))
    if n_results == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Domain boundary
    theta = np.linspace(0, 2*np.pi, 100)
    boundary_x = np.cos(theta)
    boundary_y = np.sin(theta)
    
    # Get max intensity for consistent scaling
    max_true_intensity = max(abs(q) for _, q in sources_true)
    
    for idx, result in enumerate(results):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        # Plot boundary
        ax.plot(boundary_x, boundary_y, 'k-', linewidth=2)
        
        # For linear solvers, show intensity field as colored scatter
        if result.method_type == 'linear' and result.grid_positions is not None:
            positions = result.grid_positions
            intensities = result.grid_intensities
            
            vmax = max(abs(intensities.min()), abs(intensities.max()))
            if vmax > 1e-10:
                scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                                   c=intensities, cmap='RdBu_r', 
                                   vmin=-vmax, vmax=vmax,
                                   s=40, alpha=0.8, edgecolors='none')
                plt.colorbar(scatter, ax=ax, shrink=0.6, label='q')
        
        # For nonlinear solvers, show discrete markers with size ~ intensity
        elif result.sources_recovered:
            max_rec_intensity = max(abs(q) for _, q in result.sources_recovered) if result.sources_recovered else 1
            scale_factor = 300 / max(max_true_intensity, max_rec_intensity)
            
            for (x, y), q in result.sources_recovered:
                color = 'red' if q > 0 else 'blue'
                size = abs(q) * scale_factor
                ax.scatter(x, y, c=color, s=size, marker='+', linewidths=3, zorder=5)
                # Add intensity label
                ax.annotate(f'{q:+.2f}', (x, y), textcoords='offset points', 
                           xytext=(5, 5), fontsize=7, color=color)
        
        # Plot true sources (always as circles with size ~ intensity)
        scale_factor_true = 300 / max_true_intensity
        for (x, y), q in sources_true:
            color = 'darkred' if q > 0 else 'darkblue'
            size = abs(q) * scale_factor_true
            ax.scatter(x, y, s=size, facecolors='none', edgecolors=color, 
                      linewidths=2.5, marker='o', zorder=4)
            # Add true intensity label for nonlinear plots
            if result.method_type == 'nonlinear':
                ax.annotate(f'({q:+.1f})', (x, y), textcoords='offset points',
                           xytext=(-15, -12), fontsize=6, color='gray')
        
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect('equal')
        ax.set_title(f"{result.solver_name}\nPos RMSE={result.position_rmse:.3f}, "
                    f"Int RMSE={result.intensity_rmse:.3f}, t={result.time_seconds:.1f}s",
                    fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_results, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    # Add legend
    fig.text(0.5, 0.02, 'True sources: ○ (size~|q|) | Nonlinear recovered: + (size~|q|, labeled) | Linear: color = intensity', 
             ha='center', fontsize=9)
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to {save_path}")
    
    return fig


# =============================================================================
# OPTIMAL ALPHA SELECTION AND COMPARISON
# =============================================================================

def find_optimal_alpha(solver, u_measured, method: str, 
                       alphas: np.ndarray = None, verbose: bool = False) -> Tuple[float, dict]:
    """
    Find optimal regularization parameter using L-curve criterion.
    
    Parameters
    ----------
    solver : BEMLinearInverseSolver or FEMLinearInverseSolver
        Linear inverse solver with Green's matrix built
    u_measured : array
        Boundary measurements
    method : str
        'l1', 'l2', 'tv_admm', 'tv_cp'
    alphas : array, optional
        Alpha values to test (default: logspace from 1e-6 to 1e-1)
    verbose : bool
        Print progress
        
    Returns
    -------
    alpha_opt : float
        Optimal regularization parameter
    sweep_data : dict
        Full sweep results
    """
    if alphas is None:
        alphas = np.logspace(-6, -1, 30)
    
    residuals = []
    reg_norms = []
    
    u = u_measured - np.mean(u_measured)
    
    for alpha in alphas:
        if method == 'l1':
            q = solver.solve_l1(u_measured, alpha=alpha)
        elif method == 'l2':
            q = solver.solve_l2(u_measured, alpha=alpha)
        elif method in ('tv', 'tv_admm'):
            q = solver.solve_tv(u_measured, alpha=alpha, method='admm')
        elif method == 'tv_cp':
            q = solver.solve_tv(u_measured, alpha=alpha, method='chambolle_pock')
        else:
            raise ValueError(f"Unknown method: {method}")
        
        residual = np.linalg.norm(solver.G @ q - u)
        
        if method == 'l1':
            reg_norm = np.sum(np.abs(q))
        elif method == 'l2':
            reg_norm = np.linalg.norm(q)
        else:  # TV
            from scipy.spatial import Delaunay
            tri = Delaunay(solver.interior_points)
            edges = set()
            for s in tri.simplices:
                for i in range(3):
                    edges.add(tuple(sorted([s[i], s[(i+1)%3]])))
            D = np.zeros((len(edges), solver.n_interior))
            for k, (i, j) in enumerate(edges):
                D[k, i], D[k, j] = 1, -1
            reg_norm = np.sum(np.abs(D @ q))
        
        residuals.append(residual)
        reg_norms.append(reg_norm)
    
    residuals = np.array(residuals)
    reg_norms = np.array(reg_norms)
    
    # L-curve criterion: find corner (maximum curvature)
    log_res = np.log10(residuals + 1e-14)
    log_reg = np.log10(reg_norms + 1e-14)
    
    # Compute curvature
    dx = np.gradient(log_res)
    dy = np.gradient(log_reg)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-14)**1.5
    
    # Find maximum curvature (avoiding endpoints)
    idx_opt = np.argmax(curvature[2:-2]) + 2
    alpha_opt = alphas[idx_opt]
    
    if verbose:
        print(f"  {method}: optimal α = {alpha_opt:.2e} (corner at idx {idx_opt})")
    
    return alpha_opt, {
        'alphas': alphas,
        'residuals': residuals,
        'reg_norms': reg_norms,
        'curvature': curvature,
        'idx_opt': idx_opt
    }


def compare_with_optimal_alpha(sources_true: List[Tuple[Tuple[float, float], float]],
                                noise_level: float = 0.001,
                                methods: List[str] = None,
                                include_nonlinear: bool = True,
                                seed: int = 42,
                                verbose: bool = True) -> List[ComparisonResult]:
    """
    Compare solvers using optimal regularization parameter for each method.
    
    This runs a parameter sweep first to find optimal α, then compares.
    
    Parameters
    ----------
    sources_true : list
        True source configuration
    noise_level : float
        Noise standard deviation
    methods : list, optional
        Linear methods to compare (default: ['l1', 'l2', 'tv_admm', 'tv_cp'])
    include_nonlinear : bool
        Include nonlinear solvers (these don't need α selection)
    seed : int
        Random seed for reproducibility
    verbose : bool
        Print progress
        
    Returns
    -------
    results : list of ComparisonResult
    """
    try:
        from .analytical_solver import (
            AnalyticalForwardSolver as BEMForwardSolver,
            AnalyticalLinearInverseSolver as BEMLinearInverseSolver,
            AnalyticalNonlinearInverseSolver as BEMNonlinearInverseSolver
        )
        from .fem_solver import FEMLinearInverseSolver, FEMNonlinearInverseSolver
    except ImportError:
        from analytical_solver import (
            AnalyticalForwardSolver as BEMForwardSolver,
            AnalyticalLinearInverseSolver as BEMLinearInverseSolver,
            AnalyticalNonlinearInverseSolver as BEMNonlinearInverseSolver
        )
        from fem_solver import FEMLinearInverseSolver, FEMNonlinearInverseSolver
    from scipy.interpolate import interp1d
    
    if methods is None:
        methods = ['l1', 'l2', 'tv_admm', 'tv_cp']
    
    # Generate synthetic data
    if verbose:
        print("="*70)
        print("COMPARISON WITH OPTIMAL REGULARIZATION PARAMETERS")
        print("="*70)
        print(f"\nGenerating synthetic data (seed={seed})...")
    
    forward = BEMForwardSolver(n_boundary_points=100)
    u_clean = forward.solve(sources_true)
    np.random.seed(seed)
    u_measured = u_clean + noise_level * np.random.randn(len(u_clean))
    
    n_sources = len(sources_true)
    results = []
    
    # =========================================================================
    # ANALYTICAL LINEAR - Find optimal α for each method
    # =========================================================================
    if verbose:
        print("\n" + "-"*70)
        print("STEP 1: Finding optimal α for BEM Linear methods")
        print("-"*70)
    
    bem_linear = BEMLinearInverseSolver(n_boundary=100, source_resolution=0.15, verbose=False)
    bem_linear.build_greens_matrix(verbose=False)
    
    bem_alphas = {}
    for method in methods:
        if verbose:
            print(f"\nSweeping {method.upper()}...")
        alpha_opt, _ = find_optimal_alpha(bem_linear, u_measured, method, verbose=verbose)
        bem_alphas[method] = alpha_opt
    
    # =========================================================================
    # FEM LINEAR - Find optimal α for each method
    # =========================================================================
    if verbose:
        print("\n" + "-"*70)
        print("STEP 2: Finding optimal α for FEM Linear methods")
        print("-"*70)
    
    fem_linear = FEMLinearInverseSolver(forward_resolution=0.1, source_resolution=0.15, verbose=False)
    fem_linear.build_greens_matrix(verbose=False)
    
    # Interpolate to FEM boundary
    theta_meas = np.linspace(0, 2*np.pi, len(u_measured), endpoint=False)
    fem_theta_normalized = fem_linear.theta.copy()
    fem_theta_normalized[fem_theta_normalized < 0] += 2*np.pi
    sort_idx = np.argsort(fem_theta_normalized)
    fem_theta_sorted = fem_theta_normalized[sort_idx]
    
    theta_ext = np.concatenate([theta_meas - 2*np.pi, theta_meas, theta_meas + 2*np.pi])
    u_ext = np.concatenate([u_measured, u_measured, u_measured])
    interp = interp1d(theta_ext, u_ext, kind='linear')
    u_fem_sorted = interp(fem_theta_sorted)
    u_fem = np.zeros_like(u_fem_sorted)
    u_fem[sort_idx] = u_fem_sorted
    
    fem_alphas = {}
    for method in methods:
        if verbose:
            print(f"\nSweeping {method.upper()}...")
        alpha_opt, _ = find_optimal_alpha(fem_linear, u_fem, method, verbose=verbose)
        fem_alphas[method] = alpha_opt
    
    # =========================================================================
    # STEP 3: Run comparisons with optimal α
    # =========================================================================
    if verbose:
        print("\n" + "-"*70)
        print("STEP 3: Running comparisons with optimal α")
        print("-"*70)
    
    # BEM Linear
    for method in methods:
        if verbose:
            print(f"\nBEM Linear ({method.upper()}) with α={bem_alphas[method]:.2e}...")
        
        t0 = time()
        if method == 'l1':
            q = bem_linear.solve_l1(u_measured, alpha=bem_alphas[method])
        elif method == 'l2':
            q = bem_linear.solve_l2(u_measured, alpha=bem_alphas[method])
        elif method == 'tv_admm':
            q = bem_linear.solve_tv(u_measured, alpha=bem_alphas[method], method='admm')
        elif method == 'tv_cp':
            q = bem_linear.solve_tv(u_measured, alpha=bem_alphas[method], method='chambolle_pock')
        elapsed = time() - t0
        
        # Extract sources and compute metrics
        threshold = 0.1 * np.max(np.abs(q))
        significant_idx = np.where(np.abs(q) > threshold)[0]
        sources_rec = [((bem_linear.interior_points[i][0], bem_linear.interior_points[i][1]), q[i]) 
                       for i in significant_idx]
        
        u_rec = bem_linear.G @ q
        metrics = compute_metrics(sources_true, sources_rec, 
                                  u_measured - np.mean(u_measured), 
                                  u_rec - np.mean(u_rec))
        
        method_name = method.upper().replace('_', '-')
        results.append(ComparisonResult(
            solver_name=f"Analytical Linear ({method_name})",
            method_type='linear',
            forward_type='bem',
            position_rmse=metrics['position_rmse'],
            intensity_rmse=metrics['intensity_rmse'],
            boundary_residual=metrics['boundary_residual'],
            time_seconds=elapsed,
            sources_recovered=sources_rec,
            grid_positions=bem_linear.interior_points.copy(),
            grid_intensities=q.copy()
        ))
        
        if verbose:
            print(f"  Pos RMSE: {metrics['position_rmse']:.4f}, Int RMSE: {metrics['intensity_rmse']:.4f}")
    
    # FEM Linear
    for method in methods:
        if verbose:
            print(f"\nFEM Linear ({method.upper()}) with α={fem_alphas[method]:.2e}...")
        
        t0 = time()
        if method == 'l1':
            q = fem_linear.solve_l1(u_fem, alpha=fem_alphas[method])
        elif method == 'l2':
            q = fem_linear.solve_l2(u_fem, alpha=fem_alphas[method])
        elif method == 'tv_admm':
            q = fem_linear.solve_tv(u_fem, alpha=fem_alphas[method], method='admm')
        elif method == 'tv_cp':
            q = fem_linear.solve_tv(u_fem, alpha=fem_alphas[method], method='chambolle_pock')
        elapsed = time() - t0
        
        threshold = 0.1 * np.max(np.abs(q))
        significant_idx = np.where(np.abs(q) > threshold)[0]
        sources_rec = [((fem_linear.interior_points[i][0], fem_linear.interior_points[i][1]), q[i]) 
                       for i in significant_idx]
        
        u_rec = fem_linear.G @ q
        metrics = compute_metrics(sources_true, sources_rec,
                                  u_fem - np.mean(u_fem),
                                  u_rec - np.mean(u_rec))
        
        method_name = method.upper().replace('_', '-')
        results.append(ComparisonResult(
            solver_name=f"FEM Linear ({method_name})",
            method_type='linear',
            forward_type='fem',
            position_rmse=metrics['position_rmse'],
            intensity_rmse=metrics['intensity_rmse'],
            boundary_residual=metrics['boundary_residual'],
            time_seconds=elapsed,
            sources_recovered=sources_rec,
            grid_positions=fem_linear.interior_points.copy(),
            grid_intensities=q.copy()
        ))
        
        if verbose:
            print(f"  Pos RMSE: {metrics['position_rmse']:.4f}, Int RMSE: {metrics['intensity_rmse']:.4f}")
    
    # =========================================================================
    # Nonlinear solvers (no α needed)
    # =========================================================================
    if include_nonlinear:
        if verbose:
            print("\n" + "-"*70)
            print(f"STEP 4: Nonlinear solvers (differential_evolution, seed={seed})")
            print("-"*70)
            print("NOTE: Nonlinear results are seed-dependent. Use --seed to control.")
        
        # BEM Nonlinear
        if verbose:
            print(f"\nBEM Nonlinear (differential_evolution, seed={seed})...")
        result = run_bem_nonlinear(u_measured, sources_true, n_sources=n_sources,
                                   optimizer='differential_evolution', seed=seed)
        results.append(result)
        if verbose:
            print(f"  Pos RMSE: {result.position_rmse:.4f}, Int RMSE: {result.intensity_rmse:.4f}")
        
        # FEM Nonlinear
        if verbose:
            print(f"\nFEM Nonlinear (differential_evolution, seed={seed})...")
        result = run_fem_nonlinear(u_measured, sources_true, n_sources=n_sources,
                                   optimizer='differential_evolution', seed=seed)
        results.append(result)
        if verbose:
            print(f"  Pos RMSE: {result.position_rmse:.4f}, Int RMSE: {result.intensity_rmse:.4f}")
    
    return results


def main():
    """Main entry point for comparison script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare all inverse solvers')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer solvers)')
    parser.add_argument('--noise', type=float, default=0.001, help='Noise level')
    parser.add_argument('--alpha', type=float, default=1e-4, help='Regularization parameter')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
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
    print(f"Seed: {args.seed}")
    
    # Run comparison
    results = compare_all_solvers(
        sources_true,
        noise_level=args.noise,
        alpha_linear=args.alpha,
        quick=args.quick,
        seed=args.seed
    )
    
    # Print table
    print_comparison_table(results)
    
    # Plot
    if not args.no_plot:
        fig = plot_comparison(results, sources_true, save_path=args.save or 'results/comparison.png')
        plt.show()


if __name__ == '__main__':
    main()
