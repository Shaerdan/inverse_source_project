#!/usr/bin/env python3
"""
Diagnose the three failing nonlinear solvers:
1. Star - Conformal L-BFGS-B (RMSE 0.4484)
2. Square - FEM L-BFGS-B (RMSE 0.5602)
3. Brain - FEM diff_evol (RMSE 0.5139)

Tests:
1. Seed sensitivity - do failures depend on random seed?
2. Gradient at truth - is the objective smooth?
3. More restarts - does increasing restarts help?
4. Initial guess analysis - where do solvers start?

Usage:
    cd src/
    python diagnose_three_failures.py
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from comparison import create_domain_sources, get_conformal_map
from conformal_solver import ConformalForwardSolver, ConformalNonlinearInverseSolver
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def compute_position_rmse(sources_true, sources_rec):
    """Compute position RMSE with optimal matching."""
    pos_true = np.array([[s[0][0], s[0][1]] for s in sources_true])
    pos_rec = np.array([[s[0][0], s[0][1]] for s in sources_rec])
    
    cost = cdist(pos_true, pos_rec)
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_dist = cost[row_ind, col_ind]
    
    return np.sqrt(np.mean(matched_dist**2))


# =============================================================================
# TEST 1: STAR - Conformal L-BFGS-B
# =============================================================================

def diagnose_star_conformal():
    """Diagnose Star Conformal L-BFGS-B failure."""
    print("\n" + "="*70)
    print("FAILURE 1: Star - Conformal L-BFGS-B")
    print("="*70)
    
    sources = create_domain_sources('star')
    cmap = get_conformal_map('star')
    n_sources = len(sources)
    
    print(f"\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources):
        print(f"  {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
    
    # Generate data
    forward = ConformalForwardSolver(cmap, n_boundary=100)
    u_data = forward.solve(sources)
    
    # Test 1a: Gradient at truth
    print(f"\n--- Test 1a: Gradient at Truth ---")
    inverse = ConformalNonlinearInverseSolver(cmap, n_sources=n_sources, n_boundary=100)
    
    params = []
    for (x, y), q in sources:
        params.extend([x, y])
    for (x, y), q in sources:
        params.append(q)
    params = np.array(params)
    
    obj_at_truth = inverse._objective(params, u_data)
    
    # Numerical gradient
    eps = 1e-6
    grad = np.zeros_like(params)
    for i in range(len(params)):
        p_plus = params.copy()
        p_plus[i] += eps
        p_minus = params.copy()
        p_minus[i] -= eps
        grad[i] = (inverse._objective(p_plus, u_data) - inverse._objective(p_minus, u_data)) / (2*eps)
    
    print(f"  Objective at truth: {obj_at_truth:.6e}")
    print(f"  ||gradient||: {np.linalg.norm(grad):.6e}")
    print(f"  Max |grad component|: {np.max(np.abs(grad)):.6e}")
    
    if np.linalg.norm(grad) > 1e3:
        print(f"  ⚠️ LARGE GRADIENT - similar to ellipse branch cut issue?")
    
    # Test 1b: Seed sensitivity
    print(f"\n--- Test 1b: Seed Sensitivity (L-BFGS-B, n_restarts=5) ---")
    print(f"  {'Seed':<6} {'RMSE':<12} {'Residual':<12} {'Status'}")
    print("  " + "-"*45)
    
    for seed in range(40, 50):
        inverse = ConformalNonlinearInverseSolver(cmap, n_sources=n_sources, n_boundary=100)
        try:
            sources_rec, residual = inverse.solve(u_data, method='L-BFGS-B', n_restarts=5, seed=seed)
            rmse = compute_position_rmse(sources, sources_rec)
            status = "✅" if rmse < 0.05 else "❌"
            print(f"  {seed:<6} {rmse:<12.6f} {residual:<12.4e} {status}")
        except Exception as e:
            print(f"  {seed:<6} ERROR: {e}")
    
    # Test 1c: More restarts
    print(f"\n--- Test 1c: More Restarts (seed=42) ---")
    for n_restarts in [5, 10, 20, 50]:
        inverse = ConformalNonlinearInverseSolver(cmap, n_sources=n_sources, n_boundary=100)
        sources_rec, residual = inverse.solve(u_data, method='L-BFGS-B', n_restarts=n_restarts, seed=42)
        rmse = compute_position_rmse(sources, sources_rec)
        status = "✅" if rmse < 0.05 else "❌"
        print(f"  n_restarts={n_restarts:<3}: RMSE={rmse:.6f}, residual={residual:.4e} {status}")


# =============================================================================
# TEST 2: SQUARE - FEM L-BFGS-B
# =============================================================================

def diagnose_square_fem():
    """Diagnose Square FEM L-BFGS-B failure."""
    print("\n" + "="*70)
    print("FAILURE 2: Square - FEM L-BFGS-B")
    print("="*70)
    
    try:
        from fem_solver import FEMNonlinearInverseSolver, FEMForwardSolver
        from mesh import create_polygon_mesh
    except ImportError as e:
        print(f"  Import error: {e}")
        return
    
    sources = create_domain_sources('square')
    n_sources = len(sources)
    vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    
    print(f"\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources):
        print(f"  {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
    
    # Generate data
    mesh_data = create_polygon_mesh(vertices, 0.1)
    forward = FEMForwardSolver(resolution=0.1, verbose=False, mesh_data=mesh_data)
    u_data = forward.solve(sources)
    
    # Test 2a: Seed sensitivity
    print(f"\n--- Test 2a: Seed Sensitivity (L-BFGS-B, n_restarts=5) ---")
    print(f"  {'Seed':<6} {'RMSE':<12} {'Residual':<12} {'Status'}")
    print("  " + "-"*45)
    
    for seed in range(40, 50):
        try:
            inverse = FEMNonlinearInverseSolver(n_sources=n_sources, resolution=0.1,
                                                 verbose=False, mesh_data=mesh_data)
            inverse.set_measured_data(u_data)
            np.random.seed(seed)
            result = inverse.solve(method='L-BFGS-B', n_restarts=5, maxiter=1000)
            sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
            rmse = compute_position_rmse(sources, sources_rec)
            status = "✅" if rmse < 0.05 else "❌"
            print(f"  {seed:<6} {rmse:<12.6f} {result.residual:<12.4e} {status}")
        except Exception as e:
            print(f"  {seed:<6} ERROR: {e}")
    
    # Test 2b: More restarts
    print(f"\n--- Test 2b: More Restarts (seed=42) ---")
    for n_restarts in [5, 10, 20, 50]:
        try:
            inverse = FEMNonlinearInverseSolver(n_sources=n_sources, resolution=0.1,
                                                 verbose=False, mesh_data=mesh_data)
            inverse.set_measured_data(u_data)
            np.random.seed(42)
            result = inverse.solve(method='L-BFGS-B', n_restarts=n_restarts, maxiter=1000)
            sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
            rmse = compute_position_rmse(sources, sources_rec)
            status = "✅" if rmse < 0.05 else "❌"
            print(f"  n_restarts={n_restarts:<3}: RMSE={rmse:.6f}, residual={result.residual:.4e} {status}")
        except Exception as e:
            print(f"  n_restarts={n_restarts:<3}: ERROR: {e}")
    
    # Test 2c: Compare with diff_evol on same seed
    print(f"\n--- Test 2c: L-BFGS-B vs DiffEvol (seed=42) ---")
    for method in ['L-BFGS-B', 'differential_evolution']:
        try:
            inverse = FEMNonlinearInverseSolver(n_sources=n_sources, resolution=0.1,
                                                 verbose=False, mesh_data=mesh_data)
            inverse.set_measured_data(u_data)
            np.random.seed(42)
            if method == 'L-BFGS-B':
                result = inverse.solve(method=method, n_restarts=5, maxiter=1000)
            else:
                result = inverse.solve(method=method, maxiter=500)
            sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
            rmse = compute_position_rmse(sources, sources_rec)
            status = "✅" if rmse < 0.05 else "❌"
            print(f"  {method:<25}: RMSE={rmse:.6f} {status}")
        except Exception as e:
            print(f"  {method:<25}: ERROR: {e}")


# =============================================================================
# TEST 3: BRAIN - FEM diff_evol
# =============================================================================

def diagnose_brain_fem():
    """Diagnose Brain FEM diff_evol failure."""
    print("\n" + "="*70)
    print("FAILURE 3: Brain - FEM diff_evol")
    print("="*70)
    
    try:
        from fem_solver import FEMNonlinearInverseSolver, FEMForwardSolver
        from mesh import create_brain_mesh
    except ImportError as e:
        print(f"  Import error: {e}")
        return
    
    sources = create_domain_sources('brain')
    n_sources = len(sources)
    
    print(f"\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources):
        print(f"  {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
    
    # Generate data
    mesh_data = create_brain_mesh(0.1)
    forward = FEMForwardSolver(resolution=0.1, verbose=False, mesh_data=mesh_data)
    u_data = forward.solve(sources)
    
    # Test 3a: Seed sensitivity for diff_evol
    print(f"\n--- Test 3a: Seed Sensitivity (diff_evol) ---")
    print(f"  {'Seed':<6} {'RMSE':<12} {'Residual':<12} {'Status'}")
    print("  " + "-"*45)
    
    for seed in range(40, 50):
        try:
            inverse = FEMNonlinearInverseSolver(n_sources=n_sources, resolution=0.1,
                                                 verbose=False, mesh_data=mesh_data)
            inverse.set_measured_data(u_data)
            np.random.seed(seed)
            result = inverse.solve(method='differential_evolution', maxiter=500)
            sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
            rmse = compute_position_rmse(sources, sources_rec)
            status = "✅" if rmse < 0.05 else "❌"
            print(f"  {seed:<6} {rmse:<12.6f} {result.residual:<12.4e} {status}")
        except Exception as e:
            print(f"  {seed:<6} ERROR: {e}")
    
    # Test 3b: More iterations
    print(f"\n--- Test 3b: More Iterations (seed=42) ---")
    for maxiter in [500, 1000, 2000]:
        try:
            inverse = FEMNonlinearInverseSolver(n_sources=n_sources, resolution=0.1,
                                                 verbose=False, mesh_data=mesh_data)
            inverse.set_measured_data(u_data)
            np.random.seed(42)
            result = inverse.solve(method='differential_evolution', maxiter=maxiter)
            sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
            rmse = compute_position_rmse(sources, sources_rec)
            status = "✅" if rmse < 0.05 else "❌"
            print(f"  maxiter={maxiter:<5}: RMSE={rmse:.6f}, residual={result.residual:.4e} {status}")
        except Exception as e:
            print(f"  maxiter={maxiter:<5}: ERROR: {e}")
    
    # Test 3c: Compare with L-BFGS-B
    print(f"\n--- Test 3c: L-BFGS-B vs DiffEvol (seed=42) ---")
    for method in ['L-BFGS-B', 'differential_evolution']:
        try:
            inverse = FEMNonlinearInverseSolver(n_sources=n_sources, resolution=0.1,
                                                 verbose=False, mesh_data=mesh_data)
            inverse.set_measured_data(u_data)
            np.random.seed(42)
            if method == 'L-BFGS-B':
                result = inverse.solve(method=method, n_restarts=5, maxiter=1000)
            else:
                result = inverse.solve(method=method, maxiter=500)
            sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
            rmse = compute_position_rmse(sources, sources_rec)
            status = "✅" if rmse < 0.05 else "❌"
            print(f"  {method:<25}: RMSE={rmse:.6f} {status}")
        except Exception as e:
            print(f"  {method:<25}: ERROR: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("DIAGNOSING THREE FAILING SOLVERS")
    print("="*70)
    
    diagnose_star_conformal()
    diagnose_square_fem()
    diagnose_brain_fem()
    
    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
