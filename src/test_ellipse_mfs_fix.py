#!/usr/bin/env python3
"""
Test if using MFSConformalMap instead of EllipseMap fixes the ellipse problem.

The hypothesis is that EllipseMap (Joukowsky transform) has a branch cut issue
that causes sources on the real axis to fail. MFSConformalMap uses numerical
Laplace solving which should not have this problem.

Usage:
    cd src/
    python test_ellipse_mfs_fix.py
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from conformal_solver import (ConformalForwardSolver, ConformalNonlinearInverseSolver,
                               EllipseMap, MFSConformalMap)
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


def create_ellipse_boundary_func(a, b):
    """Create ellipse boundary parameterization function for MFSConformalMap."""
    def gamma(t):
        return a * np.cos(t) + 1j * b * np.sin(t)
    return gamma


def test_gradient_at_truth(cmap, sources, map_name):
    """Check if gradient is well-behaved at the true solution."""
    forward = ConformalForwardSolver(cmap, n_boundary=100)
    u_data = forward.solve(sources)
    
    inverse = ConformalNonlinearInverseSolver(cmap, n_sources=len(sources), n_boundary=100)
    
    # Build params
    params = []
    for (x, y), q in sources:
        params.extend([x, y])
    for (x, y), q in sources:
        params.append(q)
    params = np.array(params)
    
    # Compute gradient numerically
    eps = 1e-6
    grad = np.zeros_like(params)
    f0 = inverse._objective(params, u_data)
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        params_minus = params.copy()
        params_minus[i] -= eps
        grad[i] = (inverse._objective(params_plus, u_data) - 
                   inverse._objective(params_minus, u_data)) / (2 * eps)
    
    print(f"\n  {map_name}:")
    print(f"    Objective at truth: {f0:.6e}")
    print(f"    ||gradient||: {np.linalg.norm(grad):.6e}")
    print(f"    Max |gradient component|: {np.max(np.abs(grad)):.6e}")
    
    # Show gradient for y1, y2 (the problematic ones for EllipseMap)
    print(f"    ∂f/∂y1 = {grad[1]:.6e}")
    print(f"    ∂f/∂y2 = {grad[3]:.6e}")
    
    return np.linalg.norm(grad)


def test_solve(cmap, sources, map_name):
    """Test actual solving with L-BFGS-B and diff_evol."""
    forward = ConformalForwardSolver(cmap, n_boundary=100)
    u_data = forward.solve(sources)
    
    results = {}
    
    # L-BFGS-B
    print(f"\n  {map_name} - L-BFGS-B (n_restarts=5):")
    inverse = ConformalNonlinearInverseSolver(cmap, n_sources=len(sources), n_boundary=100)
    try:
        sources_rec, residual = inverse.solve(u_data, method='L-BFGS-B', n_restarts=5, seed=42)
        rmse = compute_position_rmse(sources, sources_rec)
        print(f"    Residual: {residual:.6e}")
        print(f"    Position RMSE: {rmse:.6f}")
        results['lbfgs'] = rmse
        
        if rmse < 0.01:
            print(f"    ✅ SUCCESS")
        else:
            print(f"    ❌ FAILED (RMSE > 0.01)")
    except Exception as e:
        print(f"    ERROR: {e}")
        results['lbfgs'] = float('inf')
    
    # Differential evolution
    print(f"\n  {map_name} - differential_evolution:")
    inverse = ConformalNonlinearInverseSolver(cmap, n_sources=len(sources), n_boundary=100)
    try:
        sources_rec, residual = inverse.solve(u_data, method='differential_evolution', seed=42)
        rmse = compute_position_rmse(sources, sources_rec)
        print(f"    Residual: {residual:.6e}")
        print(f"    Position RMSE: {rmse:.6f}")
        results['de'] = rmse
        
        if rmse < 0.01:
            print(f"    ✅ SUCCESS")
        else:
            print(f"    ❌ FAILED (RMSE > 0.01)")
    except Exception as e:
        print(f"    ERROR: {e}")
        results['de'] = float('inf')
    
    return results


def main():
    a, b = 2.0, 1.0
    
    # The 4-source case that fails with EllipseMap
    sources_4 = [
        ((1.0, 0.0), 1.0),
        ((-1.0, 0.0), -1.0),
        ((0.0, 0.5), 1.0),
        ((0.0, -0.5), -1.0),
    ]
    
    print("="*70)
    print("TEST: EllipseMap vs MFSConformalMap for Ellipse Domain")
    print("="*70)
    print(f"\nEllipse: a={a}, b={b}")
    print(f"Focal distance c = sqrt(a²-b²) = {np.sqrt(a**2 - b**2):.4f}")
    print(f"\nSources:")
    for i, ((x, y), q) in enumerate(sources_4):
        print(f"  {i+1}: ({x}, {y}), q={q}")
    
    # Create both maps
    print("\n" + "="*70)
    print("Creating conformal maps...")
    print("="*70)
    
    # EllipseMap (Joukowsky - has branch cut issue)
    ellipse_map = EllipseMap(a=a, b=b)
    print(f"\nEllipseMap created (Joukowsky transform)")
    
    # MFSConformalMap (numerical - no branch cut)
    ellipse_gamma = create_ellipse_boundary_func(a, b)
    mfs_map = MFSConformalMap(ellipse_gamma, n_boundary=200, n_charge=80)
    print(f"MFSConformalMap created (numerical, 200 boundary pts, 80 charges)")
    
    # Test boundary mapping quality
    print("\n" + "="*70)
    print("TEST 1: Boundary Mapping Quality")
    print("="*70)
    
    test_theta = np.linspace(0, 2*np.pi, 20, endpoint=False)
    test_boundary = a * np.cos(test_theta) + 1j * b * np.sin(test_theta)
    
    print("\n  EllipseMap boundary |w|:")
    w_ellipse = ellipse_map.to_disk(test_boundary)
    print(f"    min={np.abs(w_ellipse).min():.6f}, max={np.abs(w_ellipse).max():.6f}")
    
    print("\n  MFSConformalMap boundary |w|:")
    w_mfs = mfs_map.to_disk(test_boundary)
    print(f"    min={np.abs(w_mfs).min():.6f}, max={np.abs(w_mfs).max():.6f}")
    
    # Test gradient at truth
    print("\n" + "="*70)
    print("TEST 2: Gradient at True Solution")
    print("="*70)
    
    grad_ellipse = test_gradient_at_truth(ellipse_map, sources_4, "EllipseMap")
    grad_mfs = test_gradient_at_truth(mfs_map, sources_4, "MFSConformalMap")
    
    # Test actual solving
    print("\n" + "="*70)
    print("TEST 3: Actual Nonlinear Solve")
    print("="*70)
    
    results_ellipse = test_solve(ellipse_map, sources_4, "EllipseMap")
    results_mfs = test_solve(mfs_map, sources_4, "MFSConformalMap")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'Map':<20} {'||grad||':<15} {'L-BFGS-B RMSE':<15} {'Diff Evol RMSE':<15}")
    print("-"*70)
    print(f"{'EllipseMap':<20} {grad_ellipse:<15.2e} {results_ellipse['lbfgs']:<15.6f} {results_ellipse['de']:<15.6f}")
    print(f"{'MFSConformalMap':<20} {grad_mfs:<15.2e} {results_mfs['lbfgs']:<15.6f} {results_mfs['de']:<15.6f}")
    
    print("\n" + "="*70)
    if results_mfs['lbfgs'] < 0.01 and results_ellipse['lbfgs'] > 0.1:
        print("✅ CONFIRMED: MFSConformalMap FIXES the ellipse problem!")
        print("   Recommendation: Use MFSConformalMap instead of EllipseMap")
    elif results_mfs['lbfgs'] > 0.1:
        print("❌ MFSConformalMap does NOT fix the problem - investigate further")
    else:
        print("Both maps work - the issue may be configuration-specific")
    print("="*70)


if __name__ == '__main__':
    main()
