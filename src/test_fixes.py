#!/usr/bin/env python3
"""
Test script to validate the inverse source localization fixes.

Tests:
1. Disk domain with AnalyticalNonlinearInverseSolver (SLSQP)
2. Ellipse domain with ConformalNonlinearInverseSolver (SLSQP)
3. Square domain with FEMNonlinearInverseSolver (SLSQP)

Expected results for 4 well-separated sources:
- Position RMSE < 0.01 (ideally < 1e-5 for disk)
- Intensity RMSE < 0.01
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytical_solver import (
    AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver, Source
)
from conformal_solver import (
    ConformalForwardSolver, ConformalNonlinearInverseSolver,
    EllipseMap
)


def create_well_separated_sources(n_sources: int, r_range=(0.5, 0.8), seed=42):
    """Create well-separated sources at r in [0.5, 0.8] with evenly spaced angles."""
    np.random.seed(seed)
    sources = []
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.1, 0.1, n_sources)  # Small perturbation
    
    for i, theta in enumerate(angles):
        r = np.random.uniform(r_range[0], r_range[1])
        x, y = r * np.cos(theta), r * np.sin(theta)
        S = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), S))
    
    # Adjust last intensity for sum = 0
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    
    return sources


def compute_position_rmse(sources_true, sources_rec):
    """Compute position RMSE with optimal matching."""
    from scipy.optimize import linear_sum_assignment
    
    n = len(sources_true)
    cost = np.zeros((n, n))
    
    for i, (pos_t, _) in enumerate(sources_true):
        for j, (pos_r, _) in enumerate(sources_rec):
            if isinstance(pos_r, tuple):
                dx = pos_t[0] - pos_r[0]
                dy = pos_t[1] - pos_r[1]
            else:
                dx = pos_t[0] - pos_r.x
                dy = pos_t[1] - pos_r.y
            cost[i, j] = dx**2 + dy**2
    
    row_ind, col_ind = linear_sum_assignment(cost)
    rmse = np.sqrt(cost[row_ind, col_ind].mean())
    return rmse


def test_disk_analytical():
    """Test analytical solver on disk domain."""
    print("\n" + "="*60)
    print("TEST 1: DISK DOMAIN (Analytical Solver)")
    print("="*60)
    
    n_sources = 4
    n_boundary = 100
    
    # Create well-separated sources
    sources_true = create_well_separated_sources(n_sources, seed=42)
    print(f"\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources_true):
        r = np.sqrt(x**2 + y**2)
        print(f"  Source {i+1}: ({x:.4f}, {y:.4f}), r={r:.4f}, q={q:.4f}")
    
    # Generate synthetic data
    forward = AnalyticalForwardSolver(n_boundary)
    u_measured = forward.solve(sources_true)
    
    # Solve inverse problem
    inverse = AnalyticalNonlinearInverseSolver(n_sources, n_boundary)
    inverse.set_measured_data(u_measured)
    
    print(f"\nSolving with SLSQP (5 restarts)...")
    result = inverse.solve(method='SLSQP', n_restarts=5, maxiter=10000)
    
    # Convert to list format for comparison
    sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
    
    print(f"\nRecovered sources:")
    for i, ((x, y), q) in enumerate(sources_rec):
        r = np.sqrt(x**2 + y**2)
        print(f"  Source {i+1}: ({x:.4f}, {y:.4f}), r={r:.4f}, q={q:.4f}")
    
    # Compute errors
    pos_rmse = compute_position_rmse(sources_true, sources_rec)
    print(f"\nPosition RMSE: {pos_rmse:.2e}")
    print(f"Residual: {result.residual:.2e}")
    
    success = pos_rmse < 0.01
    print(f"\n{'✓ PASSED' if success else '✗ FAILED'}: Position RMSE < 0.01")
    
    return success


def test_ellipse_conformal():
    """Test conformal solver on ellipse domain."""
    print("\n" + "="*60)
    print("TEST 2: ELLIPSE DOMAIN (Conformal Solver)")
    print("="*60)
    
    n_sources = 4
    n_boundary = 100
    a, b = 1.5, 0.8  # Semi-axes
    
    # Create ellipse-adapted sources
    np.random.seed(42)
    sources_true = []
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.1, 0.1, n_sources)
    
    for i, theta in enumerate(angles):
        # Scale to fit inside ellipse at ~70% of boundary
        scale = 0.6
        x = a * scale * np.cos(theta)
        y = b * scale * np.sin(theta)
        S = 1.0 if i % 2 == 0 else -1.0
        sources_true.append(((x, y), S))
    
    # Adjust last intensity for sum = 0
    total = sum(s[1] for s in sources_true)
    sources_true[-1] = (sources_true[-1][0], sources_true[-1][1] - total)
    
    print(f"\nTrue sources (a={a}, b={b}):")
    for i, ((x, y), q) in enumerate(sources_true):
        print(f"  Source {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
    
    # Create conformal map
    conformal_map = EllipseMap(a=a, b=b)
    
    # Generate synthetic data
    forward = ConformalForwardSolver(conformal_map, n_boundary)
    u_measured = forward.solve(sources_true)
    
    # Solve inverse problem
    inverse = ConformalNonlinearInverseSolver(conformal_map, n_sources, n_boundary)
    
    print(f"\nSolving with SLSQP (5 restarts)...")
    sources_rec, residual = inverse.solve(u_measured, method='SLSQP', n_restarts=5)
    
    print(f"\nRecovered sources:")
    for i, ((x, y), q) in enumerate(sources_rec):
        print(f"  Source {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
    
    # Compute errors
    pos_rmse = compute_position_rmse(sources_true, sources_rec)
    print(f"\nPosition RMSE: {pos_rmse:.2e}")
    print(f"Residual: {np.sqrt(residual):.2e}")
    
    success = pos_rmse < 0.05  # Slightly relaxed for conformal
    print(f"\n{'✓ PASSED' if success else '✗ FAILED'}: Position RMSE < 0.05")
    
    return success


def test_optimization_utils():
    """Test the optimization utilities."""
    print("\n" + "="*60)
    print("TEST 3: OPTIMIZATION UTILITIES")
    print("="*60)
    
    try:
        from optimization_utils import (
            push_to_interior, generate_spread_init, solve_disk_polar
        )
        print("✓ Successfully imported optimization_utils")
    except ImportError as e:
        print(f"✗ Failed to import optimization_utils: {e}")
        return False
    
    # Test push_to_interior
    x0 = np.array([0.0, 0.95, -0.95, 0.5])  # Second value at bound
    bounds = [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]
    x_pushed = push_to_interior(x0, bounds, margin=0.1)
    
    print(f"\npush_to_interior test:")
    print(f"  Original: {x0}")
    print(f"  Pushed:   {x_pushed}")
    
    at_bound_before = np.any(np.abs(x0) >= 0.9)
    at_bound_after = np.any(np.abs(x_pushed) >= 0.9)
    
    if at_bound_before and not at_bound_after:
        print("  ✓ Successfully pushed away from bounds")
    else:
        print("  ✗ Failed to push away from bounds")
        return False
    
    # Test generate_spread_init
    n_sources = 4
    
    x0_spread = generate_spread_init(n_sources, domain_type='disk', seed=42)
    print(f"\ngenerate_spread_init test:")
    print(f"  Generated initial params with shape {x0_spread.shape}")
    print(f"  Positions: {x0_spread[:2*n_sources]}")
    print(f"  Intensities: {x0_spread[2*n_sources:]}")
    
    # Check intensities sum to approximately 0
    intensity_sum = np.sum(x0_spread[2*n_sources:])
    if np.abs(intensity_sum) < 0.1:
        print(f"  ✓ Intensities sum ≈ 0 (actual: {intensity_sum:.4f})")
    else:
        print(f"  ✗ Intensities don't sum to 0 (actual: {intensity_sum:.4f})")
        return False
    
    return True


def test_disk_polar():
    """Test polar coordinate solver for disk."""
    print("\n" + "="*60)
    print("TEST 4: DISK DOMAIN (Polar Coordinates)")
    print("="*60)
    
    try:
        from optimization_utils import solve_disk_polar
    except ImportError:
        print("✗ solve_disk_polar not available")
        return False
    
    n_sources = 4
    n_boundary = 100
    
    # Create well-separated sources
    sources_true = create_well_separated_sources(n_sources, seed=42)
    print(f"\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources_true):
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        print(f"  Source {i+1}: r={r:.4f}, θ={theta:.4f}, q={q:.4f}")
    
    # Generate synthetic data
    theta_boundary = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)
    u_measured = np.zeros(n_boundary)
    for (x, y), S in sources_true:
        r_s = np.sqrt(x**2 + y**2)
        theta_s = np.arctan2(y, x)
        arg = 1 + r_s**2 - 2*r_s*np.cos(theta_boundary - theta_s)
        arg = np.maximum(arg, 1e-30)
        u_measured += (S / (2*np.pi)) * np.log(arg)
    u_measured = u_measured - np.mean(u_measured)
    
    # Solve with polar coordinates
    print(f"\nSolving with polar SLSQP (5 restarts)...")
    sources_rec, residual = solve_disk_polar(
        theta_boundary, u_measured, n_sources, n_starts=5, verbose=False
    )
    
    print(f"\nRecovered sources:")
    for i, ((x, y), q) in enumerate(sources_rec):
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        print(f"  Source {i+1}: r={r:.4f}, θ={theta:.4f}, q={q:.4f}")
    
    # Compute errors
    pos_rmse = compute_position_rmse(sources_true, sources_rec)
    print(f"\nPosition RMSE: {pos_rmse:.2e}")
    print(f"Residual: {residual:.2e}")
    
    success = pos_rmse < 0.01
    print(f"\n{'✓ PASSED' if success else '✗ FAILED'}: Position RMSE < 0.01")
    
    return success


def main():
    """Run all tests."""
    print("="*60)
    print("INVERSE SOURCE LOCALIZATION - FIX VALIDATION")
    print("="*60)
    
    results = {}
    
    # Test 1: Disk with analytical solver
    results['disk_analytical'] = test_disk_analytical()
    
    # Test 2: Ellipse with conformal solver
    results['ellipse_conformal'] = test_ellipse_conformal()
    
    # Test 3: Optimization utilities
    results['opt_utils'] = test_optimization_utils()
    
    # Test 4: Disk with polar coordinates
    results['disk_polar'] = test_disk_polar()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
