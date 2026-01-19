#!/usr/bin/env python3
"""
Test script for IPOPT-based nonlinear inverse solver.

Run this locally where cyipopt is installed via conda-forge:
    conda install -c conda-forge cyipopt
    python test_ipopt_basic.py

Expected result: Mean position error < 1e-5 for well-separated sources.
"""

import numpy as np
import sys

# Check cyipopt first
try:
    import cyipopt
    print(f"✓ cyipopt available (version: {cyipopt.__version__})")
except ImportError:
    print("✗ cyipopt not found!")
    print("Install via: conda install -c conda-forge cyipopt")
    sys.exit(1)

from ipopt_solver import (
    IPOPTNonlinearInverseSolver,
    check_cyipopt_available,
    get_ipopt_version
)
from analytical_solver import AnalyticalForwardSolver
from scipy.optimize import linear_sum_assignment


def create_well_separated_sources(n_sources, r_range=(0.6, 0.8), seed=42):
    """Create well-separated test sources."""
    np.random.seed(seed)
    
    sources = []
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += 0.1 * np.random.randn(n_sources)
    
    for i, theta in enumerate(angles):
        r = np.random.uniform(r_range[0], r_range[1])
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        intensity = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), intensity))
    
    # Enforce sum = 0
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    
    return sources


def compute_position_error(sources_true, sources_recovered):
    """Compute mean position error using optimal assignment."""
    n = len(sources_true)
    cost = np.zeros((n, n))
    
    for i, ((tx, ty), _) in enumerate(sources_true):
        for j, s in enumerate(sources_recovered):
            cost[i, j] = np.sqrt((tx - s.x)**2 + (ty - s.y)**2)
    
    row_ind, col_ind = linear_sum_assignment(cost)
    return cost[row_ind, col_ind].mean()


def test_disk_solver(n_sources=4, n_boundary=100, n_restarts=10):
    """Test IPOPT solver on disk domain."""
    print(f"\n{'='*60}")
    print(f"Test: {n_sources} sources on disk")
    print(f"{'='*60}")
    
    # Create test problem
    sources_true = create_well_separated_sources(n_sources)
    
    print("\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources_true):
        print(f"  {i+1}: ({x:+.4f}, {y:+.4f}), q={q:+.4f}")
    
    # Generate data
    forward = AnalyticalForwardSolver(n_boundary)
    u_measured = forward.solve(sources_true)
    
    # Solve
    print(f"\nSolving with IPOPT ({n_restarts} restarts)...")
    solver = IPOPTNonlinearInverseSolver(n_sources=n_sources, n_boundary=n_boundary)
    solver.set_measured_data(u_measured)
    result = solver.solve(n_restarts=n_restarts, verbose=True)
    
    # Results
    print("\nRecovered sources:")
    for i, s in enumerate(result.sources):
        print(f"  {i+1}: ({s.x:+.4f}, {s.y:+.4f}), q={s.intensity:+.4f}")
    
    error = compute_position_error(sources_true, result.sources)
    
    print(f"\n{'─'*40}")
    print(f"Mean position error: {error:.2e}")
    print(f"Residual (RMS):      {result.residual:.2e}")
    print(f"Success:             {result.success}")
    
    if error < 1e-5:
        print("✓ PASSED: Error < 1e-5")
        return True
    elif error < 1e-3:
        print("~ ACCEPTABLE: Error < 1e-3")
        return True
    else:
        print("✗ FAILED: Error > 1e-3")
        return False


def test_conformal_solver(domain='ellipse', n_sources=4, n_boundary=100, n_restarts=10):
    """Test IPOPT solver on conformal domain."""
    print(f"\n{'='*60}")
    print(f"Test: {n_sources} sources on {domain}")
    print(f"{'='*60}")
    
    from conformal_solver import create_conformal_map, ConformalForwardSolver
    from ipopt_solver import IPOPTConformalInverseSolver
    
    # Create conformal map
    if domain == 'ellipse':
        cmap = create_conformal_map('ellipse', a=1.5, b=1.0)
    elif domain == 'star':
        cmap = create_conformal_map('star', n_points=5, r_inner=0.5, r_outer=1.0)
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    # Create sources inside domain
    np.random.seed(42)
    sources_true = []
    
    # Sample from domain interior
    boundary = cmap.boundary_physical(100)
    cx, cy = np.real(boundary).mean(), np.imag(boundary).mean()
    
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    for i, theta in enumerate(angles):
        # Try to place source inside domain
        for r in np.linspace(0.5, 0.1, 10):
            x = cx + r * np.cos(theta) * (np.real(boundary).max() - cx)
            y = cy + r * np.sin(theta) * (np.imag(boundary).max() - cy)
            if cmap.is_inside(complex(x, y)):
                intensity = 1.0 if i % 2 == 0 else -1.0
                sources_true.append(((x, y), intensity))
                break
    
    # Enforce sum = 0
    total = sum(s[1] for s in sources_true)
    sources_true[-1] = (sources_true[-1][0], sources_true[-1][1] - total)
    
    print("\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources_true):
        print(f"  {i+1}: ({x:+.4f}, {y:+.4f}), q={q:+.4f}")
    
    # Generate data
    forward = ConformalForwardSolver(cmap, n_boundary)
    u_measured = forward.solve(sources_true)
    
    # Solve
    print(f"\nSolving with IPOPT ({n_restarts} restarts)...")
    solver = IPOPTConformalInverseSolver(cmap, n_sources=n_sources, n_boundary=n_boundary)
    solver.set_measured_data(u_measured)
    result = solver.solve(n_restarts=n_restarts, verbose=True)
    
    # Results
    print("\nRecovered sources:")
    for i, s in enumerate(result.sources):
        print(f"  {i+1}: ({s.x:+.4f}, {s.y:+.4f}), q={s.intensity:+.4f}")
    
    error = compute_position_error(sources_true, result.sources)
    
    print(f"\n{'─'*40}")
    print(f"Mean position error: {error:.2e}")
    print(f"Residual (RMS):      {result.residual:.2e}")
    print(f"Success:             {result.success}")
    
    if error < 1e-4:
        print("✓ PASSED: Error < 1e-4")
        return True
    elif error < 1e-2:
        print("~ ACCEPTABLE: Error < 1e-2")
        return True
    else:
        print("✗ FAILED: Error > 1e-2")
        return False


if __name__ == "__main__":
    print("IPOPT Inverse Solver Tests")
    print("=" * 60)
    
    results = []
    
    # Test 1: 2 sources on disk (should be easy)
    results.append(("2 sources, disk", test_disk_solver(n_sources=2, n_restarts=5)))
    
    # Test 2: 4 sources on disk (standard test)
    results.append(("4 sources, disk", test_disk_solver(n_sources=4, n_restarts=10)))
    
    # Test 3: 4 sources on ellipse (conformal)
    try:
        results.append(("4 sources, ellipse", test_conformal_solver(domain='ellipse', n_sources=4, n_restarts=10)))
    except Exception as e:
        print(f"\nEllipse test failed with error: {e}")
        results.append(("4 sources, ellipse", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed - see details above")
    print("=" * 60)
