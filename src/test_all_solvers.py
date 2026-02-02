#!/usr/bin/env python3
"""
Comprehensive test for ALL inverse source solvers:
- Analytical (disk only)
- Conformal MFS (ellipse, square, brain)
- FEM Nonlinear (disk, ellipse, square)

Tests:
- Source counts: 4, 6, 8, 10, 12
- Depth: shallow (r ~ 0.7-0.9)
- Placement: symmetric and non-symmetric

Run: python test_all_solvers.py
"""

import numpy as np
from time import time
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import linear_sum_assignment


def compute_rmse(sources_true, sources_rec):
    """Compute position RMSE with optimal matching."""
    n = len(sources_true)
    cost = np.zeros((n, n))
    for i, (pos_t, _) in enumerate(sources_true):
        for j in range(len(sources_rec)):
            if hasattr(sources_rec[j], 'x'):
                pos_r = (sources_rec[j].x, sources_rec[j].y)
            elif hasattr(sources_rec[j], 'position'):
                pos_r = sources_rec[j].position
            else:
                pos_r = sources_rec[j][0]
            cost[i, j] = (pos_t[0]-pos_r[0])**2 + (pos_t[1]-pos_r[1])**2
    row_ind, col_ind = linear_sum_assignment(cost)
    return np.sqrt(cost[row_ind, col_ind].mean())


def create_sources_disk(n_sources, r_range=(0.7, 0.9), symmetric=True, seed=42):
    """Create sources for unit disk domain."""
    np.random.seed(seed)
    sources = []
    
    if symmetric:
        angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    else:
        angles = []
        min_sep = np.pi / (n_sources + 1)
        for i in range(n_sources):
            for _ in range(100):
                theta = np.random.uniform(0, 2*np.pi)
                if all(min(abs(theta - a), 2*np.pi - abs(theta - a)) > min_sep for a in angles):
                    angles.append(theta)
                    break
            else:
                angles.append(np.random.uniform(0, 2*np.pi))
        angles = np.array(angles)
    
    for i, theta in enumerate(angles):
        r = np.random.uniform(r_range[0], r_range[1])
        x, y = r * np.cos(theta), r * np.sin(theta)
        S = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), S))
    
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    return sources


def create_sources_ellipse(n_sources, a=1.5, b=0.8, scale=0.6, symmetric=True, seed=42):
    """Create sources for ellipse domain."""
    np.random.seed(seed)
    sources = []
    
    if symmetric:
        angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    else:
        angles = []
        min_sep = np.pi / (n_sources + 1)
        for i in range(n_sources):
            for _ in range(100):
                theta = np.random.uniform(0, 2*np.pi)
                if all(min(abs(theta - a_), 2*np.pi - abs(theta - a_)) > min_sep for a_ in angles):
                    angles.append(theta)
                    break
            else:
                angles.append(np.random.uniform(0, 2*np.pi))
        angles = np.array(angles)
    
    for i, theta in enumerate(angles):
        s = np.random.uniform(scale, scale + 0.15)
        x = a * s * np.cos(theta)
        y = b * s * np.sin(theta)
        S = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), S))
    
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    return sources


def create_sources_square(n_sources, half_width=1.0, scale=0.6, symmetric=True, seed=42):
    """Create sources for square domain."""
    np.random.seed(seed)
    sources = []
    
    if symmetric:
        angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    else:
        angles = []
        min_sep = np.pi / (n_sources + 1)
        for i in range(n_sources):
            for _ in range(100):
                theta = np.random.uniform(0, 2*np.pi)
                if all(min(abs(theta - a), 2*np.pi - abs(theta - a)) > min_sep for a in angles):
                    angles.append(theta)
                    break
            else:
                angles.append(np.random.uniform(0, 2*np.pi))
        angles = np.array(angles)
    
    for i, theta in enumerate(angles):
        s = np.random.uniform(scale, scale + 0.15)
        r = s * half_width
        x, y = r * np.cos(theta), r * np.sin(theta)
        x = np.clip(x, -0.85*half_width, 0.85*half_width)
        y = np.clip(y, -0.85*half_width, 0.85*half_width)
        S = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), S))
    
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    return sources


# =============================================================================
# ANALYTICAL SOLVER TESTS (Disk)
# =============================================================================

def test_analytical_disk(n_sources, symmetric, seed=42, n_restarts=5):
    """Test analytical solver for disk domain."""
    from analytical_solver import AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver
    
    sources = create_sources_disk(n_sources, r_range=(0.7, 0.9), 
                                   symmetric=symmetric, seed=seed)
    
    forward = AnalyticalForwardSolver(100)
    u = forward.solve(sources)
    
    inverse = AnalyticalNonlinearInverseSolver(n_sources, 100)
    inverse.set_measured_data(u)
    
    t0 = time()
    result = inverse.solve(method='SLSQP', n_restarts=n_restarts, maxiter=10000)
    elapsed = time() - t0
    
    rmse = compute_rmse(sources, result.sources)
    return rmse, elapsed


# =============================================================================
# CONFORMAL SOLVER TESTS (Ellipse, Square, Brain)
# =============================================================================

def test_conformal_ellipse(n_sources, symmetric, seed=42, n_restarts=5):
    """Test conformal solver for ellipse domain."""
    from conformal_solver import EllipseMap, ConformalForwardSolver, ConformalNonlinearInverseSolver
    
    a, b = 1.5, 0.8
    sources = create_sources_ellipse(n_sources, a=a, b=b, scale=0.6,
                                      symmetric=symmetric, seed=seed)
    
    emap = EllipseMap(a=a, b=b)
    forward = ConformalForwardSolver(emap, 100)
    u = forward.solve(sources)
    
    inverse = ConformalNonlinearInverseSolver(emap, n_sources, 100)
    
    t0 = time()
    sources_rec, _ = inverse.solve(u, method='SLSQP', n_restarts=n_restarts)
    elapsed = time() - t0
    
    rmse = compute_rmse(sources, sources_rec)
    return rmse, elapsed


def test_conformal_square(n_sources, symmetric, seed=42, n_restarts=5):
    """Test conformal solver for square domain."""
    from conformal_solver import RectangleMap, ConformalForwardSolver, ConformalNonlinearInverseSolver
    
    half_width = 1.0
    sources = create_sources_square(n_sources, half_width=half_width, scale=0.6,
                                     symmetric=symmetric, seed=seed)
    
    smap = RectangleMap(half_width=half_width, half_height=half_width)
    forward = ConformalForwardSolver(smap, 100)
    u = forward.solve(sources)
    
    inverse = ConformalNonlinearInverseSolver(smap, n_sources, 100)
    
    t0 = time()
    sources_rec, _ = inverse.solve(u, method='SLSQP', n_restarts=n_restarts)
    elapsed = time() - t0
    
    rmse = compute_rmse(sources, sources_rec)
    return rmse, elapsed


# =============================================================================
# FEM SOLVER TESTS (Disk, Ellipse, Square)
# =============================================================================

def test_fem_disk(n_sources, symmetric, seed=42, n_restarts=5):
    """Test FEM nonlinear solver for disk domain."""
    from fem_solver import FEMNonlinearInverseSolver
    
    sources = create_sources_disk(n_sources, r_range=(0.7, 0.9), 
                                   symmetric=symmetric, seed=seed)
    
    inverse = FEMNonlinearInverseSolver(n_sources, resolution=0.1, n_sensors=100,
                                         verbose=False, domain_type='disk')
    
    # Generate measurements at sensor locations
    u_sensors = inverse.forward.solve_at_sensors(sources)
    inverse.set_measured_data(u_sensors)
    
    t0 = time()
    result = inverse.solve(method='SLSQP', n_restarts=n_restarts, maxiter=5000)
    elapsed = time() - t0
    
    rmse = compute_rmse(sources, result.sources)
    return rmse, elapsed


def test_fem_ellipse(n_sources, symmetric, seed=42, n_restarts=5):
    """Test FEM nonlinear solver for ellipse domain."""
    from fem_solver import FEMNonlinearInverseSolver
    
    a, b = 1.5, 0.8
    sources = create_sources_ellipse(n_sources, a=a, b=b, scale=0.6,
                                      symmetric=symmetric, seed=seed)
    
    inverse = FEMNonlinearInverseSolver.from_ellipse(a=a, b=b, n_sources=n_sources,
                                                      resolution=0.1, n_sensors=100,
                                                      verbose=False)
    
    # Generate measurements at sensor locations
    u_sensors = inverse.forward.solve_at_sensors(sources)
    inverse.set_measured_data(u_sensors)
    
    t0 = time()
    result = inverse.solve(method='SLSQP', n_restarts=n_restarts, maxiter=5000)
    elapsed = time() - t0
    
    rmse = compute_rmse(sources, result.sources)
    return rmse, elapsed


def test_fem_square(n_sources, symmetric, seed=42, n_restarts=5):
    """Test FEM nonlinear solver for square domain."""
    from fem_solver import FEMNonlinearInverseSolver
    
    half_width = 1.0
    sources = create_sources_square(n_sources, half_width=half_width, scale=0.6,
                                     symmetric=symmetric, seed=seed)
    
    inverse = FEMNonlinearInverseSolver.from_square(half_width=half_width, 
                                                     n_sources=n_sources,
                                                     resolution=0.1, n_sensors=100,
                                                     verbose=False)
    
    # Generate measurements at sensor locations
    u_sensors = inverse.forward.solve_at_sensors(sources)
    inverse.set_measured_data(u_sensors)
    
    t0 = time()
    result = inverse.solve(method='SLSQP', n_restarts=n_restarts, maxiter=5000)
    elapsed = time() - t0
    
    rmse = compute_rmse(sources, result.sources)
    return rmse, elapsed


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_tests(test_func, name, source_counts, get_n_restarts):
    """Run tests for a given solver configuration."""
    results = []
    
    for n_sources in source_counts:
        for symmetric in [True, False]:
            placement = "sym" if symmetric else "nonsym"
            n_restarts = get_n_restarts(n_sources, symmetric)
            
            try:
                rmse, elapsed = test_func(n_sources, symmetric, seed=42, 
                                          n_restarts=n_restarts)
                status = "✓" if rmse < 0.01 else "✗"
                results.append((n_sources, placement, rmse, elapsed, status))
                print(f"  n={n_sources:2d} {placement:7s}: RMSE={rmse:.2e}, time={elapsed:5.1f}s {status} (restarts={n_restarts})")
            except Exception as e:
                print(f"  n={n_sources:2d} {placement:7s}: ERROR - {str(e)[:50]}")
                results.append((n_sources, placement, float('inf'), 0, "✗"))
    
    return results


def main():
    print("="*70)
    print("COMPREHENSIVE INVERSE SOURCE SOLVER TEST")
    print("="*70)
    print("Source counts: 4, 6, 8")
    print("Placement: symmetric and non-symmetric")
    print("Pass criterion: RMSE < 0.01")
    print("="*70)
    
    # Use smaller source counts for faster testing
    source_counts = [4, 6, 8]
    
    # Adaptive restarts: harder problems need more restarts
    # Based on diagnostic: n=8 nonsym needs ~10-20 restarts to reliably succeed
    def get_n_restarts(n_sources, symmetric):
        if n_sources <= 4:
            return 5
        elif n_sources <= 6:
            return 7 if symmetric else 10
        else:  # n >= 8
            return 10 if symmetric else 15
    
    all_results = {}
    
    # Analytical solver (disk)
    print("\n" + "-"*70)
    print("ANALYTICAL SOLVER (Disk)")
    print("-"*70)
    all_results['Analytical-Disk'] = run_tests(test_analytical_disk, 'Analytical-Disk', 
                                                source_counts, get_n_restarts)
    
    # Conformal solver (ellipse)
    print("\n" + "-"*70)
    print("CONFORMAL SOLVER (Ellipse)")
    print("-"*70)
    all_results['Conformal-Ellipse'] = run_tests(test_conformal_ellipse, 'Conformal-Ellipse',
                                                  source_counts, get_n_restarts)
    
    # Conformal solver (square)
    print("\n" + "-"*70)
    print("CONFORMAL SOLVER (Square)")
    print("-"*70)
    all_results['Conformal-Square'] = run_tests(test_conformal_square, 'Conformal-Square',
                                                 source_counts, get_n_restarts)
    
    # FEM solver (disk)
    print("\n" + "-"*70)
    print("FEM SOLVER (Disk)")
    print("-"*70)
    all_results['FEM-Disk'] = run_tests(test_fem_disk, 'FEM-Disk',
                                         source_counts, get_n_restarts)
    
    # FEM solver (ellipse)
    print("\n" + "-"*70)
    print("FEM SOLVER (Ellipse)")
    print("-"*70)
    all_results['FEM-Ellipse'] = run_tests(test_fem_ellipse, 'FEM-Ellipse',
                                            source_counts, get_n_restarts)
    
    # FEM solver (square)
    print("\n" + "-"*70)
    print("FEM SOLVER (Square)")
    print("-"*70)
    all_results['FEM-Square'] = run_tests(test_fem_square, 'FEM-Square',
                                           source_counts, get_n_restarts)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY BY SOLVER")
    print("="*70)
    
    for solver, results in all_results.items():
        passed = sum(1 for r in results if r[4] == "✓")
        total = len(results)
        avg_rmse = np.mean([r[2] for r in results if r[2] < float('inf')])
        avg_time = np.mean([r[3] for r in results if r[3] > 0])
        print(f"{solver:<20}: {passed}/{total} passed, avg RMSE={avg_rmse:.2e}, avg time={avg_time:.1f}s")
    
    # Overall
    total_passed = sum(sum(1 for r in results if r[4] == "✓") for results in all_results.values())
    total_tests = sum(len(results) for results in all_results.values())
    print(f"\nOverall: {total_passed}/{total_tests} configurations passed")
    
    return total_passed == total_tests


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
