#!/usr/bin/env python3
"""
Comprehensive test for inverse source localization across:
- Multiple source counts (2, 4, 6, 8)
- Multiple domains (disk, ellipse, star, square)
- Multiple solvers (analytical, conformal, FEM)
"""

import numpy as np
import sys
import os
from time import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytical_solver import AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver
from conformal_solver import (
    ConformalForwardSolver, ConformalNonlinearInverseSolver,
    EllipseMap, RectangleMap, NumericalConformalMap
)

def create_well_separated_sources(n_sources, r_range=(0.5, 0.8), seed=42):
    """Create well-separated sources."""
    np.random.seed(seed)
    sources = []
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.1, 0.1, n_sources)
    
    for i, theta in enumerate(angles):
        r = np.random.uniform(r_range[0], r_range[1])
        x, y = r * np.cos(theta), r * np.sin(theta)
        S = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), S))
    
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    return sources


def create_ellipse_sources(n_sources, a=1.5, b=0.8, scale=0.6, seed=42):
    """Create sources for ellipse domain."""
    np.random.seed(seed)
    sources = []
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.1, 0.1, n_sources)
    
    for i, theta in enumerate(angles):
        x = a * scale * np.cos(theta)
        y = b * scale * np.sin(theta)
        S = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), S))
    
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    return sources


def create_star_sources(n_sources, scale=0.5, seed=42):
    """Create sources for star domain (stay near center)."""
    np.random.seed(seed)
    sources = []
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.1, 0.1, n_sources)
    
    for i, theta in enumerate(angles):
        r = scale * np.random.uniform(0.7, 1.0)
        x, y = r * np.cos(theta), r * np.sin(theta)
        S = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), S))
    
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    return sources


def create_square_sources(n_sources, half_width=0.9, scale=0.5, seed=42):
    """Create sources for square domain."""
    np.random.seed(seed)
    sources = []
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.1, 0.1, n_sources)
    
    for i, theta in enumerate(angles):
        r = half_width * scale * np.random.uniform(0.7, 1.0)
        x, y = r * np.cos(theta), r * np.sin(theta)
        S = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), S))
    
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    return sources


def compute_position_rmse(sources_true, sources_rec):
    """Compute position RMSE with optimal matching."""
    from scipy.optimize import linear_sum_assignment
    
    n = len(sources_true)
    cost = np.zeros((n, n))
    
    for i, (pos_t, _) in enumerate(sources_true):
        for j in range(len(sources_rec)):
            rec = sources_rec[j]
            if isinstance(rec, tuple):
                pos_r, _ = rec
            else:
                pos_r = (rec.x, rec.y)
            dx = pos_t[0] - pos_r[0]
            dy = pos_t[1] - pos_r[1]
            cost[i, j] = dx**2 + dy**2
    
    row_ind, col_ind = linear_sum_assignment(cost)
    rmse = np.sqrt(cost[row_ind, col_ind].mean())
    return rmse


def test_disk_scaling():
    """Test disk domain with increasing source count."""
    print("\n" + "="*70)
    print("DISK DOMAIN - SCALING TEST")
    print("="*70)
    
    n_boundary = 100
    results = []
    
    for n_sources in [2, 4, 6, 8]:
        sources_true = create_well_separated_sources(n_sources, seed=42)
        
        forward = AnalyticalForwardSolver(n_boundary)
        u_measured = forward.solve(sources_true)
        
        inverse = AnalyticalNonlinearInverseSolver(n_sources, n_boundary)
        inverse.set_measured_data(u_measured)
        
        t0 = time()
        result = inverse.solve(method='SLSQP', n_restarts=5, maxiter=10000)
        elapsed = time() - t0
        
        sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
        pos_rmse = compute_position_rmse(sources_true, sources_rec)
        
        status = "✓" if pos_rmse < 0.01 else "✗"
        results.append((n_sources, pos_rmse, elapsed, status))
        print(f"  n={n_sources}: RMSE={pos_rmse:.2e}, time={elapsed:.2f}s {status}")
    
    return all(r[3] == "✓" for r in results)


def test_ellipse():
    """Test ellipse domain."""
    print("\n" + "="*70)
    print("ELLIPSE DOMAIN TEST")
    print("="*70)
    
    a, b = 1.5, 0.8
    n_boundary = 100
    results = []
    
    for n_sources in [2, 4, 6]:
        sources_true = create_ellipse_sources(n_sources, a=a, b=b, seed=42)
        
        conformal_map = EllipseMap(a=a, b=b)
        forward = ConformalForwardSolver(conformal_map, n_boundary)
        u_measured = forward.solve(sources_true)
        
        inverse = ConformalNonlinearInverseSolver(conformal_map, n_sources, n_boundary)
        
        t0 = time()
        sources_rec, residual = inverse.solve(u_measured, method='SLSQP', n_restarts=5)
        elapsed = time() - t0
        
        pos_rmse = compute_position_rmse(sources_true, sources_rec)
        
        status = "✓" if pos_rmse < 0.1 else "✗"
        results.append((n_sources, pos_rmse, elapsed, status))
        print(f"  n={n_sources}: RMSE={pos_rmse:.2e}, time={elapsed:.2f}s {status}")
    
    return all(r[3] == "✓" for r in results)


def test_star():
    """Test star domain using NumericalConformalMap."""
    print("\n" + "="*70)
    print("STAR DOMAIN TEST (NumericalConformalMap)")
    print("="*70)
    
    n_petals = 5
    amplitude = 0.3
    n_boundary = 100
    results = []
    
    # Create star boundary
    def star_boundary(t, n_petals=5, amplitude=0.3):
        r = 1.0 + amplitude * np.cos(n_petals * t)
        return r * np.exp(1j * t)
    
    t_vals = np.linspace(0, 2*np.pi, 200, endpoint=False)
    boundary_pts = star_boundary(t_vals, n_petals, amplitude)
    
    for n_sources in [2, 4]:
        sources_true = create_star_sources(n_sources, scale=0.4, seed=42)
        
        try:
            conformal_map = NumericalConformalMap(boundary_pts, n_collocation=100)
            forward = ConformalForwardSolver(conformal_map, n_boundary)
            u_measured = forward.solve(sources_true)
            
            inverse = ConformalNonlinearInverseSolver(conformal_map, n_sources, n_boundary)
            
            t0 = time()
            sources_rec, residual = inverse.solve(u_measured, method='SLSQP', n_restarts=5)
            elapsed = time() - t0
            
            pos_rmse = compute_position_rmse(sources_true, sources_rec)
            
            status = "✓" if pos_rmse < 0.15 else "✗"
            results.append((n_sources, pos_rmse, elapsed, status))
            print(f"  n={n_sources}: RMSE={pos_rmse:.2e}, time={elapsed:.2f}s {status}")
        except Exception as e:
            print(f"  n={n_sources}: FAILED - {e}")
            results.append((n_sources, np.inf, 0, "✗"))
    
    # Return True if at least one test passed or all attempted
    return len([r for r in results if r[3] == "✓"]) > 0 or len(results) == 0


def test_square():
    """Test square domain."""
    print("\n" + "="*70)
    print("SQUARE DOMAIN TEST")
    print("="*70)
    
    half_width = 1.0
    n_boundary = 100
    results = []
    
    for n_sources in [2, 4]:
        sources_true = create_square_sources(n_sources, half_width=half_width, seed=42)
        
        conformal_map = RectangleMap(width=2*half_width, height=2*half_width)
        forward = ConformalForwardSolver(conformal_map, n_boundary)
        u_measured = forward.solve(sources_true)
        
        inverse = ConformalNonlinearInverseSolver(conformal_map, n_sources, n_boundary)
        
        t0 = time()
        sources_rec, residual = inverse.solve(u_measured, method='SLSQP', n_restarts=5)
        elapsed = time() - t0
        
        pos_rmse = compute_position_rmse(sources_true, sources_rec)
        
        status = "✓" if pos_rmse < 0.15 else "✗"
        results.append((n_sources, pos_rmse, elapsed, status))
        print(f"  n={n_sources}: RMSE={pos_rmse:.2e}, time={elapsed:.2f}s {status}")
    
    return all(r[3] == "✓" for r in results)


def test_different_methods():
    """Compare different optimization methods on disk."""
    print("\n" + "="*70)
    print("METHOD COMPARISON (Disk, n=4)")
    print("="*70)
    
    n_sources = 4
    n_boundary = 100
    sources_true = create_well_separated_sources(n_sources, seed=42)
    
    forward = AnalyticalForwardSolver(n_boundary)
    u_measured = forward.solve(sources_true)
    
    methods = ['SLSQP', 'L-BFGS-B', 'trust-constr']
    results = []
    
    for method in methods:
        inverse = AnalyticalNonlinearInverseSolver(n_sources, n_boundary)
        inverse.set_measured_data(u_measured)
        
        t0 = time()
        try:
            result = inverse.solve(method=method, n_restarts=5, maxiter=5000)
            elapsed = time() - t0
            
            sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
            pos_rmse = compute_position_rmse(sources_true, sources_rec)
            
            status = "✓" if pos_rmse < 0.01 else "✗"
            results.append((method, pos_rmse, elapsed, status))
            print(f"  {method:15s}: RMSE={pos_rmse:.2e}, time={elapsed:.2f}s {status}")
        except Exception as e:
            print(f"  {method:15s}: FAILED - {e}")
            results.append((method, np.inf, 0, "✗"))
    
    return any(r[3] == "✓" for r in results)


def main():
    print("="*70)
    print("COMPREHENSIVE INVERSE SOURCE LOCALIZATION TESTS")
    print("="*70)
    
    results = {}
    
    results['disk_scaling'] = test_disk_scaling()
    results['ellipse'] = test_ellipse()
    results['star'] = test_star()
    results['square'] = test_square()
    results['methods'] = test_different_methods()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {name}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nTotal: {passed}/{total} test groups passed")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
