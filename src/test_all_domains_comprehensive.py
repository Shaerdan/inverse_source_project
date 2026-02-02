#!/usr/bin/env python3
"""
Comprehensive test for inverse source localization across multiple domains.

Tests:
- Domains: Disk, Ellipse, Square, Brain
- Source counts: 4, 6, 8, 10, 12
- Depth: r ∈ [0.7, 0.9] (shallow, near boundary)
- Placement: symmetric and non-symmetric

Run: python test_all_domains_comprehensive.py
"""

import numpy as np
from time import time
import warnings
warnings.filterwarnings('ignore')

from analytical_solver import AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver
from conformal_solver import (
    EllipseMap, RectangleMap, MFSConformalMap,
    ConformalForwardSolver, ConformalNonlinearInverseSolver
)
from scipy.optimize import linear_sum_assignment


def compute_rmse(sources_true, sources_rec):
    """Compute position RMSE with optimal matching."""
    n = len(sources_true)
    cost = np.zeros((n, n))
    for i, (pos_t, _) in enumerate(sources_true):
        for j in range(len(sources_rec)):
            if hasattr(sources_rec[j], 'x'):
                pos_r = (sources_rec[j].x, sources_rec[j].y)
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
        # Random angles with minimum separation
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
    
    # Adjust for sum = 0
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    return sources


def create_sources_ellipse(n_sources, a=1.5, b=0.8, scale=0.7, symmetric=True, seed=42):
    """
    Create sources for ellipse domain.
    Sources placed at scaled ellipse coordinates.
    scale=0.7 means sources at ~70% of the way to boundary.
    """
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
        # Random scale factor between scale and scale+0.15
        s = np.random.uniform(scale, scale + 0.15)
        x = a * s * np.cos(theta)
        y = b * s * np.sin(theta)
        S = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), S))
    
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    return sources


def create_sources_square(n_sources, half_width=1.0, scale=0.7, symmetric=True, seed=42):
    """
    Create sources for square domain [-half_width, half_width]^2.
    Sources placed at scale * half_width from center.
    """
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
        # Clamp to square interior
        x = np.clip(x, -0.9*half_width, 0.9*half_width)
        y = np.clip(y, -0.9*half_width, 0.9*half_width)
        S = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), S))
    
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    return sources


def create_sources_brain(n_sources, scale=0.5, symmetric=True, seed=42):
    """
    Create sources for brain-shaped domain.
    Brain is roughly elliptical, keep sources near center.
    """
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
        # Brain is roughly 1.0 x 0.8, so scale accordingly
        x = s * np.cos(theta)
        y = 0.8 * s * np.sin(theta)
        S = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), S))
    
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    return sources


def test_disk(n_sources, symmetric, seed=42, n_restarts=5, verbose=False):
    """Test disk domain."""
    sources = create_sources_disk(n_sources, r_range=(0.7, 0.9), 
                                   symmetric=symmetric, seed=seed)
    
    if verbose:
        print(f"  Sources:")
        for i, ((x, y), q) in enumerate(sources):
            r = np.sqrt(x**2 + y**2)
            print(f"    {i+1}: r={r:.3f}, q={q:+.2f}")
    
    forward = AnalyticalForwardSolver(100)
    u = forward.solve(sources)
    
    inverse = AnalyticalNonlinearInverseSolver(n_sources, 100)
    inverse.set_measured_data(u)
    
    t0 = time()
    result = inverse.solve(method='SLSQP', n_restarts=n_restarts, maxiter=10000)
    elapsed = time() - t0
    
    rmse = compute_rmse(sources, result.sources)
    return rmse, elapsed


def test_ellipse(n_sources, symmetric, seed=42, n_restarts=5, verbose=False):
    """Test ellipse domain."""
    a, b = 1.5, 0.8
    sources = create_sources_ellipse(n_sources, a=a, b=b, scale=0.6,
                                      symmetric=symmetric, seed=seed)
    
    if verbose:
        print(f"  Sources:")
        for i, ((x, y), q) in enumerate(sources):
            # Normalized radius for ellipse
            r_norm = np.sqrt((x/a)**2 + (y/b)**2)
            print(f"    {i+1}: r_norm={r_norm:.3f}, q={q:+.2f}")
    
    emap = EllipseMap(a=a, b=b)
    forward = ConformalForwardSolver(emap, 100)
    u = forward.solve(sources)
    
    inverse = ConformalNonlinearInverseSolver(emap, n_sources, 100)
    
    t0 = time()
    sources_rec, _ = inverse.solve(u, method='SLSQP', n_restarts=n_restarts)
    elapsed = time() - t0
    
    rmse = compute_rmse(sources, sources_rec)
    return rmse, elapsed


def test_square(n_sources, symmetric, seed=42, n_restarts=5, verbose=False):
    """Test square domain."""
    half_width = 1.0
    sources = create_sources_square(n_sources, half_width=half_width, scale=0.6,
                                     symmetric=symmetric, seed=seed)
    
    if verbose:
        print(f"  Sources:")
        for i, ((x, y), q) in enumerate(sources):
            print(f"    {i+1}: ({x:.3f}, {y:.3f}), q={q:+.2f}")
    
    smap = RectangleMap(half_width=half_width, half_height=half_width)
    forward = ConformalForwardSolver(smap, 100)
    u = forward.solve(sources)
    
    inverse = ConformalNonlinearInverseSolver(smap, n_sources, 100)
    
    t0 = time()
    sources_rec, _ = inverse.solve(u, method='SLSQP', n_restarts=n_restarts)
    elapsed = time() - t0
    
    rmse = compute_rmse(sources, sources_rec)
    return rmse, elapsed


def test_brain(n_sources, symmetric, seed=42, n_restarts=5, verbose=False):
    """Test brain-shaped domain."""
    def brain_boundary(t):
        r = 1.0 + 0.15*np.cos(2*t) - 0.1*np.cos(4*t) + 0.05*np.cos(3*t)
        r = r * (1 - 0.1*np.sin(t)**4)
        return r * np.cos(t) + 1j * 0.8 * r * np.sin(t)
    
    sources = create_sources_brain(n_sources, scale=0.5, 
                                    symmetric=symmetric, seed=seed)
    
    if verbose:
        print(f"  Sources:")
        for i, ((x, y), q) in enumerate(sources):
            print(f"    {i+1}: ({x:.3f}, {y:.3f}), q={q:+.2f}")
    
    bmap = MFSConformalMap(brain_boundary, n_boundary=256, n_charge=200, charge_offset=0.3)
    forward = ConformalForwardSolver(bmap, 100)
    u = forward.solve(sources)
    
    inverse = ConformalNonlinearInverseSolver(bmap, n_sources, 100)
    
    t0 = time()
    sources_rec, _ = inverse.solve(u, method='SLSQP', n_restarts=n_restarts)
    elapsed = time() - t0
    
    rmse = compute_rmse(sources, sources_rec)
    return rmse, elapsed


def run_domain_tests(domain_name, test_func, source_counts, n_restarts=5):
    """Run all tests for a single domain."""
    print(f"\n{'='*70}")
    print(f"DOMAIN: {domain_name.upper()}")
    print(f"{'='*70}")
    
    results = []
    
    for n_sources in source_counts:
        for symmetric in [True, False]:
            placement = "symmetric" if symmetric else "non-symmetric"
            print(f"\n  n={n_sources}, {placement}...", end=" ", flush=True)
            
            try:
                rmse, elapsed = test_func(n_sources, symmetric, seed=42, 
                                          n_restarts=n_restarts, verbose=False)
                status = "✓" if rmse < 0.01 else "✗"
                print(f"RMSE={rmse:.2e}, Time={elapsed:.1f}s {status}")
                results.append((n_sources, placement, rmse, elapsed, status))
            except Exception as e:
                print(f"ERROR: {e}")
                results.append((n_sources, placement, float('inf'), 0, "✗"))
    
    return results


def main():
    print("="*70)
    print("COMPREHENSIVE MULTI-DOMAIN INVERSE SOURCE TEST")
    print("="*70)
    print("Source counts: 4, 6, 8, 10, 12")
    print("Placement: symmetric and non-symmetric")
    print("Pass criterion: RMSE < 0.01")
    
    source_counts = [4, 6, 8, 10, 12]
    n_restarts = 5
    
    all_results = {}
    
    # Test each domain
    all_results['Disk'] = run_domain_tests('Disk', test_disk, source_counts, n_restarts)
    all_results['Ellipse'] = run_domain_tests('Ellipse', test_ellipse, source_counts, n_restarts)
    all_results['Square'] = run_domain_tests('Square', test_square, source_counts, n_restarts)
    all_results['Brain'] = run_domain_tests('Brain', test_brain, source_counts, n_restarts)
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Domain':<10} {'n':>3} {'Placement':<15} {'RMSE':<12} {'Time':<8} {'Status'}")
    print("-"*60)
    
    for domain, results in all_results.items():
        for n, placement, rmse, t, status in results:
            print(f"{domain:<10} {n:>3} {placement:<15} {rmse:<12.2e} {t:<8.1f}s {status}")
    
    # Summary by domain
    print("\n" + "="*70)
    print("SUMMARY BY DOMAIN")
    print("="*70)
    
    for domain, results in all_results.items():
        passed = sum(1 for r in results if r[4] == "✓")
        total = len(results)
        avg_rmse = np.mean([r[2] for r in results if r[2] < float('inf')])
        print(f"{domain:<10}: {passed}/{total} passed, avg RMSE = {avg_rmse:.2e}")
    
    # Overall summary
    total_passed = sum(sum(1 for r in results if r[4] == "✓") for results in all_results.values())
    total_tests = sum(len(results) for results in all_results.values())
    print(f"\nOverall: {total_passed}/{total_tests} configurations passed")
    
    return total_passed == total_tests


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
