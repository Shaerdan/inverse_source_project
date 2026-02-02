#!/usr/bin/env python3
"""
Comprehensive test for inverse source localization.

Tests:
- Source counts: 4, 6, 8, 10, 12
- Depth: r in [0.7, 0.9] (shallow, near boundary)
- Placement: symmetric vs non-symmetric

Run: python test_shallow_sources.py
"""

import numpy as np
from time import time
import warnings
warnings.filterwarnings('ignore')

from analytical_solver import AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver
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


def create_symmetric_sources(n_sources, r_range=(0.7, 0.9), seed=42):
    """
    Create sources with evenly spaced angles (symmetric).
    Sources at r in [0.7, 0.9], angles evenly distributed.
    """
    np.random.seed(seed)
    sources = []
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    
    for i, theta in enumerate(angles):
        r = np.random.uniform(r_range[0], r_range[1])
        x, y = r * np.cos(theta), r * np.sin(theta)
        S = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), S))
    
    # Adjust last intensity for sum = 0
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    return sources


def create_nonsymmetric_sources(n_sources, r_range=(0.7, 0.9), seed=42):
    """
    Create sources with random angles (non-symmetric).
    Sources at r in [0.7, 0.9], angles randomly distributed but well-separated.
    """
    np.random.seed(seed)
    sources = []
    
    # Generate random angles but ensure minimum separation
    angles = []
    min_sep = np.pi / (n_sources + 1)  # Minimum angular separation
    
    for i in range(n_sources):
        for _ in range(100):  # Try up to 100 times
            theta = np.random.uniform(0, 2*np.pi)
            if all(min(abs(theta - a), 2*np.pi - abs(theta - a)) > min_sep for a in angles):
                angles.append(theta)
                break
        else:
            # Fallback: just use random angle
            angles.append(np.random.uniform(0, 2*np.pi))
    
    for i, theta in enumerate(angles):
        r = np.random.uniform(r_range[0], r_range[1])
        x, y = r * np.cos(theta), r * np.sin(theta)
        S = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), S))
    
    # Adjust last intensity for sum = 0
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    return sources


def test_configuration(n_sources, symmetric, r_range=(0.7, 0.9), seed=42, 
                       n_restarts=5, verbose=False):
    """
    Test a single configuration.
    
    Returns: (rmse, time, success)
    """
    # Create sources
    if symmetric:
        sources = create_symmetric_sources(n_sources, r_range, seed)
    else:
        sources = create_nonsymmetric_sources(n_sources, r_range, seed)
    
    if verbose:
        print(f"\n  Sources (n={n_sources}, {'symmetric' if symmetric else 'non-symmetric'}):")
        for i, ((x, y), q) in enumerate(sources):
            r = np.sqrt(x**2 + y**2)
            theta = np.degrees(np.arctan2(y, x))
            print(f"    {i+1}: r={r:.3f}, θ={theta:6.1f}°, q={q:+.2f}")
    
    # Forward solve
    forward = AnalyticalForwardSolver(100)
    u = forward.solve(sources)
    
    # Inverse solve
    inverse = AnalyticalNonlinearInverseSolver(n_sources, 100)
    inverse.set_measured_data(u)
    
    t0 = time()
    result = inverse.solve(method='SLSQP', n_restarts=n_restarts, maxiter=10000)
    elapsed = time() - t0
    
    # Compute RMSE
    rmse = compute_rmse(sources, result.sources)
    
    if verbose:
        print(f"  Recovered:")
        for i, s in enumerate(result.sources):
            r = np.sqrt(s.x**2 + s.y**2)
            theta = np.degrees(np.arctan2(s.y, s.x))
            print(f"    {i+1}: r={r:.3f}, θ={theta:6.1f}°, q={s.intensity:+.2f}")
        print(f"  RMSE: {rmse:.2e}, Time: {elapsed:.1f}s")
    
    return rmse, elapsed, rmse < 0.01


def main():
    print("="*70)
    print("INVERSE SOURCE LOCALIZATION - SHALLOW SOURCES TEST")
    print("="*70)
    print("Depth: r ∈ [0.7, 0.9]")
    print("Source counts: 4, 6, 8, 10, 12")
    print("Placement: symmetric and non-symmetric")
    print("="*70)
    
    results = []
    
    source_counts = [4, 6, 8, 10, 12]
    
    for n_sources in source_counts:
        for symmetric in [True, False]:
            placement = "symmetric" if symmetric else "non-symmetric"
            print(f"\nTesting n={n_sources}, {placement}...")
            
            rmse, elapsed, success = test_configuration(
                n_sources, 
                symmetric, 
                r_range=(0.7, 0.9),
                seed=42,
                n_restarts=5,
                verbose=True
            )
            
            status = "✓ PASS" if success else "✗ FAIL"
            results.append((n_sources, placement, rmse, elapsed, status))
    
    # Print summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'n':>3}  {'Placement':<15} {'RMSE':<12} {'Time':<8} {'Status'}")
    print("-"*55)
    
    for n, placement, rmse, t, status in results:
        print(f"{n:>3}  {placement:<15} {rmse:<12.2e} {t:<8.1f}s {status}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY BY SOURCE COUNT")
    print("="*70)
    
    for n in source_counts:
        subset = [r for r in results if r[0] == n]
        avg_rmse = np.mean([r[2] for r in subset])
        max_rmse = np.max([r[2] for r in subset])
        all_pass = all("PASS" in r[4] for r in subset)
        print(f"n={n:2d}: avg RMSE = {avg_rmse:.2e}, max RMSE = {max_rmse:.2e}, all pass = {all_pass}")
    
    # Overall
    total_pass = sum(1 for r in results if "PASS" in r[4])
    total = len(results)
    print(f"\nOverall: {total_pass}/{total} configurations passed (RMSE < 0.01)")
    
    return total_pass == total


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
