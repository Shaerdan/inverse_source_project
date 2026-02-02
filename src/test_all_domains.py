#!/usr/bin/env python3
"""
Test inverse source localization on all domains.

Tests:
- Disk (Analytical solver with SLSQP)
- Ellipse (Conformal MFS + SLSQP)
- Square (Conformal MFS + SLSQP)
- Brain (Conformal MFS + SLSQP)
- Star (Conformal MFS + Differential Evolution)

Expected results: All domains should achieve RMSE < 0.1
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
            if isinstance(sources_rec[j], tuple):
                pos_r = sources_rec[j][0]
            else:
                pos_r = (sources_rec[j].x, sources_rec[j].y)
            cost[i, j] = (pos_t[0]-pos_r[0])**2 + (pos_t[1]-pos_r[1])**2
    row_ind, col_ind = linear_sum_assignment(cost)
    return np.sqrt(cost[row_ind, col_ind].mean())


def test_disk():
    """Test disk domain with analytical solver."""
    print('\n1. DISK (Analytical + SLSQP)...')
    sources = [
        ((0.4, 0.0), 1.0), 
        ((0.0, 0.4), -1.0), 
        ((-0.4, 0.0), 1.0), 
        ((0.0, -0.4), -1.0)
    ]
    
    forward = AnalyticalForwardSolver(100)
    u = forward.solve(sources)
    
    inverse = AnalyticalNonlinearInverseSolver(4, 100)
    inverse.set_measured_data(u)
    
    t0 = time()
    result = inverse.solve(method='SLSQP', n_restarts=5)
    elapsed = time() - t0
    
    sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
    rmse = compute_rmse(sources, sources_rec)
    
    return 'Disk', rmse, elapsed


def test_ellipse():
    """Test ellipse domain with conformal MFS solver."""
    print('\n2. ELLIPSE (Conformal MFS + SLSQP)...')
    a, b = 1.5, 0.8
    sources = [
        ((0.5, 0.0), 1.0), 
        ((0.0, 0.3), -1.0), 
        ((-0.5, 0.0), 1.0), 
        ((0.0, -0.3), -1.0)
    ]
    
    emap = EllipseMap(a=a, b=b)
    forward = ConformalForwardSolver(emap, 100)
    u = forward.solve(sources)
    
    inverse = ConformalNonlinearInverseSolver(emap, 4, 100)
    
    t0 = time()
    sources_rec, _ = inverse.solve(u, method='SLSQP', n_restarts=5)
    elapsed = time() - t0
    
    rmse = compute_rmse(sources, sources_rec)
    return 'Ellipse', rmse, elapsed


def test_square():
    """Test square domain with conformal MFS solver."""
    print('\n3. SQUARE (Conformal MFS + SLSQP)...')
    sources = [
        ((0.3, 0.0), 1.0), 
        ((0.0, 0.3), -1.0), 
        ((-0.3, 0.0), 1.0), 
        ((0.0, -0.3), -1.0)
    ]
    
    smap = RectangleMap(half_width=1.0, half_height=1.0)
    forward = ConformalForwardSolver(smap, 100)
    u = forward.solve(sources)
    
    inverse = ConformalNonlinearInverseSolver(smap, 4, 100)
    
    t0 = time()
    sources_rec, _ = inverse.solve(u, method='SLSQP', n_restarts=5)
    elapsed = time() - t0
    
    rmse = compute_rmse(sources, sources_rec)
    return 'Square', rmse, elapsed


def test_brain():
    """Test brain-shaped domain with conformal MFS solver."""
    print('\n4. BRAIN (Conformal MFS + SLSQP)...')
    
    def brain_boundary(t):
        r = 1.0 + 0.15*np.cos(2*t) - 0.1*np.cos(4*t) + 0.05*np.cos(3*t)
        r = r * (1 - 0.1*np.sin(t)**4)
        return r * np.cos(t) + 1j * 0.8 * r * np.sin(t)
    
    sources = [
        ((-0.3, 0.2), 1.0), 
        ((0.3, 0.2), 1.0), 
        ((-0.2, -0.15), -1.0), 
        ((0.2, -0.15), -1.0)
    ]
    
    bmap = MFSConformalMap(brain_boundary, n_boundary=256, n_charge=200, charge_offset=0.3)
    forward = ConformalForwardSolver(bmap, 100)
    u = forward.solve(sources)
    
    inverse = ConformalNonlinearInverseSolver(bmap, 4, 100)
    
    t0 = time()
    sources_rec, _ = inverse.solve(u, method='SLSQP', n_restarts=5)
    elapsed = time() - t0
    
    rmse = compute_rmse(sources, sources_rec)
    return 'Brain', rmse, elapsed


def test_star():
    """Test star domain with conformal MFS solver and differential evolution."""
    print('\n5. STAR (Conformal MFS + Differential Evolution)...')
    
    def star_boundary(t):
        return (1.0 + 0.3 * np.cos(5 * t)) * np.exp(1j * t)
    
    sources = [
        ((0.3, 0.0), 1.0), 
        ((0.0, 0.3), -1.0), 
        ((-0.3, 0.0), 1.0), 
        ((0.0, -0.3), -1.0)
    ]
    
    stmap = MFSConformalMap(star_boundary, n_boundary=512, n_charge=400, charge_offset=0.2)
    forward = ConformalForwardSolver(stmap, 100)
    u = forward.solve(sources)
    
    inverse = ConformalNonlinearInverseSolver(stmap, 4, 100)
    
    t0 = time()
    # Star needs global optimizer due to complex boundary
    sources_rec, _ = inverse.solve(u, method='differential_evolution', seed=42)
    elapsed = time() - t0
    
    rmse = compute_rmse(sources, sources_rec)
    return 'Star', rmse, elapsed


def main():
    print('='*60)
    print('INVERSE SOURCE LOCALIZATION - ALL DOMAINS TEST')
    print('='*60)
    
    results = []
    
    results.append(test_disk())
    print(f'   RMSE: {results[-1][1]:.2e}, Time: {results[-1][2]:.1f}s')
    
    results.append(test_ellipse())
    print(f'   RMSE: {results[-1][1]:.2e}, Time: {results[-1][2]:.1f}s')
    
    results.append(test_square())
    print(f'   RMSE: {results[-1][1]:.2e}, Time: {results[-1][2]:.1f}s')
    
    results.append(test_brain())
    print(f'   RMSE: {results[-1][1]:.2e}, Time: {results[-1][2]:.1f}s')
    
    results.append(test_star())
    print(f'   RMSE: {results[-1][1]:.2e}, Time: {results[-1][2]:.1f}s')
    
    # Summary
    print('\n' + '='*60)
    print('RESULTS SUMMARY')
    print('='*60)
    print(f'{"Domain":<12} {"RMSE":<14} {"Time":<10} {"Status"}')
    print('-'*50)
    
    for name, rmse, t in results:
        status = '✓ PASS' if rmse < 0.1 else '✗ FAIL'
        print(f'{name:<12} {rmse:<14.2e} {t:<10.1f}s {status}')
    
    passed = sum(1 for _, rmse, _ in results if rmse < 0.1)
    print(f'\nTotal: {passed}/{len(results)} domains passed')
    
    return passed == len(results)


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
