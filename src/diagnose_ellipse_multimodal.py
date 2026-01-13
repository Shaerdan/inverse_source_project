#!/usr/bin/env python3
"""
Investigate why 4-source ellipse fails while 2-source works.

Since bounds/barriers are NOT the issue, we investigate:
1. Is the 4-source problem multimodal (multiple local minima)?
2. Are there degenerate/equivalent solutions?
3. What does the mapped geometry look like?
4. Is there something special about how sources interact in ellipse?

Usage:
    cd src/
    python diagnose_ellipse_multimodal.py
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from conformal_solver import (ConformalForwardSolver, ConformalNonlinearInverseSolver,
                               EllipseMap, DiskMap)
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


def test_mapped_source_positions():
    """
    TEST 1: Where do sources map to in the unit disk?
    
    Maybe sources that are well-separated in the ellipse become
    clustered in the disk, making recovery harder.
    """
    print("="*60)
    print("TEST 1: Source Positions in Physical vs Mapped Domain")
    print("="*60)
    
    a, b = 2.0, 1.0
    cmap = EllipseMap(a=a, b=b)
    
    # 4-source case that fails
    sources_4 = [
        ((1.0, 0.0), 1.0),
        ((-1.0, 0.0), -1.0),
        ((0.0, 0.5), 1.0),
        ((0.0, -0.5), -1.0),
    ]
    
    # 2-source case that works
    sources_2 = [
        ((0.8, 0.0), 1.0),
        ((-0.8, 0.0), -1.0),
    ]
    
    print("\n2-source case (WORKS):")
    print(f"  {'Physical (x,y)':<25} {'Mapped w':<25} {'|w|':<10} {'arg(w)'}")
    print("  " + "-"*70)
    for (x, y), q in sources_2:
        z = complex(x, y)
        w = cmap.to_disk(z)
        print(f"  ({x:6.3f}, {y:6.3f})".ljust(25) + 
              f"{w.real:6.3f} + {w.imag:6.3f}j".ljust(25) + 
              f"{abs(w):.4f}".ljust(10) + 
              f"{np.angle(w):.4f}")
    
    # Compute pairwise distances in mapped domain
    w_2 = [cmap.to_disk(complex(x, y)) for (x, y), q in sources_2]
    dist_2 = abs(w_2[0] - w_2[1])
    print(f"\n  Pairwise distance in disk: {dist_2:.4f}")
    
    print("\n4-source case (FAILS):")
    print(f"  {'Physical (x,y)':<25} {'Mapped w':<25} {'|w|':<10} {'arg(w)'}")
    print("  " + "-"*70)
    for (x, y), q in sources_4:
        z = complex(x, y)
        w = cmap.to_disk(z)
        print(f"  ({x:6.3f}, {y:6.3f})".ljust(25) + 
              f"{w.real:6.3f} + {w.imag:6.3f}j".ljust(25) + 
              f"{abs(w):.4f}".ljust(10) + 
              f"{np.angle(w):.4f}")
    
    # Compute pairwise distances in mapped domain
    w_4 = [cmap.to_disk(complex(x, y)) for (x, y), q in sources_4]
    print(f"\n  Pairwise distances in disk:")
    for i in range(4):
        for j in range(i+1, 4):
            dist = abs(w_4[i] - w_4[j])
            print(f"    Sources {i+1}-{j+1}: {dist:.4f}")


def test_multiple_restarts():
    """
    TEST 2: Run many restarts and see if we find multiple local minima
    """
    print("\n" + "="*60)
    print("TEST 2: Multiple Restarts - Finding Local Minima")
    print("="*60)
    
    a, b = 2.0, 1.0
    cmap = EllipseMap(a=a, b=b)
    
    sources = [
        ((1.0, 0.0), 1.0),
        ((-1.0, 0.0), -1.0),
        ((0.0, 0.5), 1.0),
        ((0.0, -0.5), -1.0),
    ]
    
    forward = ConformalForwardSolver(cmap, n_boundary=100)
    u_data = forward.solve(sources)
    
    print(f"\nRunning 20 independent L-BFGS-B runs with different seeds...")
    print(f"\n{'Seed':<6} {'Residual':<12} {'RMSE':<10} {'Solution summary'}")
    print("-"*70)
    
    solutions = []
    for seed in range(20):
        inverse = ConformalNonlinearInverseSolver(cmap, n_sources=4, n_boundary=100)
        try:
            sources_rec, residual = inverse.solve(u_data, method='L-BFGS-B', 
                                                   n_restarts=1, seed=seed)
            rmse = compute_position_rmse(sources, sources_rec)
            
            # Summarize solution
            pos = np.array([[s[0][0], s[0][1]] for s in sources_rec])
            summary = f"x_range=[{pos[:,0].min():.2f},{pos[:,0].max():.2f}]"
            
            solutions.append({'seed': seed, 'residual': residual, 'rmse': rmse, 
                            'sources': sources_rec})
            
            status = "✅" if rmse < 0.1 else ""
            print(f"{seed:<6} {residual:<12.4e} {rmse:<10.4f} {summary} {status}")
            
        except Exception as e:
            print(f"{seed:<6} ERROR: {e}")
    
    # Analyze solutions
    residuals = [s['residual'] for s in solutions]
    rmses = [s['rmse'] for s in solutions]
    
    print(f"\nSummary:")
    print(f"  Best residual: {min(residuals):.4e} (seed {residuals.index(min(residuals))})")
    print(f"  Best RMSE: {min(rmses):.4f} (seed {rmses.index(min(rmses))})")
    print(f"  Solutions with RMSE < 0.1: {sum(1 for r in rmses if r < 0.1)}/20")
    
    # Show the best solution
    best_idx = rmses.index(min(rmses))
    best = solutions[best_idx]
    print(f"\n  Best solution (seed {best['seed']}):")
    for i, ((x, y), q) in enumerate(best['sources']):
        print(f"    {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")


def test_objective_at_permuted_solutions():
    """
    TEST 3: Are there multiple equivalent solutions (permutations, sign flips)?
    """
    print("\n" + "="*60)
    print("TEST 3: Equivalent Solutions (Permutations/Symmetries)")
    print("="*60)
    
    a, b = 2.0, 1.0
    cmap = EllipseMap(a=a, b=b)
    
    sources = [
        ((1.0, 0.0), 1.0),
        ((-1.0, 0.0), -1.0),
        ((0.0, 0.5), 1.0),
        ((0.0, -0.5), -1.0),
    ]
    
    forward = ConformalForwardSolver(cmap, n_boundary=100)
    u_data = forward.solve(sources)
    
    inverse = ConformalNonlinearInverseSolver(cmap, n_sources=4, n_boundary=100)
    
    def to_params(src_list):
        params = []
        for (x, y), q in src_list:
            params.extend([x, y])
        for (x, y), q in src_list:
            params.append(q)
        return np.array(params)
    
    # Original solution
    params_orig = to_params(sources)
    obj_orig = inverse._objective(params_orig, u_data)
    print(f"\nOriginal solution: obj = {obj_orig:.6e}")
    
    # Test some permutations
    from itertools import permutations
    
    print(f"\nTesting all 24 permutations of the 4 sources:")
    perm_objectives = []
    for i, perm in enumerate(permutations(range(4))):
        permuted = [sources[p] for p in perm]
        params = to_params(permuted)
        obj = inverse._objective(params, u_data)
        perm_objectives.append(obj)
        if i < 10:  # Show first 10
            print(f"  Perm {perm}: obj = {obj:.6e}")
    
    print(f"  ...")
    print(f"\n  All permutations give objective ~0: {all(o < 1e-6 for o in perm_objectives)}")
    
    # Test sign-flipped solutions (all intensities negated)
    flipped = [((x, y), -q) for (x, y), q in sources]
    params_flipped = to_params(flipped)
    obj_flipped = inverse._objective(params_flipped, u_data)
    print(f"\nSign-flipped solution: obj = {obj_flipped:.6e}")
    
    # Test spatially reflected solutions
    reflected_x = [((-x, y), q) for (x, y), q in sources]
    params_rx = to_params(reflected_x)
    obj_rx = inverse._objective(params_rx, u_data)
    print(f"X-reflected solution: obj = {obj_rx:.6e}")
    
    reflected_y = [((x, -y), q) for (x, y), q in sources]
    params_ry = to_params(reflected_y)
    obj_ry = inverse._objective(params_ry, u_data)
    print(f"Y-reflected solution: obj = {obj_ry:.6e}")


def test_gradient_at_truth():
    """
    TEST 4: Check gradient near the true solution
    """
    print("\n" + "="*60)
    print("TEST 4: Numerical Gradient Near Truth")
    print("="*60)
    
    a, b = 2.0, 1.0
    cmap = EllipseMap(a=a, b=b)
    
    sources = [
        ((1.0, 0.0), 1.0),
        ((-1.0, 0.0), -1.0),
        ((0.0, 0.5), 1.0),
        ((0.0, -0.5), -1.0),
    ]
    
    forward = ConformalForwardSolver(cmap, n_boundary=100)
    u_data = forward.solve(sources)
    
    inverse = ConformalNonlinearInverseSolver(cmap, n_sources=4, n_boundary=100)
    
    def to_params(src_list):
        params = []
        for (x, y), q in src_list:
            params.extend([x, y])
        for (x, y), q in src_list:
            params.append(q)
        return np.array(params)
    
    params = to_params(sources)
    
    # Compute numerical gradient
    eps = 1e-6
    grad = np.zeros_like(params)
    f0 = inverse._objective(params, u_data)
    
    print(f"\nObjective at truth: {f0:.10e}")
    print(f"\nNumerical gradient (should be ~0 at minimum):")
    print(f"  {'Param':<10} {'Value':<12} {'∂f/∂param':<15}")
    print("  " + "-"*40)
    
    param_names = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'q1', 'q2', 'q3', 'q4']
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = inverse._objective(params_plus, u_data)
        
        params_minus = params.copy()
        params_minus[i] -= eps
        f_minus = inverse._objective(params_minus, u_data)
        
        grad[i] = (f_plus - f_minus) / (2 * eps)
        print(f"  {param_names[i]:<10} {params[i]:<12.4f} {grad[i]:<15.6e}")
    
    print(f"\n  ||gradient||: {np.linalg.norm(grad):.6e}")
    
    # Check Hessian eigenvalues (is it positive definite?)
    print(f"\nNumerical Hessian eigenvalues (checking if minimum is well-conditioned):")
    
    n = len(params)
    H = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            params_pp = params.copy()
            params_pp[i] += eps
            params_pp[j] += eps
            
            params_pm = params.copy()
            params_pm[i] += eps
            params_pm[j] -= eps
            
            params_mp = params.copy()
            params_mp[i] -= eps
            params_mp[j] += eps
            
            params_mm = params.copy()
            params_mm[i] -= eps
            params_mm[j] -= eps
            
            H[i,j] = (inverse._objective(params_pp, u_data) 
                     - inverse._objective(params_pm, u_data)
                     - inverse._objective(params_mp, u_data) 
                     + inverse._objective(params_mm, u_data)) / (4 * eps * eps)
    
    eigenvalues = np.linalg.eigvalsh(H)
    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  Min eigenvalue: {eigenvalues.min():.6e}")
    print(f"  Max eigenvalue: {eigenvalues.max():.6e}")
    print(f"  Condition number: {eigenvalues.max() / max(eigenvalues.min(), 1e-10):.2e}")
    
    if eigenvalues.min() < 0:
        print(f"  ⚠️ NEGATIVE EIGENVALUE - not a proper minimum!")
    elif eigenvalues.min() < 1e-6:
        print(f"  ⚠️ NEAR-ZERO EIGENVALUE - degenerate/flat direction!")


def test_compare_disk_4source():
    """
    TEST 5: Does 4-source work on disk? (isolate ellipse mapping issues)
    """
    print("\n" + "="*60)
    print("TEST 5: 4-Source on Disk (baseline comparison)")
    print("="*60)
    
    # Equivalent 4-source problem on unit disk
    disk_sources = [
        ((0.5, 0.0), 1.0),
        ((-0.5, 0.0), -1.0),
        ((0.0, 0.5), 1.0),
        ((0.0, -0.5), -1.0),
    ]
    
    cmap = DiskMap(radius=1.0)
    forward = ConformalForwardSolver(cmap, n_boundary=100)
    u_data = forward.solve(disk_sources)
    
    inverse = ConformalNonlinearInverseSolver(cmap, n_sources=4, n_boundary=100)
    
    print(f"\nDisk 4-source test:")
    print(f"  Sources: {disk_sources}")
    
    sources_rec, residual = inverse.solve(u_data, method='L-BFGS-B', n_restarts=5, seed=42)
    rmse = compute_position_rmse(disk_sources, sources_rec)
    
    print(f"\n  L-BFGS-B result:")
    print(f"    Residual: {residual:.6e}")
    print(f"    RMSE: {rmse:.6f}")
    
    if rmse < 0.01:
        print(f"    ✅ Disk 4-source WORKS - problem is specific to ellipse")
    else:
        print(f"    ❌ Disk 4-source also fails - problem is general")


def main():
    test_mapped_source_positions()
    test_multiple_restarts()
    test_objective_at_permuted_solutions()
    test_gradient_at_truth()
    test_compare_disk_4source()
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
