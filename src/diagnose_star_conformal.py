#!/usr/bin/env python3
"""
Focused diagnostic for Star Conformal L-BFGS-B failure.

Key observations from previous diagnostic:
- Objective at truth: 0.0
- BUT gradient at truth: 0.6 (should be ~1e-10 if at true minimum)
- Residual stuck at ~3.0 even with 50 restarts

This suggests the inverse solver's forward model doesn't match the 
forward model used to generate the data.

Usage:
    cd src/
    python diagnose_star_conformal.py
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from comparison import create_domain_sources, get_conformal_map
from conformal_solver import (ConformalForwardSolver, ConformalNonlinearInverseSolver,
                               MFSConformalMap)
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def compute_position_rmse(sources_true, sources_rec):
    pos_true = np.array([[s[0][0], s[0][1]] for s in sources_true])
    pos_rec = np.array([[s[0][0], s[0][1]] for s in sources_rec])
    cost = cdist(pos_true, pos_rec)
    row_ind, col_ind = linear_sum_assignment(cost)
    return np.sqrt(np.mean(cost[row_ind, col_ind]**2))


def main():
    print("="*70)
    print("STAR CONFORMAL DIAGNOSTIC - Forward Model Consistency")
    print("="*70)
    
    sources = create_domain_sources('star')
    n_sources = len(sources)
    
    print(f"\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources):
        print(f"  {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
    
    # Get conformal map
    cmap = get_conformal_map('star')
    print(f"\nConformal map type: {type(cmap).__name__}")
    
    # TEST 1: Forward solver consistency
    print("\n" + "="*60)
    print("TEST 1: Forward Solver Configuration")
    print("="*60)
    
    # What the CLI uses for data generation
    forward_data = ConformalForwardSolver(cmap, n_boundary=100)
    u_data = forward_data.solve(sources)
    print(f"\nForward solver for data generation:")
    print(f"  n_boundary: {forward_data.n_boundary}")
    print(f"  u_data shape: {u_data.shape}")
    print(f"  u_data range: [{u_data.min():.4f}, {u_data.max():.4f}]")
    
    # What the inverse solver uses internally
    inverse = ConformalNonlinearInverseSolver(cmap, n_sources=n_sources, n_boundary=100)
    print(f"\nInverse solver configuration:")
    print(f"  n_sources: {inverse.n_sources}")
    print(f"  n_boundary: {inverse.n_boundary}")
    print(f"  Has internal forward solver: {hasattr(inverse, 'forward')}")
    
    if hasattr(inverse, 'forward'):
        print(f"  Internal forward n_boundary: {inverse.forward.n_boundary}")
    
    # TEST 2: Forward model gives same result?
    print("\n" + "="*60)
    print("TEST 2: Forward Model Consistency Check")
    print("="*60)
    
    u_inverse_forward = inverse.forward.solve(sources)
    print(f"\nComparing forward solvers:")
    print(f"  u_data (external forward):    shape={u_data.shape}")
    print(f"  u_inverse (internal forward): shape={u_inverse_forward.shape}")
    
    if u_data.shape == u_inverse_forward.shape:
        diff = np.abs(u_data - u_inverse_forward)
        print(f"  Max difference: {diff.max():.6e}")
        print(f"  Mean difference: {diff.mean():.6e}")
        print(f"  Are they close? {np.allclose(u_data, u_inverse_forward)}")
    else:
        print(f"  ⚠️ SHAPES DON'T MATCH - this is likely the bug!")
    
    # TEST 3: Boundary points comparison
    print("\n" + "="*60)
    print("TEST 3: Boundary Points Comparison")
    print("="*60)
    
    theta_data = forward_data.theta_boundary if hasattr(forward_data, 'theta_boundary') else None
    theta_inv = inverse.forward.theta_boundary if hasattr(inverse.forward, 'theta_boundary') else None
    
    print(f"\nExternal forward theta_boundary: {theta_data is not None}")
    print(f"Internal forward theta_boundary: {theta_inv is not None}")
    
    if theta_data is not None and theta_inv is not None:
        print(f"  External: {len(theta_data)} points")
        print(f"  Internal: {len(theta_inv)} points")
        if len(theta_data) == len(theta_inv):
            print(f"  Max theta diff: {np.max(np.abs(theta_data - theta_inv)):.6e}")
    
    # TEST 4: Conformal map consistency
    print("\n" + "="*60)
    print("TEST 4: Conformal Map Quality for Sources")
    print("="*60)
    
    print("\nTesting round-trip z -> w -> z' for each source:")
    for i, ((x, y), q) in enumerate(sources):
        z = complex(x, y)
        w = cmap.to_disk(z)
        z_back = cmap.from_disk(w)
        error = abs(z - z_back)
        print(f"  Source {i+1}: z={z:.4f}, w={w:.4f}, |w|={abs(w):.4f}, round-trip error={error:.2e}")
        if error > 1e-6:
            print(f"    ⚠️ LARGE ROUND-TRIP ERROR!")
    
    # TEST 5: Objective function analysis
    print("\n" + "="*60)
    print("TEST 5: Objective Function at True Solution")
    print("="*60)
    
    # Build params in the format inverse solver expects
    # Conformal solver uses: params = [x1, y1, x2, y2, ..., xn, yn, q1, q2, ..., qn]
    print("\nBuilding params vector...")
    
    params = []
    # First all positions
    for (x, y), q in sources:
        params.extend([x, y])
    # Then all intensities
    for (x, y), q in sources:
        params.append(q)
    params = np.array(params)
    print(f"  params shape: {params.shape} (expected {3*n_sources} = {3*n_sources})")
    
    # Evaluate objective (pass u_data as second argument)
    obj = inverse._objective(params, u_data)
    print(f"\n  Objective at true params: {obj:.6e}")
    
    # Check gradient numerically
    eps = 1e-7
    grad = np.zeros_like(params)
    for i in range(len(params)):
        p_plus = params.copy()
        p_plus[i] += eps
        p_minus = params.copy()
        p_minus[i] -= eps
        grad[i] = (inverse._objective(p_plus, u_data) - inverse._objective(p_minus, u_data)) / (2*eps)
    
    print(f"  ||gradient||: {np.linalg.norm(grad):.6e}")
    print(f"  Max |grad|: {np.max(np.abs(grad)):.6e}")
    
    if np.linalg.norm(grad) > 1e-3:
        print(f"\n  ⚠️ GRADIENT IS LARGE - true solution is NOT a minimum!")
        print(f"  Gradient components:")
        param_names = []
        for i in range(n_sources):
            param_names.extend([f'x{i+1}', f'y{i+1}'])
        for i in range(n_sources):
            param_names.append(f'q{i+1}')
        for name, g in zip(param_names, grad):
            if abs(g) > 1e-3:
                print(f"    {name}: {g:.6e}")
    
    # TEST 6: What if we use different n_boundary?
    print("\n" + "="*60)
    print("TEST 6: Does n_boundary mismatch cause the issue?")
    print("="*60)
    
    for n_inv in [50, 100, 200]:
        inverse_test = ConformalNonlinearInverseSolver(cmap, n_sources=n_sources, n_boundary=n_inv)
        u_test = inverse_test.forward.solve(sources)
        
        # Compare with original data
        if len(u_test) == len(u_data):
            diff = np.max(np.abs(u_test - u_data))
            print(f"  n_boundary={n_inv}: same shape, max_diff={diff:.6e}")
        else:
            print(f"  n_boundary={n_inv}: DIFFERENT SHAPE ({len(u_test)} vs {len(u_data)})")
    
    # TEST 7: Run optimization and analyze result
    print("\n" + "="*60)
    print("TEST 7: Optimization Analysis")
    print("="*60)
    
    print("\nRunning L-BFGS-B...")
    sources_rec, residual = inverse.solve(u_data, method='L-BFGS-B', n_restarts=5, seed=42)
    rmse = compute_position_rmse(sources, sources_rec)
    
    print(f"  Final residual: {residual:.6e}")
    print(f"  Position RMSE: {rmse:.6f}")
    
    print(f"\n  Recovered sources:")
    for i, ((x, y), q) in enumerate(sources_rec):
        print(f"    {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
    
    # Verify recovered sources give the claimed residual
    u_rec = inverse.forward.solve(sources_rec)
    actual_residual = np.linalg.norm(u_rec - u_data)
    print(f"\n  Actual residual ||u_rec - u_data||: {actual_residual:.6e}")
    print(f"  Claimed residual: {residual:.6e}")
    
    # TEST 8: Compare with diff_evol
    print("\n" + "="*60)
    print("TEST 8: Comparison with differential_evolution")
    print("="*60)
    
    inverse2 = ConformalNonlinearInverseSolver(cmap, n_sources=n_sources, n_boundary=100)
    sources_rec2, residual2 = inverse2.solve(u_data, method='differential_evolution', seed=42)
    rmse2 = compute_position_rmse(sources, sources_rec2)
    
    print(f"  diff_evol residual: {residual2:.6e}")
    print(f"  diff_evol RMSE: {rmse2:.6f}")
    
    if rmse2 < 0.05 and rmse > 0.1:
        print(f"\n  diff_evol works but L-BFGS-B doesn't - optimization landscape issue")
    elif rmse2 > 0.1 and rmse > 0.1:
        print(f"\n  ⚠️ BOTH fail - likely a fundamental bug, not optimization issue")


if __name__ == '__main__':
    main()
