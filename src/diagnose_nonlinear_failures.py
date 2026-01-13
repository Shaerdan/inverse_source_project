#!/usr/bin/env python3
"""
Diagnostic script to investigate inconsistent nonlinear solver failures.

The failures from the comparison run don't follow any logical pattern:
- Conformal L-BFGS-B fails on ellipse/star but works on brain/square
- FEM diff_evol fails on brain but works elsewhere  
- FEM L-BFGS-B fails on square but works elsewhere

This script tests several hypotheses:
1. Forward model mismatch between data generation and inverse solver
2. Boundary point inconsistency
3. Objective function not zero at true solution
4. Seed sensitivity

Usage:
    cd src/
    python diagnose_nonlinear_failures.py
"""

import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

from comparison import create_domain_sources, get_conformal_map
from conformal_solver import ConformalForwardSolver, ConformalNonlinearInverseSolver
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


def test_forward_consistency(domain_type, n_trials=3):
    """
    TEST 1: Does calling ConformalForwardSolver twice give identical results?
    
    If the MFS map is regenerated each time with different results, this would
    cause data/solver mismatch.
    """
    print(f"\n{'='*60}")
    print(f"TEST 1: Forward Model Consistency - {domain_type.upper()}")
    print('='*60)
    
    sources = create_domain_sources(domain_type)
    cmap = get_conformal_map(domain_type)
    
    results = []
    for i in range(n_trials):
        forward = ConformalForwardSolver(cmap, n_boundary=100)
        u = forward.solve(sources)
        results.append(u.copy())
        print(f"  Trial {i+1}: u range = [{u.min():.6f}, {u.max():.6f}], norm = {np.linalg.norm(u):.6f}")
    
    # Check if all results are identical
    max_diff = 0
    for i in range(1, n_trials):
        diff = np.max(np.abs(results[i] - results[0]))
        max_diff = max(max_diff, diff)
    
    print(f"\n  Max difference across trials: {max_diff:.2e}")
    if max_diff > 1e-10:
        print("  ❌ FAIL: Forward solver gives DIFFERENT results on repeated calls!")
        print("  This indicates the conformal map is not deterministic.")
        return False
    else:
        print("  ✅ PASS: Forward solver is consistent")
        return True


def test_boundary_points_match(domain_type):
    """
    TEST 2: Do data generation and inverse solver use the same boundary points?
    """
    print(f"\n{'='*60}")
    print(f"TEST 2: Boundary Points Match - {domain_type.upper()}")
    print('='*60)
    
    sources = create_domain_sources(domain_type)
    cmap = get_conformal_map(domain_type)
    n_sources = len(sources)
    
    # Simulate data generation (as in compare_all_solvers_general)
    forward_data = ConformalForwardSolver(cmap, n_boundary=100)
    u_data = forward_data.solve(sources)
    boundary_data = forward_data.boundary_points.copy()
    
    # Create inverse solver (as in run_conformal_nonlinear)
    inverse = ConformalNonlinearInverseSolver(cmap, n_sources=n_sources, n_boundary=100)
    boundary_inverse = inverse.forward.boundary_points.copy()
    
    # Compare boundary points
    print(f"  Data generation boundary shape: {boundary_data.shape}")
    print(f"  Inverse solver boundary shape: {boundary_inverse.shape}")
    
    if boundary_data.shape != boundary_inverse.shape:
        print("  ❌ FAIL: Boundary shapes don't match!")
        return False
    
    diff = np.max(np.abs(boundary_data - boundary_inverse))
    print(f"  Max boundary point difference: {diff:.2e}")
    
    if diff > 1e-10:
        print("  ❌ FAIL: Boundary points DON'T match!")
        print("  Data generation and inverse solver use DIFFERENT sensor locations.")
        return False
    else:
        print("  ✅ PASS: Boundary points match")
        return True


def test_objective_at_truth(domain_type):
    """
    TEST 3: Is the objective function zero (or near-zero) at the TRUE solution?
    
    If not, there's a fundamental bug - the forward model in the inverse solver
    doesn't match the forward model used to generate data.
    """
    print(f"\n{'='*60}")
    print(f"TEST 3: Objective at True Solution - {domain_type.upper()}")
    print('='*60)
    
    sources = create_domain_sources(domain_type)
    cmap = get_conformal_map(domain_type)
    n_sources = len(sources)
    
    print(f"  True sources: {sources}")
    
    # Generate data
    forward = ConformalForwardSolver(cmap, n_boundary=100)
    u_data = forward.solve(sources)
    
    # Create inverse solver
    inverse = ConformalNonlinearInverseSolver(cmap, n_sources=n_sources, n_boundary=100)
    
    # Build params from true sources
    # Need to check the parameter format expected by _objective
    # Looking at conformal_solver.py, it seems to be [x1,y1,x2,y2,...,q1,q2,...]
    true_params = []
    for (x, y), q in sources:
        true_params.extend([x, y])
    for (x, y), q in sources:
        true_params.append(q)
    true_params = np.array(true_params)
    
    print(f"  True params: {true_params}")
    
    # Evaluate objective at truth
    obj_at_truth = inverse._objective(true_params, u_data)
    
    print(f"  Objective at TRUE solution: {obj_at_truth:.10f}")
    
    if obj_at_truth > 1e-6:
        print("  ❌ FAIL: Objective at truth is NOT near zero!")
        print("  This indicates a bug in forward model or objective function.")
        
        # Debug: check what forward solve gives at true sources
        u_forward = inverse.forward.solve(sources)
        
        print(f"\n  Debug info:")
        print(f"    u_data range: [{u_data.min():.6f}, {u_data.max():.6f}]")
        print(f"    u_forward range: [{u_forward.min():.6f}, {u_forward.max():.6f}]")
        print(f"    ||u_forward - u_data||: {np.linalg.norm(u_forward - u_data):.6f}")
        
        return False
    else:
        print("  ✅ PASS: Objective at truth is near zero")
        return True


def test_conformal_map_consistency(domain_type):
    """
    TEST 4: Is the conformal map itself consistent?
    
    Check if to_disk and from_disk are inverses, and if boundary maps to unit circle.
    """
    print(f"\n{'='*60}")
    print(f"TEST 4: Conformal Map Consistency - {domain_type.upper()}")
    print('='*60)
    
    cmap = get_conformal_map(domain_type)
    
    # Test 4a: Boundary should map to unit circle
    z_bdy = cmap.boundary_physical(50)
    w_bdy = cmap.to_disk(z_bdy)
    w_abs = np.abs(w_bdy)
    
    print(f"  Boundary |w|: min={w_abs.min():.4f}, max={w_abs.max():.4f}")
    
    if w_abs.min() < 0.95 or w_abs.max() > 1.05:
        print("  ❌ FAIL: Boundary doesn't map to unit circle!")
        return False
    else:
        print("  ✅ Boundary maps to unit circle")
    
    # Test 4b: Interior point round-trip
    sources = create_domain_sources(domain_type)
    z_src = np.array([s[0][0] + 1j * s[0][1] for s in sources])
    w_src = cmap.to_disk(z_src)
    z_back = cmap.from_disk(w_src)
    
    roundtrip_err = np.max(np.abs(z_src - z_back))
    print(f"  Interior round-trip error: {roundtrip_err:.2e}")
    
    if roundtrip_err > 0.01:
        print("  ❌ FAIL: Large round-trip error for interior points!")
        return False
    else:
        print("  ✅ Interior round-trip OK")
    
    return True


def test_actual_solve(domain_type, seed=42):
    """
    TEST 5: Actually run the solvers and compare results.
    """
    print(f"\n{'='*60}")
    print(f"TEST 5: Actual Solve - {domain_type.upper()}")
    print('='*60)
    
    sources = create_domain_sources(domain_type)
    cmap = get_conformal_map(domain_type)
    n_sources = len(sources)
    
    print(f"  True sources:")
    for i, ((x, y), q) in enumerate(sources):
        print(f"    {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
    
    # Generate data
    forward = ConformalForwardSolver(cmap, n_boundary=100)
    u_data = forward.solve(sources)
    
    print(f"\n  Data: ||u|| = {np.linalg.norm(u_data):.6f}")
    
    results = {}
    
    # Test L-BFGS-B
    print(f"\n  Running L-BFGS-B (n_restarts=5)...")
    try:
        inverse = ConformalNonlinearInverseSolver(cmap, n_sources=n_sources, n_boundary=100)
        sources_lbfgs, residual_lbfgs = inverse.solve(u_data, method='L-BFGS-B', 
                                                       n_restarts=5, seed=seed)
        rmse_lbfgs = compute_position_rmse(sources, sources_lbfgs)
        print(f"    Recovered sources:")
        for i, ((x, y), q) in enumerate(sources_lbfgs):
            print(f"      {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
        print(f"    Residual: {residual_lbfgs:.6e}")
        print(f"    Position RMSE: {rmse_lbfgs:.6f}")
        results['lbfgs'] = rmse_lbfgs
        
        if rmse_lbfgs > 0.1:
            print(f"    ❌ FAIL: RMSE too high!")
        else:
            print(f"    ✅ PASS")
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['lbfgs'] = float('inf')
    
    # Test differential_evolution
    print(f"\n  Running differential_evolution...")
    try:
        inverse = ConformalNonlinearInverseSolver(cmap, n_sources=n_sources, n_boundary=100)
        sources_de, residual_de = inverse.solve(u_data, method='differential_evolution', 
                                                 seed=seed)
        rmse_de = compute_position_rmse(sources, sources_de)
        print(f"    Recovered sources:")
        for i, ((x, y), q) in enumerate(sources_de):
            print(f"      {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
        print(f"    Residual: {residual_de:.6e}")
        print(f"    Position RMSE: {rmse_de:.6f}")
        results['de'] = rmse_de
        
        if rmse_de > 0.1:
            print(f"    ❌ FAIL: RMSE too high!")
        else:
            print(f"    ✅ PASS")
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['de'] = float('inf')
    
    return results


def test_seed_sensitivity(domain_type, n_seeds=5):
    """
    TEST 6: Are failures seed-dependent?
    """
    print(f"\n{'='*60}")
    print(f"TEST 6: Seed Sensitivity - {domain_type.upper()}")
    print('='*60)
    
    sources = create_domain_sources(domain_type)
    cmap = get_conformal_map(domain_type)
    n_sources = len(sources)
    
    # Generate data once
    forward = ConformalForwardSolver(cmap, n_boundary=100)
    u_data = forward.solve(sources)
    
    results = []
    for seed in range(42, 42 + n_seeds):
        inverse = ConformalNonlinearInverseSolver(cmap, n_sources=n_sources, n_boundary=100)
        
        try:
            sources_rec, residual = inverse.solve(u_data, method='L-BFGS-B', 
                                                   n_restarts=5, seed=seed)
            rmse = compute_position_rmse(sources, sources_rec)
        except:
            rmse = float('inf')
        
        results.append({'seed': seed, 'rmse': rmse})
    
    print(f"\n  {'Seed':<6} {'RMSE':<12} {'Status'}")
    print("  " + "-" * 30)
    for r in results:
        status = "FAIL" if r['rmse'] > 0.1 else "OK"
        print(f"  {r['seed']:<6} {r['rmse']:<12.6f} {status}")
    
    n_failures = sum(1 for r in results if r['rmse'] > 0.1)
    print(f"\n  Failures: {n_failures}/{n_seeds}")
    
    return results


def main():
    domains = ['disk', 'ellipse', 'star', 'brain', 'square']
    
    all_results = {}
    
    for domain in domains:
        print(f"\n\n{'#'*70}")
        print(f"# DOMAIN: {domain.upper()}")
        print(f"{'#'*70}")
        
        results = {}
        
        # Run tests
        results['forward_consistent'] = test_forward_consistency(domain)
        results['boundary_match'] = test_boundary_points_match(domain)
        results['objective_at_truth'] = test_objective_at_truth(domain)
        results['conformal_map_ok'] = test_conformal_map_consistency(domain)
        solve_results = test_actual_solve(domain)
        results['rmse_lbfgs'] = solve_results.get('lbfgs', float('inf'))
        results['rmse_de'] = solve_results.get('de', float('inf'))
        
        all_results[domain] = results
    
    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    
    print(f"\n{'Domain':<10} {'Fwd OK':<8} {'Bdy OK':<8} {'Obj@T':<8} {'Map OK':<8} {'L-BFGS-B':<12} {'DiffEvol':<12}")
    print("-" * 80)
    
    for domain, results in all_results.items():
        fwd = "✅" if results['forward_consistent'] else "❌"
        bdy = "✅" if results['boundary_match'] else "❌"
        obj = "✅" if results['objective_at_truth'] else "❌"
        cmap = "✅" if results['conformal_map_ok'] else "❌"
        
        lbfgs_rmse = results['rmse_lbfgs']
        de_rmse = results['rmse_de']
        
        lbfgs_str = f"{lbfgs_rmse:.6f}" if lbfgs_rmse < 100 else "FAIL"
        de_str = f"{de_rmse:.6f}" if de_rmse < 100 else "FAIL"
        
        lbfgs_status = "❌" if lbfgs_rmse > 0.1 else ""
        de_status = "❌" if de_rmse > 0.1 else ""
        
        print(f"{domain:<10} {fwd:<8} {bdy:<8} {obj:<8} {cmap:<8} {lbfgs_str:<12} {de_str:<12}")
    
    print("\n" + "="*80)
    print("Key: Fwd OK = Forward consistent, Bdy OK = Boundary match,")
    print("     Obj@T = Objective zero at truth, Map OK = Conformal map valid")
    print("="*80)


if __name__ == '__main__':
    main()