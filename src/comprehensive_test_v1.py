#!/usr/bin/env python3
"""
Diagnostic script to investigate inconsistent nonlinear solver failures.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from comparison import get_sensor_locations, get_conformal_map, create_sources

def test_forward_model_consistency(domain, n_sources=4, seed=42):
    """
    Test 1: Does the forward model used for data generation match 
    the forward model used in the inverse solver?
    """
    print(f"\n{'='*60}")
    print(f"TEST 1: Forward Model Consistency - {domain.upper()}")
    print('='*60)
    
    # Get sensor locations (same as comparison.py uses)
    sensors = get_sensor_locations(domain, n_sensors=100, seed=seed)
    
    # Create true sources
    sources = create_sources(domain, n_sources=n_sources, seed=seed)
    print(f"True sources: {sources}")
    
    # Generate data with FEM forward model
    if domain == 'disk':
        from analytical_solver import AnalyticalForwardSolver
        fem_forward = AnalyticalForwardSolver(n_boundary=100)
        # Need to check if it uses sensor_locations
    else:
        from fem_solver import FEMForwardSolver
        fem_forward = FEMForwardSolver(domain=domain)
    
    # Generate data with Conformal forward model
    from conformal_solver import ConformalForwardSolver
    cmap = get_conformal_map(domain)
    conf_forward = ConformalForwardSolver(cmap, n_boundary=100, sensor_locations=sensors)
    
    u_conformal = conf_forward.solve(sources)
    
    print(f"Conformal forward data: min={u_conformal.min():.6f}, max={u_conformal.max():.6f}")
    print(f"Conformal forward uses {conf_forward.n_sensors} sensors")
    
    # Now check: if we use conformal forward to generate data,
    # can conformal inverse recover it?
    from conformal_solver import ConformalNonlinearInverseSolver
    
    inverse = ConformalNonlinearInverseSolver(cmap, n_sources=n_sources, 
                                               n_boundary=100, 
                                               sensor_locations=sensors)
    
    # Test with L-BFGS-B
    recovered_lbfgs, residual_lbfgs = inverse.solve(u_conformal, method='L-BFGS-B', 
                                                     n_restarts=5, seed=seed)
    print(f"\nL-BFGS-B recovery:")
    print(f"  Recovered: {recovered_lbfgs}")
    print(f"  Residual: {residual_lbfgs:.6f}")
    
    # Test with diff_evol
    recovered_de, residual_de = inverse.solve(u_conformal, method='differential_evolution', 
                                               seed=seed)
    print(f"\nDiff Evol recovery:")
    print(f"  Recovered: {recovered_de}")
    print(f"  Residual: {residual_de:.6f}")
    
    # Compute position errors
    true_pos = np.array([[s[0][0], s[0][1]] for s in sources])
    
    rec_pos_lbfgs = np.array([[s[0][0], s[0][1]] for s in recovered_lbfgs])
    rec_pos_de = np.array([[s[0][0], s[0][1]] for s in recovered_de])
    
    # Simple RMSE (not handling permutation for now)
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    
    def matched_rmse(true_pos, rec_pos):
        cost = cdist(true_pos, rec_pos)
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_dist = cost[row_ind, col_ind]
        return np.sqrt(np.mean(matched_dist**2))
    
    rmse_lbfgs = matched_rmse(true_pos, rec_pos_lbfgs)
    rmse_de = matched_rmse(true_pos, rec_pos_de)
    
    print(f"\nPosition RMSE:")
    print(f"  L-BFGS-B: {rmse_lbfgs:.6f}")
    print(f"  Diff Evol: {rmse_de:.6f}")
    
    return rmse_lbfgs, rmse_de


def test_data_generation_match(domain, n_sources=4, seed=42):
    """
    Test 2: Is the data being generated the same way for both solvers?
    """
    print(f"\n{'='*60}")
    print(f"TEST 2: Data Generation Match - {domain.upper()}")
    print('='*60)
    
    # This tests if comparison.py generates the same data for conformal and FEM
    # Check the actual comparison.py code to see how it generates data
    
    pass  # Implement based on comparison.py structure


def test_seed_sensitivity(domain, n_seeds=5):
    """
    Test 3: Are failures seed-dependent?
    """
    print(f"\n{'='*60}")
    print(f"TEST 3: Seed Sensitivity - {domain.upper()}")
    print('='*60)
    
    results = []
    for seed in range(42, 42 + n_seeds):
        rmse_lbfgs, rmse_de = test_forward_model_consistency(domain, seed=seed)
        results.append({
            'seed': seed,
            'lbfgs': rmse_lbfgs,
            'diff_evol': rmse_de
        })
    
    print(f"\nSummary across seeds:")
    print(f"{'Seed':<6} {'L-BFGS-B':<12} {'Diff Evol':<12}")
    print("-" * 30)
    for r in results:
        lbfgs_status = "FAIL" if r['lbfgs'] > 0.1 else "OK"
        de_status = "FAIL" if r['diff_evol'] > 0.1 else "OK"
        print(f"{r['seed']:<6} {r['lbfgs']:.6f} ({lbfgs_status})  {r['diff_evol']:.6f} ({de_status})")


def test_objective_at_truth(domain, n_sources=4, seed=42):
    """
    Test 4: What is the objective function value at the TRUE source locations?
    If it's not near zero, there's a forward model mismatch.
    """
    print(f"\n{'='*60}")
    print(f"TEST 4: Objective at True Solution - {domain.upper()}")
    print('='*60)
    
    from comparison import get_sensor_locations, create_sources
    from conformal_solver import ConformalForwardSolver, ConformalNonlinearInverseSolver, get_conformal_map
    
    sensors = get_sensor_locations(domain, n_sensors=100, seed=seed)
    sources = create_sources(domain, n_sources=n_sources, seed=seed)
    cmap = get_conformal_map(domain)
    
    # Generate data
    forward = ConformalForwardSolver(cmap, n_boundary=100, sensor_locations=sensors)
    u_data = forward.solve(sources)
    
    # Now evaluate objective at TRUE parameters
    inverse = ConformalNonlinearInverseSolver(cmap, n_sources=n_sources,
                                               n_boundary=100, sensor_locations=sensors)
    
    # Build params from true sources
    true_params = []
    for (x, y), q in sources:
        true_params.extend([x, y])
    for (x, y), q in sources:
        true_params.append(q)
    true_params = np.array(true_params)
    
    obj_at_truth = inverse._objective(true_params, u_data)
    print(f"Objective at TRUE solution: {obj_at_truth:.10f}")
    
    if obj_at_truth > 1e-6:
        print("WARNING: Objective at truth is not near zero!")
        print("This indicates a bug in the forward model or objective function.")
    else:
        print("OK: Objective at truth is near zero.")


if __name__ == '__main__':
    domains = ['ellipse', 'star', 'brain', 'square']
    
    for domain in domains:
        print(f"\n\n{'#'*70}")
        print(f"# DOMAIN: {domain.upper()}")
        print(f"{'#'*70}")
        
        test_objective_at_truth(domain)
        test_forward_model_consistency(domain)