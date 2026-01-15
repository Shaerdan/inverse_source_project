#!/usr/bin/env python3
"""
Investigate why L-BFGS-B fails so often for simple 4-source problems.

For 4 well-separated sources, L-BFGS-B should converge reliably.
30-55% success rate is unacceptable.

Questions:
1. What do the failed initializations look like vs successful ones?
2. Is there a pattern in where L-BFGS-B gets stuck?
3. Is the intensity parameterization causing issues?
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from comparison import create_domain_sources, get_sensor_locations
from fem_solver import FEMForwardSolver, FEMNonlinearInverseSolver
from mesh import create_disk_mesh
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment, minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_position_rmse(sources_true, sources_rec):
    pos_true = np.array([[s[0][0], s[0][1]] for s in sources_true])
    pos_rec = np.array([[s[0][0], s[0][1]] for s in sources_rec])
    cost = cdist(pos_true, pos_rec)
    row_ind, col_ind = linear_sum_assignment(cost)
    return np.sqrt(np.mean(cost[row_ind, col_ind]**2))


def main():
    print("="*70)
    print("INVESTIGATING L-BFGS-B FAILURES")
    print("="*70)
    
    # Setup disk problem
    sources_true = create_domain_sources('disk')
    n_sources = len(sources_true)
    
    print(f"\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources_true):
        print(f"  {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
    
    sensor_locations = get_sensor_locations('disk', None, 100)
    mesh_data = create_disk_mesh(0.1, sensor_locations=sensor_locations)
    
    forward = FEMForwardSolver(resolution=0.1, verbose=False, mesh_data=mesh_data,
                                sensor_locations=sensor_locations)
    u_data = forward.solve(sources_true)
    
    inverse = FEMNonlinearInverseSolver(n_sources=n_sources, resolution=0.1,
                                         verbose=False, mesh_data=mesh_data)
    inverse.set_measured_data(u_data)
    
    # Build bounds
    bounds = []
    for i in range(n_sources):
        bounds.extend([inverse.x_bounds, inverse.y_bounds])
        if i < n_sources - 1:
            bounds.append((-5.0, 5.0))
    
    # Collect successful and failed runs
    print("\n" + "="*60)
    print("ANALYZING 20 RUNS")
    print("="*60)
    
    successes = []
    failures = []
    
    for seed in range(20):
        np.random.seed(seed)
        x0 = inverse._get_initial_guess('random', 0)  # seed=0 but random state set externally
        x0 = np.array(x0)
        
        # Compute initial sources
        init_sources = inverse._params_to_sources(x0)
        init_obj = inverse._objective(x0)
        
        # Run optimization
        result = minimize(inverse._objective, x0, method='L-BFGS-B',
                         bounds=bounds, options={'maxiter': 2000})
        
        final_sources = inverse._params_to_sources(result.x)
        rmse = compute_position_rmse(sources_true, final_sources)
        
        run_data = {
            'seed': seed,
            'init_sources': init_sources,
            'init_obj': init_obj,
            'final_sources': final_sources,
            'final_obj': result.fun,
            'rmse': rmse,
            'success': rmse < 0.05
        }
        
        if rmse < 0.05:
            successes.append(run_data)
        else:
            failures.append(run_data)
        
        status = "✅" if rmse < 0.05 else "❌"
        print(f"  Seed {seed:2d}: init_obj={init_obj:.2f}, final_obj={result.fun:.4e}, RMSE={rmse:.4f} {status}")
    
    print(f"\n{len(successes)} successes, {len(failures)} failures")
    
    # Analyze failures
    print("\n" + "="*60)
    print("ANALYZING FAILURES")
    print("="*60)
    
    if failures:
        print("\n--- Initial positions of FAILED runs ---")
        for run in failures[:5]:
            print(f"\nSeed {run['seed']} (init_obj={run['init_obj']:.2f}, final RMSE={run['rmse']:.3f}):")
            print("  Init sources:")
            for i, ((x, y), q) in enumerate(run['init_sources']):
                r = np.sqrt(x**2 + y**2)
                print(f"    {i+1}: ({x:+.3f}, {y:+.3f}), r={r:.3f}, q={q:+.3f}")
            print("  Final sources:")
            for i, ((x, y), q) in enumerate(run['final_sources']):
                r = np.sqrt(x**2 + y**2)
                print(f"    {i+1}: ({x:+.3f}, {y:+.3f}), r={r:.3f}, q={q:+.3f}")
    
    # Analyze successes
    print("\n" + "="*60)
    print("ANALYZING SUCCESSES")
    print("="*60)
    
    if successes:
        print("\n--- Initial positions of SUCCESSFUL runs ---")
        for run in successes[:5]:
            print(f"\nSeed {run['seed']} (init_obj={run['init_obj']:.2f}, final RMSE={run['rmse']:.6f}):")
            print("  Init sources:")
            for i, ((x, y), q) in enumerate(run['init_sources']):
                r = np.sqrt(x**2 + y**2)
                print(f"    {i+1}: ({x:+.3f}, {y:+.3f}), r={r:.3f}, q={q:+.3f}")
    
    # Check intensity distribution
    print("\n" + "="*60)
    print("INTENSITY ANALYSIS")
    print("="*60)
    
    print("\n--- Failed runs: final intensities ---")
    for run in failures[:5]:
        intensities = [s[1] for s in run['final_sources']]
        print(f"  Seed {run['seed']}: q = [{', '.join(f'{q:+.2f}' for q in intensities)}]")
        print(f"            sum = {sum(intensities):.4f}, max|q| = {max(abs(q) for q in intensities):.2f}")
    
    print("\n--- Successful runs: final intensities ---")
    for run in successes[:5]:
        intensities = [s[1] for s in run['final_sources']]
        print(f"  Seed {run['seed']}: q = [{', '.join(f'{q:+.2f}' for q in intensities)}]")
    
    # Key insight: check if failures have sources clustered
    print("\n" + "="*60)
    print("SOURCE CLUSTERING IN FAILURES")
    print("="*60)
    
    print("\n--- Checking pairwise distances in failed runs ---")
    for run in failures[:5]:
        positions = np.array([s[0] for s in run['final_sources']])
        n = len(positions)
        
        print(f"\nSeed {run['seed']}:")
        min_dist = np.inf
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                min_dist = min(min_dist, d)
                if d < 0.5:
                    print(f"  Sources {i+1}-{j+1}: distance = {d:.3f} (CLOSE!)")
        print(f"  Min pairwise distance: {min_dist:.3f}")
    
    # Test: What if we constrain intensities more tightly?
    print("\n" + "="*60)
    print("TEST: TIGHTER INTENSITY BOUNDS")
    print("="*60)
    
    tight_bounds = []
    for i in range(n_sources):
        tight_bounds.extend([inverse.x_bounds, inverse.y_bounds])
        if i < n_sources - 1:
            tight_bounds.append((-2.0, 2.0))  # Was (-5, 5)
    
    tight_successes = 0
    print("\n--- L-BFGS-B with q ∈ [-2, 2] instead of [-5, 5] ---")
    for seed in range(20):
        np.random.seed(seed)
        x0 = inverse._get_initial_guess('random', 0)
        
        result = minimize(inverse._objective, np.array(x0), method='L-BFGS-B',
                         bounds=tight_bounds, options={'maxiter': 2000})
        
        final_sources = inverse._params_to_sources(result.x)
        rmse = compute_position_rmse(sources_true, final_sources)
        
        if rmse < 0.05:
            tight_successes += 1
        
        status = "✅" if rmse < 0.05 else "❌"
        print(f"  Seed {seed:2d}: obj={result.fun:.4e}, RMSE={rmse:.4f} {status}")
    
    print(f"\n  Tight bounds success: {tight_successes}/20 ({100*tight_successes/20:.0f}%)")
    print(f"  Original bounds success: {len(successes)}/20 ({100*len(successes)/20:.0f}%)")


if __name__ == '__main__':
    main()
