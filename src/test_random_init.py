#!/usr/bin/env python3
"""
Test TRULY RANDOM initialization for L-BFGS-B.

User's valid point: For 4 well-separated sources at corners of a square,
random initialization anywhere in the domain should converge to the truth
in most cases. If it doesn't, something is fundamentally wrong.

Let's test this hypothesis.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from comparison import create_domain_sources, get_sensor_locations
from fem_solver import FEMForwardSolver, FEMNonlinearInverseSolver
from mesh import create_polygon_mesh
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment, minimize


def compute_position_rmse(sources_true, sources_rec):
    pos_true = np.array([[s[0][0], s[0][1]] for s in sources_true])
    pos_rec = np.array([[s[0][0], s[0][1]] for s in sources_rec])
    cost = cdist(pos_true, pos_rec)
    row_ind, col_ind = linear_sum_assignment(cost)
    return np.sqrt(np.mean(cost[row_ind, col_ind]**2))


def truly_random_init(n_sources, x_bounds, y_bounds, seed):
    """Truly random initialization - uniform in domain."""
    np.random.seed(seed)
    
    x_lo, x_hi = x_bounds
    y_lo, y_hi = y_bounds
    
    x0 = []
    for i in range(n_sources):
        x = np.random.uniform(x_lo, x_hi)
        y = np.random.uniform(y_lo, y_hi)
        x0.extend([x, y])
        if i < n_sources - 1:
            q = np.random.uniform(-2, 2)
            x0.append(q)
    
    return np.array(x0)


def main():
    print("="*70)
    print("TRULY RANDOM INITIALIZATION TEST")
    print("="*70)
    
    vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    sources_true = create_domain_sources('square')
    n_sources = len(sources_true)
    
    print(f"\nTrue sources (at corners):")
    for i, ((x, y), q) in enumerate(sources_true):
        print(f"  {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
    
    # Setup
    sensor_locations = get_sensor_locations('square', None, 100)
    mesh_data = create_polygon_mesh(vertices, 0.1, sensor_locations=sensor_locations)
    
    forward = FEMForwardSolver(resolution=0.1, verbose=False, mesh_data=mesh_data,
                                sensor_locations=sensor_locations)
    u_data = forward.solve(sources_true)
    
    inverse = FEMNonlinearInverseSolver.from_polygon(
        vertices, n_sources=n_sources, resolution=0.1,
        verbose=False, sensor_locations=sensor_locations, mesh_data=mesh_data
    )
    inverse.set_measured_data(u_data)
    
    # Build bounds
    bounds = []
    for i in range(n_sources):
        bounds.extend([inverse.x_bounds, inverse.y_bounds])
        if i < n_sources - 1:
            bounds.append((-5.0, 5.0))
    
    print(f"\nBounds: x in {inverse.x_bounds}, y in {inverse.y_bounds}")
    
    # Test 50 truly random initializations
    print("\n" + "="*60)
    print("50 TRULY RANDOM INITIALIZATIONS (L-BFGS-B)")
    print("="*60)
    
    successes = 0
    failures = 0
    results = []
    
    for seed in range(50):
        x0 = truly_random_init(n_sources, inverse.x_bounds, inverse.y_bounds, seed)
        init_obj = inverse._objective(x0)
        
        result = minimize(
            inverse._objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000}
        )
        
        sources_rec = inverse._params_to_sources(result.x)
        rmse = compute_position_rmse(sources_true, sources_rec)
        
        success = rmse < 0.05
        if success:
            successes += 1
        else:
            failures += 1
        
        results.append((seed, init_obj, result.fun, rmse, success))
        
        status = "✅" if success else "❌"
        print(f"  Seed {seed:2d}: init_obj={init_obj:8.2f}, final_obj={result.fun:10.4e}, RMSE={rmse:.4f} {status}")
    
    print(f"\n" + "="*60)
    print(f"SUMMARY: {successes}/50 succeeded ({100*successes/50:.0f}%), {failures}/50 failed")
    print("="*60)
    
    if failures > 0:
        print(f"\n⚠️ {failures} out of 50 random inits FAILED to find global minimum!")
        print("This is concerning for such a simple problem.")
        
        # Analyze the failures
        print("\n--- Analyzing failures ---")
        failed_results = [(s, io, fo, r) for s, io, fo, r, success in results if not success]
        
        # Check the final objectives
        final_objs = [fo for _, _, fo, _ in failed_results]
        print(f"  Failed final objectives: min={min(final_objs):.4e}, max={max(final_objs):.4e}")
        
        # Are they converging to similar local minima?
        print(f"\n  Failed runs converged to these residuals:")
        for seed, init_obj, final_obj, rmse in failed_results[:10]:
            print(f"    Seed {seed}: residual={np.sqrt(final_obj):.4f}, RMSE={rmse:.4f}")
    
    # Also test: what if we DON'T use bounds?
    print("\n" + "="*60)
    print("TEST: L-BFGS-B WITHOUT BOUNDS")
    print("="*60)
    
    successes_unbounded = 0
    for seed in range(20):
        x0 = truly_random_init(n_sources, inverse.x_bounds, inverse.y_bounds, seed)
        
        result = minimize(
            inverse._objective,
            x0,
            method='L-BFGS-B',
            # NO BOUNDS!
            options={'maxiter': 2000}
        )
        
        sources_rec = inverse._params_to_sources(result.x)
        rmse = compute_position_rmse(sources_true, sources_rec)
        
        success = rmse < 0.05
        if success:
            successes_unbounded += 1
        
        status = "✅" if success else "❌"
        print(f"  Seed {seed:2d}: final_obj={result.fun:10.4e}, RMSE={rmse:.4f} {status}")
    
    print(f"\n  Unbounded: {successes_unbounded}/20 succeeded")
    
    # Test with different optimizer
    print("\n" + "="*60)
    print("TEST: BFGS (not L-BFGS-B) WITHOUT BOUNDS")
    print("="*60)
    
    successes_bfgs = 0
    for seed in range(20):
        x0 = truly_random_init(n_sources, inverse.x_bounds, inverse.y_bounds, seed)
        
        result = minimize(
            inverse._objective,
            x0,
            method='BFGS',
            options={'maxiter': 2000}
        )
        
        sources_rec = inverse._params_to_sources(result.x)
        rmse = compute_position_rmse(sources_true, sources_rec)
        
        success = rmse < 0.05
        if success:
            successes_bfgs += 1
        
        status = "✅" if success else "❌"
        print(f"  Seed {seed:2d}: final_obj={result.fun:10.4e}, RMSE={rmse:.4f} {status}")
    
    print(f"\n  BFGS: {successes_bfgs}/20 succeeded")


if __name__ == '__main__':
    main()
