#!/usr/bin/env python3
"""
Test ellipse with FEM to see if coalescence is square-specific or FEM-specific.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from comparison import create_domain_sources, get_sensor_locations
from fem_solver import FEMForwardSolver, FEMNonlinearInverseSolver
from mesh import create_ellipse_mesh
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment, minimize


def compute_position_rmse(sources_true, sources_rec):
    pos_true = np.array([[s[0][0], s[0][1]] for s in sources_true])
    pos_rec = np.array([[s[0][0], s[0][1]] for s in sources_rec])
    cost = cdist(pos_true, pos_rec)
    row_ind, col_ind = linear_sum_assignment(cost)
    return np.sqrt(np.mean(cost[row_ind, col_ind]**2))


def truly_random_init(n_sources, x_bounds, y_bounds, seed):
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
    print("ELLIPSE FEM: Random Init Test")
    print("="*70)
    
    # Ellipse parameters
    a, b = 2.0, 1.0
    
    sources_true = create_domain_sources('ellipse')
    n_sources = len(sources_true)
    
    print(f"\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources_true):
        print(f"  {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
    
    # Setup
    sensor_locations = get_sensor_locations('ellipse', {'a': a, 'b': b}, 100)
    mesh_data = create_ellipse_mesh(a, b, 0.1, sensor_locations=sensor_locations)
    
    print(f"\nMesh: {len(mesh_data[0])} points, {len(mesh_data[4])} sensors")
    
    forward = FEMForwardSolver(resolution=0.1, verbose=False, mesh_data=mesh_data,
                                sensor_locations=sensor_locations)
    u_data = forward.solve(sources_true)
    print(f"u_data: shape={u_data.shape}, range=[{u_data.min():.4f}, {u_data.max():.4f}]")
    
    inverse = FEMNonlinearInverseSolver.from_ellipse(
        a, b, n_sources=n_sources, resolution=0.1,
        verbose=False, sensor_locations=sensor_locations, mesh_data=mesh_data
    )
    inverse.set_measured_data(u_data)
    
    print(f"\nBounds: x={inverse.x_bounds}, y={inverse.y_bounds}")
    
    # Build bounds
    bounds = []
    for i in range(n_sources):
        bounds.extend([inverse.x_bounds, inverse.y_bounds])
        if i < n_sources - 1:
            bounds.append((-5.0, 5.0))
    
    # Check distance from bounds
    print(f"\nSource distance from bounds:")
    x_lo, x_hi = inverse.x_bounds
    y_lo, y_hi = inverse.y_bounds
    for i, ((x, y), q) in enumerate(sources_true):
        dist_x = min(x - x_lo, x_hi - x)
        dist_y = min(y - y_lo, y_hi - y)
        print(f"  Source {i+1}: dist_to_x_bound={dist_x:.3f}, dist_to_y_bound={dist_y:.3f}")
    
    # Test random inits
    print("\n" + "="*60)
    print("20 RANDOM INITIALIZATIONS (L-BFGS-B)")
    print("="*60)
    
    successes = 0
    failures = []
    
    for seed in range(20):
        x0 = truly_random_init(n_sources, inverse.x_bounds, inverse.y_bounds, seed)
        
        result = minimize(inverse._objective, x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 2000})
        
        sources_rec = inverse._params_to_sources(result.x)
        rmse = compute_position_rmse(sources_true, sources_rec)
        
        success = rmse < 0.05
        if success:
            successes += 1
        else:
            failures.append((seed, result.fun, rmse, sources_rec))
        
        status = "✅" if success else "❌"
        print(f"  Seed {seed:2d}: obj={result.fun:.4e}, RMSE={rmse:.4f} {status}")
    
    print(f"\n" + "="*60)
    print(f"SUMMARY: {successes}/20 succeeded ({100*successes/20:.0f}%)")
    print("="*60)
    
    if failures:
        print(f"\nAnalyzing {len(failures)} failures:")
        for seed, obj, rmse, sources in failures[:5]:
            print(f"\n  Seed {seed} (obj={obj:.3f}, RMSE={rmse:.3f}):")
            for i, ((x, y), q) in enumerate(sources):
                print(f"    S{i+1}: ({x:+.3f}, {y:+.3f}), q={q:+.3f}")


if __name__ == '__main__':
    main()
