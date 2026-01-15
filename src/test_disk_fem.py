#!/usr/bin/env python3
"""
Test DISK with FEM to establish baseline.

If disk FEM works well, but square FEM doesn't, that points to 
something specific about the square domain setup.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from comparison import create_domain_sources, get_sensor_locations
from fem_solver import FEMForwardSolver, FEMNonlinearInverseSolver
from mesh import create_disk_mesh
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
    print("DISK FEM: Random Init Test (Baseline)")
    print("="*70)
    
    sources_true = create_domain_sources('disk')
    n_sources = len(sources_true)
    
    print(f"\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources_true):
        r = np.sqrt(x**2 + y**2)
        print(f"  {i+1}: ({x:.4f}, {y:.4f}), r={r:.3f}, q={q:.4f}")
    
    # Setup
    sensor_locations = get_sensor_locations('disk', None, 100)
    mesh_data = create_disk_mesh(0.1, sensor_locations=sensor_locations)
    
    print(f"\nMesh: {len(mesh_data[0])} points, {len(mesh_data[4])} sensors")
    
    forward = FEMForwardSolver(resolution=0.1, verbose=False, mesh_data=mesh_data,
                                sensor_locations=sensor_locations)
    u_data = forward.solve(sources_true)
    print(f"u_data: shape={u_data.shape}, range=[{u_data.min():.4f}, {u_data.max():.4f}]")
    
    inverse = FEMNonlinearInverseSolver(n_sources=n_sources, resolution=0.1,
                                         verbose=False, mesh_data=mesh_data)
    inverse.set_measured_data(u_data)
    
    print(f"\nBounds: x={inverse.x_bounds}, y={inverse.y_bounds}")
    
    # Build bounds
    bounds = []
    for i in range(n_sources):
        bounds.extend([inverse.x_bounds, inverse.y_bounds])
        if i < n_sources - 1:
            bounds.append((-5.0, 5.0))
    
    # Test random inits
    print("\n" + "="*60)
    print("20 RANDOM INITIALIZATIONS (L-BFGS-B)")
    print("="*60)
    
    successes = 0
    
    for seed in range(20):
        x0 = truly_random_init(n_sources, inverse.x_bounds, inverse.y_bounds, seed)
        
        result = minimize(inverse._objective, x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 2000})
        
        sources_rec = inverse._params_to_sources(result.x)
        rmse = compute_position_rmse(sources_true, sources_rec)
        
        success = rmse < 0.05
        if success:
            successes += 1
        
        status = "✅" if success else "❌"
        print(f"  Seed {seed:2d}: obj={result.fun:.4e}, RMSE={rmse:.4f} {status}")
    
    print(f"\n" + "="*60)
    print(f"SUMMARY: {successes}/20 succeeded ({100*successes/20:.0f}%)")
    print("="*60)


if __name__ == '__main__':
    main()
