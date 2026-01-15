#!/usr/bin/env python3
"""
Exactly replicate what the CLI does for Square FEM nonlinear.

This copies the exact code path from comparison.py to understand
why CLI gets RMSE 0.0005 but my diagnostic gets 0.556.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from comparison import (create_domain_sources, get_sensor_locations, 
                        run_fem_polygon_nonlinear, compute_metrics)
from fem_solver import FEMForwardSolver, FEMNonlinearInverseSolver
from mesh import create_polygon_mesh
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
    print("REPLICATING CLI EXACTLY")
    print("="*70)
    
    # Exact CLI parameters
    domain_type = 'square'
    seed = 42
    noise_level = 0.001
    forward_resolution = 0.1
    n_sensors = 100
    n_sources = 4
    
    # Step 1: Get sensor locations (exactly like CLI line 3099)
    sensor_locations = get_sensor_locations(domain_type, None, n_sensors)
    print(f"\n1. Sensor locations from CLI function:")
    print(f"   Shape: {sensor_locations.shape}")
    print(f"   First 3: {sensor_locations[:3]}")
    
    # Step 2: Get sources (exactly like CLI)
    sources_true = create_domain_sources(domain_type)
    print(f"\n2. True sources:")
    for i, ((x, y), q) in enumerate(sources_true):
        print(f"   {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
    
    # Step 3: Create mesh and forward solve (exactly like CLI lines 3137-3159)
    vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    polygon_mesh_data = create_polygon_mesh(vertices, forward_resolution, 
                                             sensor_locations=sensor_locations)
    print(f"\n3. Mesh created:")
    print(f"   mesh_data has {len(polygon_mesh_data)} elements")
    print(f"   Points: {len(polygon_mesh_data[0])}")
    print(f"   Sensors: {len(polygon_mesh_data[4])}")
    
    forward = FEMForwardSolver(resolution=forward_resolution, verbose=False, 
                               mesh_data=polygon_mesh_data,
                               sensor_locations=sensor_locations)
    u_clean = forward.solve(sources_true)
    print(f"\n4. Forward solve:")
    print(f"   u_clean shape: {u_clean.shape}")
    print(f"   u_clean range: [{u_clean.min():.4f}, {u_clean.max():.4f}]")
    
    # Step 4: Add noise (exactly like CLI lines 3183-3185)
    np.random.seed(seed)
    u_measured = u_clean + noise_level * np.random.randn(len(u_clean))
    print(f"\n5. After noise (seed={seed}):")
    print(f"   u_measured range: [{u_measured.min():.4f}, {u_measured.max():.4f}]")
    print(f"   Max noise: {np.max(np.abs(u_measured - u_clean)):.6f}")
    
    # Step 5: Run FEM polygon nonlinear (exactly like CLI lines 3892-3896)
    print(f"\n6. Running run_fem_polygon_nonlinear (CLI wrapper)...")
    result = run_fem_polygon_nonlinear(
        u_measured, sources_true, vertices,
        n_sources=n_sources, 
        optimizer='differential_evolution', 
        seed=seed,
        resolution=forward_resolution,
        sensor_locations=sensor_locations,
        mesh_data=polygon_mesh_data
    )
    print(f"   Position RMSE: {result.position_rmse:.6f}")
    print(f"   Boundary residual: {result.boundary_residual:.6f}")
    
    # Step 6: Also run it my way for comparison
    print(f"\n7. Running MY way for comparison...")
    
    inverse = FEMNonlinearInverseSolver.from_polygon(
        vertices, n_sources=n_sources, resolution=forward_resolution,
        verbose=False, sensor_locations=sensor_locations, mesh_data=polygon_mesh_data
    )
    inverse.set_measured_data(u_measured)
    
    np.random.seed(seed)  # CLI does this before solve
    result2 = inverse.solve(method='differential_evolution', maxiter=500)
    sources_rec = [((s.x, s.y), s.intensity) for s in result2.sources]
    rmse2 = compute_position_rmse(sources_true, sources_rec)
    
    print(f"   Position RMSE: {rmse2:.6f}")
    print(f"   Residual: {result2.residual:.6f}")
    
    # Step 7: Compare the two approaches
    print(f"\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"  CLI wrapper RMSE:  {result.position_rmse:.6f}")
    print(f"  My approach RMSE:  {rmse2:.6f}")
    print(f"  Same? {np.isclose(result.position_rmse, rmse2)}")
    
    if not np.isclose(result.position_rmse, rmse2):
        print(f"\n  ⚠️ DIFFERENT RESULTS! Let's investigate...")
        
        # Check if the measured data is the same
        print(f"\n  Checking if both use same u_measured...")


if __name__ == '__main__':
    main()
