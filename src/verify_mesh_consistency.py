#!/usr/bin/env python3
"""
Verify mesh consistency between forward and inverse solvers.

This script diagnoses the mesh mismatch bug that caused random solver failures.
The bug was: run_fem_polygon_nonlinear created a NEW mesh inside the function,
but the forward data was generated with a DIFFERENT mesh. Even with the same
sensor_locations, mesh node ordering could differ, causing sensor index mismatch.

Run this to verify the fix is working.
"""

import numpy as np
from mesh import create_polygon_mesh, get_brain_boundary, create_brain_mesh
from fem_solver import FEMForwardSolver, FEMNonlinearInverseSolver

def test_mesh_consistency():
    """Test that using the same mesh gives consistent results."""
    
    print("="*60)
    print("MESH CONSISTENCY TEST")
    print("="*60)
    
    # Create brain domain vertices
    boundary = get_brain_boundary(n_points=100)
    vertices = [tuple(p) for p in boundary]
    
    # Create sensor locations
    n_sensors = 100
    sensor_locations = get_brain_boundary(n_points=n_sensors)
    
    # Test sources
    sources = [
        ((-0.6, 0.3), 1.0),
        ((0.6, 0.3), 1.0),
        ((-0.5, -0.25), -1.0),
        ((0.5, -0.25), -1.0),
    ]
    
    print(f"\nSources: {sources}")
    print(f"Sensor locations shape: {sensor_locations.shape}")
    
    # Create ONE mesh for both forward and inverse
    print("\n--- Creating single mesh for consistency ---")
    mesh_data = create_brain_mesh(resolution=0.1, sensor_locations=sensor_locations)
    nodes, elements, boundary_idx, interior_idx, sensor_idx = mesh_data
    print(f"Mesh nodes: {len(nodes)}, Elements: {len(elements)}")
    print(f"Boundary nodes: {len(boundary_idx)}, Sensor indices: {len(sensor_idx)}")
    
    # Forward solve
    forward = FEMForwardSolver(resolution=0.1, verbose=False, 
                               mesh_data=mesh_data,
                               sensor_locations=sensor_locations)
    u_measured = forward.solve(sources)
    print(f"\nForward solution shape: {u_measured.shape}")
    print(f"Forward solution range: [{u_measured.min():.4f}, {u_measured.max():.4f}]")
    
    # Inverse solve with SAME mesh
    print("\n--- Inverse solve with SAME mesh ---")
    inverse = FEMNonlinearInverseSolver.from_polygon(vertices, n_sources=4,
                                                      resolution=0.1, verbose=False,
                                                      sensor_locations=sensor_locations,
                                                      mesh_data=mesh_data)
    inverse.set_measured_data(u_measured)
    result = inverse.solve(method='L-BFGS-B', n_restarts=5, maxiter=500)
    
    sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
    
    # Compute error
    from comparison import compute_metrics
    forward_rec = FEMForwardSolver(resolution=0.1, verbose=False, 
                                   mesh_data=mesh_data,
                                   sensor_locations=sensor_locations)
    u_rec = forward_rec.solve(sources_rec)
    metrics = compute_metrics(sources, sources_rec, u_measured, u_rec)
    
    print(f"\nRecovered sources: {sources_rec}")
    print(f"Position RMSE: {metrics['position_rmse']:.6f}")
    print(f"Intensity RMSE: {metrics['intensity_rmse']:.6f}")
    
    passed = metrics['position_rmse'] < 0.01
    print(f"\n{'PASS' if passed else 'FAIL'}: Position RMSE {'<' if passed else '>'} 0.01")
    
    # Now test what happens with a DIFFERENT mesh (the bug)
    print("\n" + "="*60)
    print("BUG DEMONSTRATION: Using DIFFERENT mesh for inverse")
    print("="*60)
    
    # Create a NEW mesh - this is what the bug was doing
    mesh_data_new = create_brain_mesh(resolution=0.1, sensor_locations=sensor_locations)
    
    # Check if sensor indices are the same
    _, _, _, _, sensor_idx_old = mesh_data
    _, _, _, _, sensor_idx_new = mesh_data_new
    
    print(f"\nOld mesh sensor indices (first 10): {sensor_idx_old[:10]}")
    print(f"New mesh sensor indices (first 10): {sensor_idx_new[:10]}")
    
    if not np.array_equal(sensor_idx_old, sensor_idx_new):
        print("\nWARNING: Sensor indices DIFFER between meshes!")
        print("This causes u_measured[i] to correspond to different physical locations!")
    else:
        print("\nNote: Sensor indices happen to match (deterministic gmsh)")
    
    return passed


if __name__ == '__main__':
    test_mesh_consistency()
