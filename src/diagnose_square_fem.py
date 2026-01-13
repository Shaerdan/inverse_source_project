#!/usr/bin/env python3
"""
Focused diagnostic for Square FEM failure.

The FEM solver requires explicit sensor_locations to work correctly.

Usage:
    cd src/
    python diagnose_square_fem.py
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from comparison import create_domain_sources
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


def generate_square_sensors(n_sensors=100):
    """Generate sensor locations evenly distributed on square boundary."""
    perimeter = 8.0  # Square with side 2 has perimeter 8
    sensor_locations = []
    for i in range(n_sensors):
        t = i * perimeter / n_sensors
        if t < 2:  # Bottom edge
            sensor_locations.append((-1 + t, -1))
        elif t < 4:  # Right edge
            sensor_locations.append((1, -1 + (t - 2)))
        elif t < 6:  # Top edge
            sensor_locations.append((1 - (t - 4), 1))
        else:  # Left edge
            sensor_locations.append((-1, 1 - (t - 6)))
    return np.array(sensor_locations)


def main():
    print("="*70)
    print("SQUARE FEM DIAGNOSTIC")
    print("="*70)
    
    vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    sources = create_domain_sources('square')
    n_sources = len(sources)
    
    print(f"\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources):
        print(f"  {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
    
    # Generate sensor locations (REQUIRED for FEM)
    print("\n" + "="*60)
    print("Setup: Creating mesh with 100 explicit sensor locations")
    print("="*60)
    
    sensor_locations = generate_square_sensors(100)
    print(f"Generated {len(sensor_locations)} sensor locations")
    
    mesh_data = create_polygon_mesh(vertices, 0.1, sensor_locations=sensor_locations)
    print(f"\nMesh created (mesh_data has {len(mesh_data)} elements):")
    if len(mesh_data) >= 1:
        print(f"  Element 0 (points): {len(mesh_data[0])} points")
    if len(mesh_data) >= 5:
        print(f"  Element 4 (sensor_indices): {len(mesh_data[4])} sensors")
    
    # Forward solve
    print("\n" + "="*60)
    print("TEST 1: Forward Solver")
    print("="*60)
    
    forward = FEMForwardSolver(resolution=0.1, verbose=False, mesh_data=mesh_data,
                                sensor_locations=sensor_locations)
    u_data = forward.solve(sources)
    print(f"\n  u_data shape: {u_data.shape}")
    print(f"  u_data range: [{u_data.min():.4f}, {u_data.max():.4f}]")
    
    if hasattr(forward, 'sensor_indices'):
        print(f"  Forward sensor_indices: {len(forward.sensor_indices)}")
    
    # Inverse solve setup - use from_polygon for proper initialization
    print("\n" + "="*60)
    print("TEST 2: Inverse Solver Setup")
    print("="*60)
    
    inverse = FEMNonlinearInverseSolver.from_polygon(
        vertices, n_sources=n_sources, resolution=0.1, 
        verbose=False, sensor_locations=sensor_locations, mesh_data=mesh_data
    )
    print(f"\nInverse solver created via from_polygon()")
    if hasattr(inverse, 'forward') and hasattr(inverse.forward, 'sensor_indices'):
        print(f"  Internal forward sensor_indices: {len(inverse.forward.sensor_indices)}")
    
    inverse.set_measured_data(u_data)
    
    # Verify forward model consistency
    print("\n--- Verifying forward model consistency ---")
    u_check = inverse.forward.solve(sources)
    print(f"  u_data shape: {u_data.shape}, u_check shape: {u_check.shape}")
    if u_data.shape == u_check.shape:
        print(f"  Forward solver in inverse gives same u_data: {np.allclose(u_data, u_check)}")
        print(f"  Max difference: {np.max(np.abs(u_data - u_check)):.2e}")
    else:
        print(f"  ⚠️ SHAPES DON'T MATCH!")
    
    # Test objective at truth
    print("\n" + "="*60)
    print("TEST 3: Objective at True Solution")
    print("="*60)
    
    # FEM solver params = [x1, y1, q1, x2, y2, q2, ..., x_{n-1}, y_{n-1}, q_{n-1}, x_n, y_n]
    params = []
    for i, ((x, y), q) in enumerate(sources):
        params.extend([x, y])
        if i < len(sources) - 1:
            params.append(q)
    params = np.array(params)
    print(f"\n  params shape: {params.shape} (expected {3*(n_sources-1) + 2} = {3*(n_sources-1) + 2})")
    
    obj_at_truth = inverse._objective(params)
    print(f"  Objective at truth: {obj_at_truth:.6e}")
    
    # Numerical gradient
    eps = 1e-6
    grad = np.zeros_like(params)
    for i in range(len(params)):
        p_plus = params.copy()
        p_plus[i] += eps
        p_minus = params.copy()
        p_minus[i] -= eps
        grad[i] = (inverse._objective(p_plus) - inverse._objective(p_minus)) / (2*eps)
    
    print(f"  ||gradient||: {np.linalg.norm(grad):.6e}")
    print(f"  Max |grad|: {np.max(np.abs(grad)):.6e}")
    
    if np.linalg.norm(grad) > 1e-3:
        print(f"\n  ⚠️ GRADIENT IS LARGE - true solution is NOT a minimum!")
    
    # Run L-BFGS-B
    print("\n" + "="*60)
    print("TEST 4: L-BFGS-B Optimization")
    print("="*60)
    
    result = inverse.solve(method='L-BFGS-B', n_restarts=5, maxiter=1000)
    sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
    rmse = compute_position_rmse(sources, sources_rec)
    
    print(f"\n  Residual: {result.residual:.6e}")
    print(f"  Position RMSE: {rmse:.6f}")
    
    print(f"\n  Recovered sources:")
    for i, ((x, y), q) in enumerate(sources_rec):
        print(f"    {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
    
    # Verify recovered sources
    u_rec = inverse.forward.solve(sources_rec)
    actual_residual = np.linalg.norm(u_rec - u_data)
    print(f"\n  Actual residual ||u_rec - u_data||: {actual_residual:.6e}")
    
    # Run differential_evolution (matching CLI's exact pattern)
    print("\n" + "="*60)
    print("TEST 5: Differential Evolution (CLI pattern)")
    print("="*60)
    
    inverse2 = FEMNonlinearInverseSolver.from_polygon(
        vertices, n_sources=n_sources, resolution=0.1,
        verbose=False, sensor_locations=sensor_locations, mesh_data=mesh_data
    )
    inverse2.set_measured_data(u_data)
    
    # CLI does: np.random.seed(seed) before calling solve
    np.random.seed(42)
    result2 = inverse2.solve(method='differential_evolution', maxiter=500)
    sources_rec2 = [((s.x, s.y), s.intensity) for s in result2.sources]
    rmse2 = compute_position_rmse(sources, sources_rec2)
    
    print(f"\n  Residual: {result2.residual:.6e}")
    print(f"  Position RMSE: {rmse2:.6f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n  L-BFGS-B RMSE: {rmse:.6f} {'✅' if rmse < 0.05 else '❌'}")
    print(f"  diff_evol RMSE: {rmse2:.6f} {'✅' if rmse2 < 0.05 else '❌'}")


if __name__ == '__main__':
    main()
