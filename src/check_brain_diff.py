#!/usr/bin/env python3
"""
Check what's different between CLI and diagnostic for brain FEM.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from comparison import create_domain_sources, get_conformal_map
from mesh import create_brain_mesh, get_brain_boundary
from fem_solver import FEMForwardSolver, FEMNonlinearInverseSolver

# Like diagnostic script does
print("="*60)
print("DIAGNOSTIC SCRIPT APPROACH (no explicit sensor_locations)")
print("="*60)

sources = create_domain_sources('brain')
print(f"Sources: {sources}")

mesh_data = create_brain_mesh(0.1)
points, cells, boundary_nodes, boundary_coords = mesh_data
print(f"\nMesh created:")
print(f"  Total points: {len(points)}")
print(f"  Boundary nodes: {len(boundary_nodes)}")

forward = FEMForwardSolver(resolution=0.1, verbose=False, mesh_data=mesh_data)
# Check what FEMForwardSolver uses as sensors
print(f"  Forward solver n_sensors: {len(forward.sensor_indices) if hasattr(forward, 'sensor_indices') else 'N/A'}")

u_data = forward.solve(sources)
print(f"  u_data shape: {u_data.shape}")

# Like CLI does
print("\n" + "="*60)
print("CLI APPROACH (with 100 fixed sensor_locations)")
print("="*60)

# Generate 100 fixed sensor locations on brain boundary
sensor_locations = get_brain_boundary(n_points=100)
print(f"Generated {len(sensor_locations)} sensor locations")

mesh_data_cli = create_brain_mesh(0.1, sensor_locations=sensor_locations)
points_cli, cells_cli, boundary_nodes_cli, boundary_coords_cli = mesh_data_cli
print(f"\nMesh created:")
print(f"  Total points: {len(points_cli)}")
print(f"  Boundary nodes: {len(boundary_nodes_cli)}")

forward_cli = FEMForwardSolver(resolution=0.1, verbose=False, mesh_data=mesh_data_cli,
                                sensor_locations=sensor_locations)
print(f"  Forward solver n_sensors: {len(forward_cli.sensor_indices) if hasattr(forward_cli, 'sensor_indices') else 'N/A'}")

u_data_cli = forward_cli.solve(sources)
print(f"  u_data_cli shape: {u_data_cli.shape}")

# Now test diff_evol with both setups
print("\n" + "="*60)
print("DIFFERENTIAL EVOLUTION COMPARISON")
print("="*60)

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def compute_position_rmse(sources_true, sources_rec):
    pos_true = np.array([[s[0][0], s[0][1]] for s in sources_true])
    pos_rec = np.array([[s[0][0], s[0][1]] for s in sources_rec])
    cost = cdist(pos_true, pos_rec)
    row_ind, col_ind = linear_sum_assignment(cost)
    return np.sqrt(np.mean(cost[row_ind, col_ind]**2))

# Diagnostic approach
print("\nDiagnostic approach (no explicit sensors):")
inverse = FEMNonlinearInverseSolver(n_sources=4, resolution=0.1, verbose=False, mesh_data=mesh_data)
inverse.set_measured_data(u_data)
result = inverse.solve(method='differential_evolution', maxiter=500)
sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
rmse = compute_position_rmse(sources, sources_rec)
print(f"  RMSE: {rmse:.6f}")

# CLI approach
print("\nCLI approach (100 fixed sensors):")
inverse_cli = FEMNonlinearInverseSolver(n_sources=4, resolution=0.1, verbose=False, 
                                         mesh_data=mesh_data_cli)
inverse_cli.set_measured_data(u_data_cli)
result_cli = inverse_cli.solve(method='differential_evolution', maxiter=500)
sources_rec_cli = [((s.x, s.y), s.intensity) for s in result_cli.sources]
rmse_cli = compute_position_rmse(sources, sources_rec_cli)
print(f"  RMSE: {rmse_cli:.6f}")
