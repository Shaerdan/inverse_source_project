#!/usr/bin/env python3
"""
Verify the sensor_locations bug in FEMNonlinearInverseSolver.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from comparison import create_domain_sources
from fem_solver import FEMForwardSolver, FEMNonlinearInverseSolver
from mesh import create_polygon_mesh


def generate_square_sensors(n_sensors=100):
    perimeter = 8.0
    sensor_locations = []
    for i in range(n_sensors):
        t = i * perimeter / n_sensors
        if t < 2:
            sensor_locations.append((-1 + t, -1))
        elif t < 4:
            sensor_locations.append((1, -1 + (t - 2)))
        elif t < 6:
            sensor_locations.append((1 - (t - 4), 1))
        else:
            sensor_locations.append((-1, 1 - (t - 6)))
    return np.array(sensor_locations)


def main():
    print("="*70)
    print("VERIFYING sensor_locations BUG")
    print("="*70)
    
    vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    sources = create_domain_sources('square')
    
    # Generate square sensors
    sensor_locations = generate_square_sensors(100)
    print(f"\nOur sensor_locations (on square boundary):")
    print(f"  Shape: {sensor_locations.shape}")
    print(f"  First 3: {sensor_locations[:3]}")
    print(f"  x range: [{sensor_locations[:,0].min():.3f}, {sensor_locations[:,0].max():.3f}]")
    print(f"  y range: [{sensor_locations[:,1].min():.3f}, {sensor_locations[:,1].max():.3f}]")
    
    # Create mesh
    mesh_data = create_polygon_mesh(vertices, 0.1, sensor_locations=sensor_locations)
    
    # Create EXTERNAL forward solver (what generates u_data)
    forward_external = FEMForwardSolver(resolution=0.1, verbose=False, mesh_data=mesh_data,
                                         sensor_locations=sensor_locations)
    
    print(f"\nExternal forward solver sensor_locations:")
    ext_sensors = forward_external.sensor_locations
    print(f"  Shape: {ext_sensors.shape}")
    print(f"  First 3: {ext_sensors[:3]}")
    print(f"  x range: [{ext_sensors[:,0].min():.3f}, {ext_sensors[:,0].max():.3f}]")
    print(f"  y range: [{ext_sensors[:,1].min():.3f}, {ext_sensors[:,1].max():.3f}]")
    
    # Create inverse solver using from_polygon
    inverse = FEMNonlinearInverseSolver.from_polygon(
        vertices, n_sources=4, resolution=0.1,
        verbose=False, sensor_locations=sensor_locations, mesh_data=mesh_data
    )
    
    # Check INTERNAL forward solver
    print(f"\nInverse solver's INTERNAL forward solver sensor_locations:")
    int_sensors = inverse.forward.sensor_locations
    print(f"  Shape: {int_sensors.shape}")
    print(f"  First 3: {int_sensors[:3]}")
    print(f"  x range: [{int_sensors[:,0].min():.3f}, {int_sensors[:,0].max():.3f}]")
    print(f"  y range: [{int_sensors[:,1].min():.3f}, {int_sensors[:,1].max():.3f}]")
    
    # Check if they're on a CIRCLE (bug) or SQUARE (correct)
    radii = np.sqrt(int_sensors[:,0]**2 + int_sensors[:,1]**2)
    print(f"\n  Radii of internal sensors:")
    print(f"    min: {radii.min():.6f}")
    print(f"    max: {radii.max():.6f}")
    print(f"    std: {radii.std():.6f}")
    
    if radii.std() < 0.01:
        print(f"\n  ⚠️ BUG CONFIRMED: Internal sensors are on a CIRCLE (std={radii.std():.6f})!")
        print(f"     They should be on the SQUARE boundary!")
    else:
        print(f"\n  ✅ Internal sensors are NOT on a circle")
    
    # Compare sensor indices
    print(f"\n  External forward sensor_indices: {forward_external.sensor_indices[:5]}...")
    print(f"  Internal forward sensor_indices: {inverse.forward.sensor_indices[:5]}...")
    print(f"  Same? {np.array_equal(forward_external.sensor_indices, inverse.forward.sensor_indices)}")


if __name__ == '__main__':
    main()
