#!/usr/bin/env python3
"""
Analyze the local minima that both DE and L-BFGS-B find on disk.

Both optimizers have 60% success rate, failing on the same seeds.
What do these local minima look like?
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from comparison import create_domain_sources, get_sensor_locations
from fem_solver import FEMForwardSolver, FEMNonlinearInverseSolver
from mesh import create_disk_mesh
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment, differential_evolution, minimize


def compute_position_rmse(sources_true, sources_rec):
    pos_true = np.array([[s[0][0], s[0][1]] for s in sources_true])
    pos_rec = np.array([[s[0][0], s[0][1]] for s in sources_rec])
    cost = cdist(pos_true, pos_rec)
    row_ind, col_ind = linear_sum_assignment(cost)
    return np.sqrt(np.mean(cost[row_ind, col_ind]**2))


def objective_no_penalty(inverse, params):
    sources = inverse._params_to_sources(params)
    u_computed = inverse.forward.solve(sources)
    if len(u_computed) != len(inverse.u_measured):
        from scipy.interpolate import interp1d
        interp = interp1d(inverse.forward.theta, u_computed, kind='linear', 
                        fill_value='extrapolate')
        theta_meas = np.linspace(0, 2*np.pi, len(inverse.u_measured), endpoint=False)
        u_computed = interp(theta_meas)
    return np.sum((u_computed - inverse.u_measured)**2)


def main():
    print("="*70)
    print("ANALYZING LOCAL MINIMA ON DISK")
    print("="*70)
    
    sources_true = create_domain_sources('disk')
    n_sources = len(sources_true)
    
    print(f"\nTrue sources (on axes, r=0.75):")
    for i, ((x, y), q) in enumerate(sources_true):
        theta = np.arctan2(y, x) * 180 / np.pi
        print(f"  {i+1}: ({x:+.4f}, {y:+.4f}), θ={theta:+6.1f}°, q={q:+.2f}")
    
    sensor_locations = get_sensor_locations('disk', None, 100)
    mesh_data = create_disk_mesh(0.1, sensor_locations=sensor_locations)
    
    forward = FEMForwardSolver(resolution=0.1, verbose=False, mesh_data=mesh_data,
                                sensor_locations=sensor_locations)
    u_data = forward.solve(sources_true)
    
    inverse = FEMNonlinearInverseSolver(n_sources=n_sources, resolution=0.1,
                                         verbose=False, mesh_data=mesh_data)
    inverse.set_measured_data(u_data)
    
    max_r = 0.85
    bounds = []
    for i in range(n_sources):
        bounds.extend([(-max_r, max_r), (-max_r, max_r)])
        if i < n_sources - 1:
            bounds.append((-5.0, 5.0))
    
    # Run many DE seeds to collect local minima
    print("\n" + "="*60)
    print("RUNNING 20 DE SEEDS")
    print("="*60)
    
    results = []
    for seed in range(20):
        result = differential_evolution(
            lambda p: objective_no_penalty(inverse, p),
            bounds,
            seed=seed,
            maxiter=500,
            polish=True,
            workers=1
        )
        
        sources_rec = inverse._params_to_sources(result.x)
        rmse = compute_position_rmse(sources_true, sources_rec)
        results.append((seed, result.fun, rmse, sources_rec))
        
        status = "✅" if rmse < 0.05 else "❌"
        print(f"  Seed {seed:2d}: obj={result.fun:.4e}, RMSE={rmse:.4f} {status}")
    
    # Analyze failures
    print("\n" + "="*60)
    print("ANALYZING FAILURES")
    print("="*60)
    
    failures = [(s, o, r, src) for s, o, r, src in results if r >= 0.05]
    successes = [(s, o, r, src) for s, o, r, src in results if r < 0.05]
    
    print(f"\n{len(successes)} successes, {len(failures)} failures")
    
    if failures:
        # Group by objective value
        obj_values = [o for _, o, _, _ in failures]
        print(f"\nFailure objectives: min={min(obj_values):.4f}, max={max(obj_values):.4f}")
        
        print("\n--- Failure patterns ---")
        for seed, obj, rmse, sources in failures[:5]:
            print(f"\nSeed {seed} (obj={obj:.4f}, RMSE={rmse:.3f}):")
            
            # Check each recovered source
            for i, ((x, y), q) in enumerate(sources):
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x) * 180 / np.pi
                
                # Find nearest true source
                true_pos = np.array([s[0] for s in sources_true])
                dists = np.sqrt((true_pos[:, 0] - x)**2 + (true_pos[:, 1] - y)**2)
                nearest = np.argmin(dists)
                
                print(f"  Rec {i+1}: ({x:+.3f}, {y:+.3f}), r={r:.3f}, θ={theta:+6.1f}°, q={q:+.3f}")
                print(f"          Nearest true: {nearest+1}, dist={dists[nearest]:.3f}")
        
        # Check for coalescence pattern
        print("\n--- Checking for coalescence ---")
        for seed, obj, rmse, sources in failures[:5]:
            positions = np.array([s[0] for s in sources])
            n = len(positions)
            
            print(f"\nSeed {seed}:")
            close_pairs = []
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < 0.3:
                        close_pairs.append((i+1, j+1, d))
            
            if close_pairs:
                print(f"  Close pairs (d < 0.3):")
                for i, j, d in close_pairs:
                    qi = sources[i-1][1]
                    qj = sources[j-1][1]
                    print(f"    Sources {i} and {j}: d={d:.3f}, q{i}={qi:+.2f}, q{j}={qj:+.2f}")
            else:
                print(f"  No close pairs - sources are separated")
                
            # Check if it's a rotation of the true solution
            rec_pos = positions
            true_pos = np.array([s[0] for s in sources_true])
            
            # Try different rotations
            print(f"  Checking for rotated solutions...")
            for rot_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
                rot_rad = rot_deg * np.pi / 180
                rot_matrix = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                                       [np.sin(rot_rad), np.cos(rot_rad)]])
                rotated_true = (rot_matrix @ true_pos.T).T
                
                # Compute RMSE between recovered and rotated true
                cost = cdist(rotated_true, rec_pos)
                row_ind, col_ind = linear_sum_assignment(cost)
                rot_rmse = np.sqrt(np.mean(cost[row_ind, col_ind]**2))
                
                if rot_rmse < 0.1:
                    print(f"    Rotation {rot_deg}°: RMSE={rot_rmse:.4f} ← MATCH!")
    
    # Check symmetry of the true solution
    print("\n" + "="*60)
    print("SYMMETRY ANALYSIS")
    print("="*60)
    
    print("\nTrue sources have 4-fold rotational symmetry (90° apart)")
    print("Intensities: +1, -1, +1, -1 (alternating)")
    print("\nThis creates a quadrupole pattern on boundary.")
    print("Local minima might be related to symmetry ambiguity.")
    
    # Compute boundary potential for a rotated solution
    print("\n--- Testing rotated solutions ---")
    
    for rot_deg in [0, 45, 90]:
        rot_rad = rot_deg * np.pi / 180
        rotated_sources = []
        for (x, y), q in sources_true:
            x_rot = x * np.cos(rot_rad) - y * np.sin(rot_rad)
            y_rot = x * np.sin(rot_rad) + y * np.cos(rot_rad)
            rotated_sources.append(((x_rot, y_rot), q))
        
        # Compute forward
        u_rotated = forward.solve(rotated_sources)
        
        # Compute residual
        residual = np.sum((u_rotated - u_data)**2)
        
        print(f"  Rotation {rot_deg}°: residual = {residual:.6e}")


if __name__ == '__main__':
    main()
