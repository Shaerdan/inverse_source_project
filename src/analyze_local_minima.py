#!/usr/bin/env python3
"""
Analyze the local minima that L-BFGS-B gets stuck in.

Questions:
1. What do these local minima look like?
2. Are they "almost correct" solutions or completely wrong?
3. Is there a pattern?
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
    print("ANALYZING LOCAL MINIMA STRUCTURE")
    print("="*70)
    
    vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    sources_true = create_domain_sources('square')
    n_sources = len(sources_true)
    
    print(f"\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources_true):
        print(f"  {i+1}: ({x:+.2f}, {y:+.2f}), q={q:+.2f}")
    
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
    
    bounds = []
    for i in range(n_sources):
        bounds.extend([inverse.x_bounds, inverse.y_bounds])
        if i < n_sources - 1:
            bounds.append((-5.0, 5.0))
    
    # Collect failed runs
    print("\n" + "="*60)
    print("COLLECTING FAILED OPTIMIZATIONS")
    print("="*60)
    
    failures = []
    successes = []
    
    for seed in range(50):
        x0 = truly_random_init(n_sources, inverse.x_bounds, inverse.y_bounds, seed)
        
        result = minimize(
            inverse._objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000}
        )
        
        sources_rec = inverse._params_to_sources(result.x)
        rmse = compute_position_rmse(sources_true, sources_rec)
        
        if rmse < 0.05:
            successes.append((seed, result.fun, sources_rec))
        else:
            failures.append((seed, result.fun, rmse, sources_rec))
    
    print(f"\n{len(successes)} successes, {len(failures)} failures")
    
    # Analyze failures
    print("\n" + "="*60)
    print("WHAT DO THE LOCAL MINIMA LOOK LIKE?")
    print("="*60)
    
    print("\nTrue sources (for reference):")
    print("  TL: (-0.75, +0.75), q=+1")
    print("  TR: (+0.75, +0.75), q=+1")
    print("  BL: (-0.75, -0.75), q=-1")
    print("  BR: (+0.75, -0.75), q=-1")
    
    # Group failures by similar final objective
    obj_groups = {}
    for seed, obj, rmse, sources in failures:
        # Round to 2 decimal places to group similar objectives
        obj_key = round(obj, 2)
        if obj_key not in obj_groups:
            obj_groups[obj_key] = []
        obj_groups[obj_key].append((seed, obj, rmse, sources))
    
    print(f"\n{len(obj_groups)} distinct local minima found:")
    for obj_key in sorted(obj_groups.keys()):
        group = obj_groups[obj_key]
        print(f"\n  Objective ≈ {obj_key:.2f} ({len(group)} runs)")
        
        # Show one example from each group
        seed, obj, rmse, sources = group[0]
        print(f"    Example (seed={seed}, RMSE={rmse:.3f}):")
        for i, ((x, y), q) in enumerate(sources):
            print(f"      S{i+1}: ({x:+.3f}, {y:+.3f}), q={q:+.3f}")
    
    # Check if sources are "swapped" in failures
    print("\n" + "="*60)
    print("ARE SOURCES 'SWAPPED' OR COMPLETELY WRONG?")
    print("="*60)
    
    print("\nFor each failure, checking if recovered sources are near TRUE positions:")
    
    true_positions = np.array([s[0] for s in sources_true])  # (4, 2)
    
    for seed, obj, rmse, sources in failures[:10]:
        rec_positions = np.array([s[0] for s in sources])
        
        # For each recovered source, find nearest true source
        print(f"\n  Seed {seed} (obj={obj:.3f}, RMSE={rmse:.3f}):")
        
        for i, (rx, ry) in enumerate(rec_positions):
            rq = sources[i][1]
            dists = np.sqrt((true_positions[:, 0] - rx)**2 + (true_positions[:, 1] - ry)**2)
            nearest_idx = np.argmin(dists)
            nearest_dist = dists[nearest_idx]
            true_q = sources_true[nearest_idx][1]
            
            if nearest_dist < 0.3:
                # Close to a true source
                sign_match = "✓" if np.sign(rq) == np.sign(true_q) else "✗ WRONG SIGN"
                print(f"    Rec {i+1}: ({rx:+.2f}, {ry:+.2f}) q={rq:+.2f} → near True {nearest_idx+1} (d={nearest_dist:.2f}) {sign_match}")
            else:
                print(f"    Rec {i+1}: ({rx:+.2f}, {ry:+.2f}) q={rq:+.2f} → NOT NEAR any true source (min d={nearest_dist:.2f})")
    
    # Check if it's a symmetry issue
    print("\n" + "="*60)
    print("CHECKING FOR SYMMETRY-RELATED ISSUES")
    print("="*60)
    
    # The square has 8-fold symmetry. Are failures related to symmetry confusion?
    print("\nTrue sources have diagonal symmetry (TL=TR q, BL=BR q)")
    print("Checking if failures have sources collapsed or mirrored...")
    
    for seed, obj, rmse, sources in failures[:5]:
        rec_positions = np.array([s[0] for s in sources])
        rec_intensities = np.array([s[1] for s in sources])
        
        # Check if any two sources are very close (collapsed)
        n = len(rec_positions)
        print(f"\n  Seed {seed}:")
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(rec_positions[i] - rec_positions[j])
                if d < 0.2:
                    print(f"    ⚠️ Sources {i+1} and {j+1} are VERY CLOSE (d={d:.3f})")
        
        # Check intensity sum
        total_q = sum(rec_intensities)
        print(f"    Intensity sum: {total_q:.4f} (should be ~0)")
    
    # Test: What if we use WIDER bounds?
    print("\n" + "="*60)
    print("TEST: WIDER BOUNDS [-0.95, 0.95]")
    print("="*60)
    
    wide_bounds = []
    for i in range(n_sources):
        wide_bounds.extend([(-0.95, 0.95), (-0.95, 0.95)])
        if i < n_sources - 1:
            wide_bounds.append((-5.0, 5.0))
    
    wide_successes = 0
    for seed in range(20):
        x0 = truly_random_init(n_sources, (-0.95, 0.95), (-0.95, 0.95), seed)
        
        result = minimize(
            inverse._objective,
            x0,
            method='L-BFGS-B',
            bounds=wide_bounds,
            options={'maxiter': 2000}
        )
        
        sources_rec = inverse._params_to_sources(result.x)
        rmse = compute_position_rmse(sources_true, sources_rec)
        
        if rmse < 0.05:
            wide_successes += 1
        
        status = "✅" if rmse < 0.05 else "❌"
        print(f"  Seed {seed}: obj={result.fun:.4e}, RMSE={rmse:.4f} {status}")
    
    print(f"\n  Wide bounds: {wide_successes}/20 succeeded")


if __name__ == '__main__':
    main()
