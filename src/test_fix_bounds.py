#!/usr/bin/env python3
"""
Test fix: Use bounds inscribed in domain instead of hard penalty.

For disk of radius r, inscribed square has half-side = r/√2
This ensures 100% of bounded points are in valid domain.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from comparison import create_domain_sources, get_sensor_locations
from fem_solver import FEMForwardSolver, FEMNonlinearInverseSolver
from mesh import create_disk_mesh, create_ellipse_mesh
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


def objective_no_penalty(inverse, params):
    """Objective WITHOUT the hard penalty - just forward solve."""
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
    print("TESTING FIX: Inscribed Bounds (No Penalty)")
    print("="*70)
    
    # =========================================================================
    # DISK
    # =========================================================================
    print("\n" + "="*60)
    print("DISK DOMAIN")
    print("="*60)
    
    sources_true = create_domain_sources('disk')
    n_sources = len(sources_true)
    
    print(f"\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources_true):
        r = np.sqrt(x**2 + y**2)
        print(f"  {i+1}: ({x:.4f}, {y:.4f}), r={r:.3f}, q={q:.4f}")
    
    sensor_locations = get_sensor_locations('disk', None, 100)
    mesh_data = create_disk_mesh(0.1, sensor_locations=sensor_locations)
    
    forward = FEMForwardSolver(resolution=0.1, verbose=False, mesh_data=mesh_data,
                                sensor_locations=sensor_locations)
    u_data = forward.solve(sources_true)
    
    inverse = FEMNonlinearInverseSolver(n_sources=n_sources, resolution=0.1,
                                         verbose=False, mesh_data=mesh_data)
    inverse.set_measured_data(u_data)
    
    # OLD bounds (outside penalty region)
    old_bounds = [(-0.9, 0.9), (-0.9, 0.9)] * n_sources
    for i in range(n_sources - 1):
        old_bounds.insert(3*i + 2, (-5.0, 5.0))
    
    # NEW bounds: inscribed in disk of radius 0.85
    # Inscribed square has half-side = 0.85/√2 ≈ 0.6
    # But true sources are at r=0.75, so we need bigger bounds
    # Use 0.75/√2 ≈ 0.53 as inner, 0.85/√2 ≈ 0.6 as margin
    # Actually, let's just use bounds that contain the true sources: 0.8
    inscribed_r = 0.85 / np.sqrt(2)  # ≈ 0.6
    print(f"\n  Old bounds: x,y ∈ (-0.9, 0.9)")
    print(f"  New bounds (inscribed): x,y ∈ ({-inscribed_r:.3f}, {inscribed_r:.3f})")
    
    # Check if true sources are within inscribed bounds
    print(f"\n  True sources within inscribed bounds?")
    all_in = True
    for i, ((x, y), q) in enumerate(sources_true):
        in_bounds = abs(x) <= inscribed_r and abs(y) <= inscribed_r
        print(f"    Source {i+1}: ({x:.2f}, {y:.2f}) -> {in_bounds}")
        if not in_bounds:
            all_in = False
    
    if not all_in:
        print(f"\n  ⚠️ True sources are OUTSIDE inscribed bounds!")
        print(f"     This means inscribed bounds are too tight for r=0.75 sources.")
        print(f"     Need bounds that contain sources but stay inside penalty region.")
        
        # Find maximum safe bounds
        # For point (x, y), need x² + y² < 0.85²
        # For corner (b, b), need 2b² < 0.85², so b < 0.85/√2 ≈ 0.6
        # But sources at r=0.75 on axes are fine (0.75, 0) is inside disk
        # The problem is only corners
        
        # Use CIRCULAR constraint via soft penalty instead
        print(f"\n  Alternative: Use POLAR coordinates to naturally constrain to disk")
    else:
        # Test with inscribed bounds
        new_bounds = [((-inscribed_r, inscribed_r), (-inscribed_r, inscribed_r))] * n_sources
        # ... build properly
    
    # Better approach: Test WITHOUT penalty, with slightly tighter bounds
    print(f"\n--- Testing with NO penalty, bounds = ±0.8 ---")
    
    safe_bound = 0.8  # This keeps corners at (0.8, 0.8) with r=1.13 > 0.85, still triggers penalty
    # Need even tighter: 0.6
    safe_bound = 0.6
    
    # But wait, true sources are at (0.75, 0) which IS within penalty (r=0.75 < 0.85)
    # The issue is random init hitting corners
    
    # Let's test: use circular-aware init (polar coords)
    print(f"\n--- Testing with POLAR-AWARE random init ---")
    
    def polar_random_init(n_sources, max_r, seed):
        """Initialize in polar coords, ensuring r < max_r"""
        np.random.seed(seed)
        x0 = []
        for i in range(n_sources):
            r = np.random.uniform(0.1, max_r * 0.95)  # Stay inside
            theta = np.random.uniform(0, 2*np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            x0.extend([x, y])
            if i < n_sources - 1:
                q = np.random.uniform(-2, 2)
                x0.append(q)
        return np.array(x0)
    
    # Build bounds that stay inside disk
    max_r = 0.85
    bounds_disk = []
    for i in range(n_sources):
        bounds_disk.extend([(-max_r, max_r), (-max_r, max_r)])
        if i < n_sources - 1:
            bounds_disk.append((-5.0, 5.0))
    
    successes_polar = 0
    successes_no_penalty = 0
    
    print(f"\n  Testing 20 polar-aware inits with original objective (has penalty):")
    for seed in range(20):
        x0 = polar_random_init(n_sources, max_r, seed)
        
        # Check if any source in penalty region
        sources_init = inverse._params_to_sources(x0)
        max_r_init = max(np.sqrt(x**2 + y**2) for (x, y), _ in sources_init)
        
        result = minimize(inverse._objective, x0, method='L-BFGS-B',
                          bounds=bounds_disk, options={'maxiter': 2000})
        
        sources_rec = inverse._params_to_sources(result.x)
        rmse = compute_position_rmse(sources_true, sources_rec)
        
        if rmse < 0.05:
            successes_polar += 1
        
        status = "✅" if rmse < 0.05 else "❌"
        penalty_hit = "PENALTY" if result.fun >= 1e9 else f"obj={result.fun:.2e}"
        print(f"    Seed {seed:2d}: max_r_init={max_r_init:.3f}, {penalty_hit}, RMSE={rmse:.4f} {status}")
    
    print(f"\n  Polar-aware init success rate: {successes_polar}/20 ({100*successes_polar/20:.0f}%)")
    
    # Test with no penalty at all
    print(f"\n  Testing 20 polar-aware inits WITHOUT penalty:")
    for seed in range(20):
        x0 = polar_random_init(n_sources, max_r, seed)
        
        result = minimize(lambda p: objective_no_penalty(inverse, p), x0, 
                          method='L-BFGS-B', bounds=bounds_disk, options={'maxiter': 2000})
        
        sources_rec = inverse._params_to_sources(result.x)
        rmse = compute_position_rmse(sources_true, sources_rec)
        
        if rmse < 0.05:
            successes_no_penalty += 1
        
        status = "✅" if rmse < 0.05 else "❌"
        print(f"    Seed {seed:2d}: obj={result.fun:.4e}, RMSE={rmse:.4f} {status}")
    
    print(f"\n  No-penalty success rate: {successes_no_penalty}/20 ({100*successes_no_penalty/20:.0f}%)")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"""
  Original (rectangular init, hard penalty): 0% success
  Polar-aware init, hard penalty:           {100*successes_polar/20:.0f}% success
  Polar-aware init, no penalty:             {100*successes_no_penalty/20:.0f}% success
  
  FIX: Either use polar-aware initialization OR remove hard penalty.
""")


if __name__ == '__main__':
    main()
