#!/usr/bin/env python3
"""
Test if differential_evolution works for disk (like it does for square).

If DE works and L-BFGS-B doesn't, the issue is local minima, not forward model.
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
    """Objective WITHOUT the hard penalty."""
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
    print("DIFFERENTIAL EVOLUTION vs L-BFGS-B on DISK")
    print("="*70)
    
    sources_true = create_domain_sources('disk')
    n_sources = len(sources_true)
    
    print(f"\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources_true):
        print(f"  {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
    
    sensor_locations = get_sensor_locations('disk', None, 100)
    mesh_data = create_disk_mesh(0.1, sensor_locations=sensor_locations)
    
    forward = FEMForwardSolver(resolution=0.1, verbose=False, mesh_data=mesh_data,
                                sensor_locations=sensor_locations)
    u_data = forward.solve(sources_true)
    
    inverse = FEMNonlinearInverseSolver(n_sources=n_sources, resolution=0.1,
                                         verbose=False, mesh_data=mesh_data)
    inverse.set_measured_data(u_data)
    
    # Bounds that stay inside disk
    max_r = 0.85
    bounds = []
    for i in range(n_sources):
        bounds.extend([(-max_r, max_r), (-max_r, max_r)])
        if i < n_sources - 1:
            bounds.append((-5.0, 5.0))
    
    # Verify objective at true solution
    true_params = []
    for i, ((x, y), q) in enumerate(sources_true):
        true_params.extend([x, y])
        if i < n_sources - 1:
            true_params.append(q)
    true_params = np.array(true_params)
    
    obj_true = objective_no_penalty(inverse, true_params)
    print(f"\nObjective at truth: {obj_true:.6e}")
    
    # =========================================================================
    # Test differential_evolution
    # =========================================================================
    print("\n" + "="*60)
    print("DIFFERENTIAL EVOLUTION (5 runs)")
    print("="*60)
    
    de_successes = 0
    for seed in range(5):
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
        
        if rmse < 0.05:
            de_successes += 1
        
        status = "✅" if rmse < 0.05 else "❌"
        print(f"  Seed {seed}: obj={result.fun:.4e}, RMSE={rmse:.6f} {status}")
    
    print(f"\n  DE success rate: {de_successes}/5 ({100*de_successes/5:.0f}%)")
    
    # =========================================================================
    # Compare with L-BFGS-B (polar init, no penalty)
    # =========================================================================
    print("\n" + "="*60)
    print("L-BFGS-B with polar init, no penalty (5 runs)")
    print("="*60)
    
    def polar_random_init(n_sources, max_r, seed):
        np.random.seed(seed)
        x0 = []
        for i in range(n_sources):
            r = np.random.uniform(0.1, max_r * 0.95)
            theta = np.random.uniform(0, 2*np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            x0.extend([x, y])
            if i < n_sources - 1:
                x0.append(np.random.uniform(-2, 2))
        return np.array(x0)
    
    lbfgsb_successes = 0
    for seed in range(5):
        x0 = polar_random_init(n_sources, max_r, seed)
        
        result = minimize(
            lambda p: objective_no_penalty(inverse, p),
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000}
        )
        
        sources_rec = inverse._params_to_sources(result.x)
        rmse = compute_position_rmse(sources_true, sources_rec)
        
        if rmse < 0.05:
            lbfgsb_successes += 1
        
        status = "✅" if rmse < 0.05 else "❌"
        print(f"  Seed {seed}: obj={result.fun:.4e}, RMSE={rmse:.6f} {status}")
    
    print(f"\n  L-BFGS-B success rate: {lbfgsb_successes}/5 ({100*lbfgsb_successes/5:.0f}%)")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"""
  Differential Evolution: {de_successes}/5 ({100*de_successes/5:.0f}%)
  L-BFGS-B (best case):   {lbfgsb_successes}/5 ({100*lbfgsb_successes/5:.0f}%)
  
  If DE works and L-BFGS-B doesn't → Local minima issue
  If both fail → Forward model issue
""")


if __name__ == '__main__':
    main()
