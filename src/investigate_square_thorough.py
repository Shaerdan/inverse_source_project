#!/usr/bin/env python3
"""
THOROUGH investigation of Square FEM solver.

Questions to answer:
1. Why does 4 extra mesh points cause RMSE to go from 0.0005 to 0.55?
2. Why does L-BFGS-B fail when diff_evol succeeds (in CLI)?
3. Is the solver fundamentally fragile?

Usage:
    cd src/
    python investigate_square_thorough.py
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from comparison import create_domain_sources, get_sensor_locations
from fem_solver import FEMForwardSolver, FEMNonlinearInverseSolver
from mesh import create_polygon_mesh
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment, minimize, differential_evolution


def compute_position_rmse(sources_true, sources_rec):
    pos_true = np.array([[s[0][0], s[0][1]] for s in sources_true])
    pos_rec = np.array([[s[0][0], s[0][1]] for s in sources_rec])
    cost = cdist(pos_true, pos_rec)
    row_ind, col_ind = linear_sum_assignment(cost)
    return np.sqrt(np.mean(cost[row_ind, col_ind]**2))


def main():
    print("="*70)
    print("THOROUGH INVESTIGATION: Square FEM Solver")
    print("="*70)
    
    vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    sources_true = create_domain_sources('square')
    n_sources = len(sources_true)
    
    print(f"\nTrue sources (at corners):")
    for i, ((x, y), q) in enumerate(sources_true):
        print(f"  {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
    
    # Use CLI's sensor generation
    sensor_locations = get_sensor_locations('square', None, 100)
    
    # Create mesh
    mesh_data = create_polygon_mesh(vertices, 0.1, sensor_locations=sensor_locations)
    n_mesh_points = len(mesh_data[0])
    print(f"\nMesh: {n_mesh_points} points, {len(mesh_data[4])} sensors")
    
    # Forward solve
    forward = FEMForwardSolver(resolution=0.1, verbose=False, mesh_data=mesh_data,
                                sensor_locations=sensor_locations)
    u_data = forward.solve(sources_true)
    print(f"u_data: shape={u_data.shape}, range=[{u_data.min():.4f}, {u_data.max():.4f}]")
    
    # Create inverse solver
    inverse = FEMNonlinearInverseSolver.from_polygon(
        vertices, n_sources=n_sources, resolution=0.1,
        verbose=False, sensor_locations=sensor_locations, mesh_data=mesh_data
    )
    inverse.set_measured_data(u_data)
    
    # =========================================================================
    # TEST 1: Verify objective function at true solution
    # =========================================================================
    print("\n" + "="*60)
    print("TEST 1: Objective at True Solution")
    print("="*60)
    
    # Build true params
    true_params = []
    for i, ((x, y), q) in enumerate(sources_true):
        true_params.extend([x, y])
        if i < n_sources - 1:
            true_params.append(q)
    true_params = np.array(true_params)
    
    obj_true = inverse._objective(true_params)
    print(f"\n  Objective at truth: {obj_true:.6e}")
    
    # Numerical gradient
    eps = 1e-7
    grad = np.zeros_like(true_params)
    for i in range(len(true_params)):
        p_plus = true_params.copy()
        p_plus[i] += eps
        p_minus = true_params.copy()
        p_minus[i] -= eps
        grad[i] = (inverse._objective(p_plus) - inverse._objective(p_minus)) / (2*eps)
    
    print(f"  ||gradient||: {np.linalg.norm(grad):.6e}")
    
    # =========================================================================
    # TEST 2: Check bounds
    # =========================================================================
    print("\n" + "="*60)
    print("TEST 2: Optimization Bounds")
    print("="*60)
    
    print(f"\n  inverse.x_bounds: {inverse.x_bounds}")
    print(f"  inverse.y_bounds: {inverse.y_bounds}")
    
    # Build bounds like solve() does
    bounds = []
    for i in range(n_sources):
        bounds.extend([inverse.x_bounds, inverse.y_bounds])
        if i < n_sources - 1:
            bounds.append((-5.0, 5.0))
    
    print(f"\n  True sources within bounds?")
    for i, ((x, y), q) in enumerate(sources_true):
        x_in = inverse.x_bounds[0] <= x <= inverse.x_bounds[1]
        y_in = inverse.y_bounds[0] <= y <= inverse.y_bounds[1]
        print(f"    Source {i+1}: x={x:.2f} in {inverse.x_bounds}? {x_in}, y={y:.2f} in {inverse.y_bounds}? {y_in}")
    
    # =========================================================================
    # TEST 3: L-BFGS-B from TRUE solution
    # =========================================================================
    print("\n" + "="*60)
    print("TEST 3: L-BFGS-B Starting from TRUE Solution")
    print("="*60)
    
    result_from_true = minimize(
        inverse._objective,
        true_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000}
    )
    
    print(f"\n  Converged: {result_from_true.success}")
    print(f"  Message: {result_from_true.message}")
    print(f"  Final objective: {result_from_true.fun:.6e}")
    print(f"  Iterations: {result_from_true.nit}")
    print(f"  Same as start? {np.allclose(result_from_true.x, true_params, atol=1e-4)}")
    
    if not np.allclose(result_from_true.x, true_params, atol=1e-4):
        print(f"\n  ⚠️ L-BFGS-B MOVED AWAY FROM TRUE SOLUTION!")
        print(f"  Start params: {true_params}")
        print(f"  Final params: {result_from_true.x}")
        
        # Check recovered sources
        final_sources = inverse._params_to_sources(result_from_true.x)
        rmse = compute_position_rmse(sources_true, final_sources)
        print(f"  RMSE: {rmse:.6f}")
    
    # =========================================================================
    # TEST 4: Check what initial guesses look like
    # =========================================================================
    print("\n" + "="*60)
    print("TEST 4: Initial Guesses from _get_initial_guess()")
    print("="*60)
    
    for restart in range(5):
        x0 = inverse._get_initial_guess('circle', restart)
        sources_init = inverse._params_to_sources(np.array(x0))
        obj_init = inverse._objective(np.array(x0))
        
        print(f"\n  Restart {restart}:")
        print(f"    Objective: {obj_init:.4f}")
        print(f"    Sources:")
        for i, ((x, y), q) in enumerate(sources_init):
            print(f"      {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
    
    # =========================================================================
    # TEST 5: Run L-BFGS-B with restarts and track each one
    # =========================================================================
    print("\n" + "="*60)
    print("TEST 5: L-BFGS-B with 5 Restarts (detailed)")
    print("="*60)
    
    best_result = None
    best_fun = np.inf
    
    for restart in range(5):
        x0 = inverse._get_initial_guess('circle', restart)
        result = minimize(
            inverse._objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        sources_rec = inverse._params_to_sources(result.x)
        rmse = compute_position_rmse(sources_true, sources_rec)
        
        print(f"\n  Restart {restart}:")
        print(f"    Init obj: {inverse._objective(np.array(x0)):.4f}")
        print(f"    Final obj: {result.fun:.6e}")
        print(f"    RMSE: {rmse:.6f}")
        print(f"    Converged: {result.success}")
        
        if result.fun < best_fun:
            best_fun = result.fun
            best_result = result
    
    best_sources = inverse._params_to_sources(best_result.x)
    best_rmse = compute_position_rmse(sources_true, best_sources)
    print(f"\n  BEST L-BFGS-B RMSE: {best_rmse:.6f}")
    
    # =========================================================================
    # TEST 6: Run differential_evolution
    # =========================================================================
    print("\n" + "="*60)
    print("TEST 6: Differential Evolution")
    print("="*60)
    
    de_result = differential_evolution(
        inverse._objective, 
        bounds, 
        maxiter=500,
        seed=42, 
        polish=True, 
        workers=1
    )
    
    de_sources = inverse._params_to_sources(de_result.x)
    de_rmse = compute_position_rmse(sources_true, de_sources)
    
    print(f"\n  Final objective: {de_result.fun:.6e}")
    print(f"  RMSE: {de_rmse:.6f}")
    print(f"  Converged: {de_result.success}")
    
    print(f"\n  Recovered sources:")
    for i, ((x, y), q) in enumerate(de_sources):
        print(f"    {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
    
    # =========================================================================
    # TEST 7: Objective landscape along line from bad init to truth
    # =========================================================================
    print("\n" + "="*60)
    print("TEST 7: Objective Landscape (init -> truth)")
    print("="*60)
    
    x0_bad = np.array(inverse._get_initial_guess('circle', 0))
    
    print(f"\n  Sampling objective along line from init to truth:")
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        x_interp = (1 - alpha) * x0_bad + alpha * true_params
        obj = inverse._objective(x_interp)
        print(f"    α={alpha:.1f}: obj={obj:.6f}")
    
    # =========================================================================
    # TEST 8: Check if problem is the bounds cutting off true solution
    # =========================================================================
    print("\n" + "="*60)
    print("TEST 8: Are true sources near/at bounds?")
    print("="*60)
    
    x_lo, x_hi = inverse.x_bounds
    y_lo, y_hi = inverse.y_bounds
    
    print(f"\n  Bounds: x in [{x_lo:.2f}, {x_hi:.2f}], y in [{y_lo:.2f}, {y_hi:.2f}]")
    print(f"\n  Distance from bounds:")
    for i, ((x, y), q) in enumerate(sources_true):
        dist_x = min(x - x_lo, x_hi - x)
        dist_y = min(y - y_lo, y_hi - y)
        print(f"    Source {i+1}: ({x:.2f}, {y:.2f}) -> dist_to_x_bound={dist_x:.3f}, dist_to_y_bound={dist_y:.3f}")
        if dist_x < 0.05 or dist_y < 0.05:
            print(f"      ⚠️ VERY CLOSE TO BOUND!")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n  Objective at truth: {obj_true:.6e}")
    print(f"  Gradient at truth: {np.linalg.norm(grad):.6e}")
    print(f"  L-BFGS-B from truth stays at truth: {np.allclose(result_from_true.x, true_params, atol=1e-4)}")
    print(f"  Best L-BFGS-B RMSE (5 restarts): {best_rmse:.6f}")
    print(f"  Differential evolution RMSE: {de_rmse:.6f}")


if __name__ == '__main__':
    main()
