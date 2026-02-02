#!/usr/bin/env python3
"""
Diagnostic: Compare analytical vs FEM disk solvers on the SAME problem.
Both should produce identical results since they're solving the same optimization.

Usage:
    python debug_comparison.py [n_sources] [symmetric]
    
Examples:
    python debug_comparison.py 8 0      # 8 sources, non-symmetric (default)
    python debug_comparison.py 4 1      # 4 sources, symmetric
    python debug_comparison.py 6        # 6 sources, non-symmetric
"""

import numpy as np
import sys
from scipy.optimize import linear_sum_assignment

# Parse command line arguments
n_sources = int(sys.argv[1]) if len(sys.argv) > 1 else 8
symmetric = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False

def compute_rmse(sources_true, sources_rec):
    n = len(sources_true)
    cost = np.zeros((n, n))
    for i, (pos_t, _) in enumerate(sources_true):
        for j in range(len(sources_rec)):
            if hasattr(sources_rec[j], 'x'):
                pos_r = (sources_rec[j].x, sources_rec[j].y)
            elif hasattr(sources_rec[j], 'position'):
                pos_r = sources_rec[j].position
            else:
                pos_r = sources_rec[j][0]
            cost[i, j] = (pos_t[0]-pos_r[0])**2 + (pos_t[1]-pos_r[1])**2
    row_ind, col_ind = linear_sum_assignment(cost)
    return np.sqrt(cost[row_ind, col_ind].mean())


def create_sources_disk(n_sources, r_range=(0.7, 0.9), symmetric=False, seed=42):
    np.random.seed(seed)
    sources = []
    
    if symmetric:
        angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    else:
        angles = []
        min_sep = np.pi / (n_sources + 1)
        for i in range(n_sources):
            for _ in range(100):
                theta = np.random.uniform(0, 2*np.pi)
                if all(min(abs(theta - a), 2*np.pi - abs(theta - a)) > min_sep for a in angles):
                    angles.append(theta)
                    break
            else:
                angles.append(np.random.uniform(0, 2*np.pi))
        angles = np.array(angles)
    
    for i, theta in enumerate(angles):
        r = np.random.uniform(r_range[0], r_range[1])
        x, y = r * np.cos(theta), r * np.sin(theta)
        S = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), S))
    
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    return sources


# Create the failing case: n=n_sources, symmetric or not
print("="*70)
print(f"DIAGNOSTIC: n={n_sources} {'symmetric' if symmetric else 'non-symmetric'} disk")
print("="*70)

sources = create_sources_disk(n_sources, r_range=(0.7, 0.9), symmetric=symmetric, seed=42)
print("\nTrue sources:")
for i, ((x, y), q) in enumerate(sources):
    r = np.sqrt(x**2 + y**2)
    theta = np.degrees(np.arctan2(y, x))
    print(f"  {i+1}: r={r:.4f}, θ={theta:7.2f}°, q={q:+.4f}")

# ============ ANALYTICAL SOLVER ============
print("\n" + "-"*70)
print("ANALYTICAL SOLVER")
print("-"*70)

from analytical_solver import AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver

forward_ana = AnalyticalForwardSolver(100)
u_ana = forward_ana.solve(sources)

inverse_ana = AnalyticalNonlinearInverseSolver(n_sources, 100)
inverse_ana.set_measured_data(u_ana)

# Check objective at true params
true_params = []
for (x, y), q in sources:
    true_params.extend([x, y])
for (x, y), q in sources:
    true_params.append(q)
true_params = np.array(true_params)

obj_at_true_ana = inverse_ana._objective_misfit(true_params)
print(f"Objective at true params: {obj_at_true_ana:.2e}")

# Solve
result_ana = inverse_ana.solve(method='SLSQP', n_restarts=5, maxiter=10000)
rmse_ana = compute_rmse(sources, result_ana.sources)
print(f"Final objective: {result_ana.residual:.2e}")
print(f"RMSE: {rmse_ana:.2e}")
print(f"Status: {'PASS' if rmse_ana < 0.01 else 'FAIL'}")

# ============ FEM SOLVER ============
print("\n" + "-"*70)
print("FEM SOLVER")
print("-"*70)

from fem_solver import FEMNonlinearInverseSolver

inverse_fem = FEMNonlinearInverseSolver(n_sources, resolution=0.1, verbose=False, domain_type='disk')
u_fem = inverse_fem.forward.solve_at_boundary(sources)
inverse_fem.set_measured_data(u_fem)

obj_at_true_fem = inverse_fem._objective_misfit(true_params)
print(f"Objective at true params: {obj_at_true_fem:.2e}")

# Compare forward models
print(f"\nForward model comparison:")
print(f"  Analytical boundary range: [{u_ana.min():.4f}, {u_ana.max():.4f}]")
print(f"  FEM boundary range:        [{u_fem.min():.4f}, {u_fem.max():.4f}]")
print(f"  Analytical n_points: {len(u_ana)}")
print(f"  FEM n_points:        {len(u_fem)}")

# Solve
result_fem = inverse_fem.solve(method='SLSQP', n_restarts=5, maxiter=10000)
rmse_fem = compute_rmse(sources, result_fem.sources)
print(f"\nFinal objective: {result_fem.residual:.2e}")
print(f"RMSE: {rmse_fem:.2e}")
print(f"Status: {'PASS' if rmse_fem < 0.01 else 'FAIL'}")

# Show recovered sources if failed
if rmse_fem > 0.01:
    print("\nRecovered sources:")
    for i, s in enumerate(result_fem.sources):
        r = np.sqrt(s.position[0]**2 + s.position[1]**2)
        theta = np.degrees(np.arctan2(s.position[1], s.position[0]))
        print(f"  {i+1}: r={r:.4f}, θ={theta:7.2f}°, q={s.intensity:+.4f}")

# ============ DIRECT COMPARISON ============
print("\n" + "-"*70)
print("DIRECT COMPARISON: Same initial guess")
print("-"*70)

# Use the SAME initial guess for both
init_guess = inverse_ana._get_initial_guess('spread', 0)
print(f"Initial guess (spread, seed=0):")
for i in range(n_sources):
    x, y = init_guess[2*i], init_guess[2*i+1]
    r = np.sqrt(x**2 + y**2)
    print(f"  Source {i+1}: r={r:.3f}")

# Evaluate objective at this initial guess
obj_ana_init = inverse_ana._objective_misfit(init_guess)
obj_fem_init = inverse_fem._objective_misfit(init_guess)
print(f"\nObjective at initial guess:")
print(f"  Analytical: {obj_ana_init:.4e}")
print(f"  FEM:        {obj_fem_init:.4e}")

# Check gradients numerically
print("\n" + "-"*70)
print("GRADIENT CHECK at initial guess")
print("-"*70)

def numerical_gradient(func, x, eps=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (func(x_plus) - func(x_minus)) / (2 * eps)
    return grad

grad_ana = numerical_gradient(inverse_ana._objective_misfit, init_guess)
grad_fem = numerical_gradient(inverse_fem._objective_misfit, init_guess)

print(f"Gradient norm (analytical): {np.linalg.norm(grad_ana):.4e}")
print(f"Gradient norm (FEM):        {np.linalg.norm(grad_fem):.4e}")
print(f"Gradient difference norm:   {np.linalg.norm(grad_ana - grad_fem):.4e}")

# Check constraint values
print("\n" + "-"*70)
print("CONSTRAINT CHECK at initial guess")
print("-"*70)

# Analytical disk constraint
def disk_constraint_ana(params):
    constraints_vals = []
    for i in range(n_sources):
        x_i = params[2*i]
        y_i = params[2*i + 1]
        constraints_vals.append(1.0 - x_i**2 - y_i**2)
    return np.array(constraints_vals)

c_ana = disk_constraint_ana(init_guess)
c_fem = inverse_fem._domain_constraint(init_guess)
print(f"Constraint values (should be > 0 for feasible):")
print(f"  Analytical: {c_ana}")
print(f"  FEM:        {c_fem}")
