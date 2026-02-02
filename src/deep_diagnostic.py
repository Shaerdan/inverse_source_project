#!/usr/bin/env python3
"""
Deep diagnostic: WHY do some solvers fail on n=8 nonsym?

Hypothesis testing:
1. Is the true solution actually at a local minimum of the objective?
2. Are there other local minima with similar objective values?
3. Is the gradient reliable at the true solution?
4. How many restarts are needed to find the true solution?
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linear_sum_assignment

def compute_rmse(sources_true, sources_rec):
    n = len(sources_true)
    cost = np.zeros((n, n))
    for i, (pos_t, _) in enumerate(sources_true):
        for j in range(len(sources_rec)):
            if isinstance(sources_rec[j], tuple):
                pos_r = sources_rec[j][0]
            else:
                pos_r = sources_rec[j][0] if isinstance(sources_rec[j][0], tuple) else (sources_rec[j].x, sources_rec[j].y)
            cost[i, j] = (pos_t[0]-pos_r[0])**2 + (pos_t[1]-pos_r[1])**2
    row_ind, col_ind = linear_sum_assignment(cost)
    return np.sqrt(cost[row_ind, col_ind].mean())

def create_sources(n_sources, domain='disk', seed=42):
    """Create sources for different domains."""
    np.random.seed(seed)
    
    # Non-symmetric angles with minimum separation
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
    
    sources = []
    for i, theta in enumerate(angles):
        if domain == 'disk':
            r = np.random.uniform(0.7, 0.9)
            x, y = r * np.cos(theta), r * np.sin(theta)
        elif domain == 'ellipse':
            a, b = 1.5, 0.8
            s = np.random.uniform(0.6, 0.75)
            x, y = a * s * np.cos(theta), b * s * np.sin(theta)
        elif domain == 'square':
            s = np.random.uniform(0.6, 0.75)
            x, y = s * np.cos(theta), s * np.sin(theta)
            x = np.clip(x, -0.85, 0.85)
            y = np.clip(y, -0.85, 0.85)
        
        S = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), S))
    
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    return sources

def sources_to_params(sources):
    """Convert sources to parameter vector."""
    params = []
    for (x, y), q in sources:
        params.extend([x, y])
    for (x, y), q in sources:
        params.append(q)
    return np.array(params)

print("="*70)
print("DEEP DIAGNOSTIC: Why do solvers fail on n=8 nonsym?")
print("="*70)

# ============================================================
# TEST 1: Check objective landscape for DISK (analytical vs FEM)
# ============================================================
print("\n" + "="*70)
print("TEST 1: DISK DOMAIN - Analytical vs FEM")
print("="*70)

from analytical_solver import AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver
from fem_solver import FEMNonlinearInverseSolver

sources_disk = create_sources(8, 'disk', seed=42)
true_params = sources_to_params(sources_disk)

print("\nTrue sources:")
for i, ((x, y), q) in enumerate(sources_disk):
    print(f"  {i+1}: ({x:+.4f}, {y:+.4f}), q={q:+.4f}")

# Setup solvers
forward_ana = AnalyticalForwardSolver(100)
u_ana = forward_ana.solve(sources_disk)
inverse_ana = AnalyticalNonlinearInverseSolver(8, 100)
inverse_ana.set_measured_data(u_ana)

inverse_fem = FEMNonlinearInverseSolver(8, resolution=0.1, verbose=False, domain_type='disk')
u_fem = inverse_fem.forward.solve_at_boundary(sources_disk)
inverse_fem.set_measured_data(u_fem)

print(f"\nObjective at TRUE solution:")
print(f"  Analytical: {inverse_ana._objective_misfit(true_params):.2e}")
print(f"  FEM:        {inverse_fem._objective_misfit(true_params):.2e}")

# Run many restarts and track all results
print(f"\nRunning 20 restarts for each solver...")

ana_results = []
fem_results = []

for restart in range(20):
    # Analytical
    x0 = inverse_ana._get_initial_guess('spread' if restart < 3 else 'random', restart)
    try:
        n = 8
        def intensity_sum(params):
            return sum(params[2*n + i] for i in range(n))
        def disk_constraint(params):
            return np.array([1.0 - params[2*i]**2 - params[2*i+1]**2 for i in range(n)])
        
        result = minimize(
            inverse_ana._objective_misfit, x0, method='SLSQP',
            bounds=[(-0.95, 0.95)]*(2*n) + [(-5, 5)]*n,
            constraints=[{'type': 'eq', 'fun': intensity_sum},
                        {'type': 'ineq', 'fun': disk_constraint}],
            options={'maxiter': 10000, 'ftol': 1e-14}
        )
        ana_results.append((result.fun, result.x))
    except:
        pass
    
    # FEM
    x0 = inverse_fem._get_initial_guess('spread' if restart < 3 else 'random', restart)
    try:
        result = minimize(
            inverse_fem._objective_misfit, x0, method='SLSQP',
            bounds=[(-0.95, 0.95)]*(2*n) + [(-5, 5)]*n,
            constraints=[{'type': 'eq', 'fun': intensity_sum},
                        {'type': 'ineq', 'fun': disk_constraint}],
            options={'maxiter': 10000, 'ftol': 1e-14}
        )
        fem_results.append((result.fun, result.x))
    except:
        pass

# Analyze results
ana_objectives = sorted([r[0] for r in ana_results])
fem_objectives = sorted([r[0] for r in fem_results])

print(f"\nAnalytical solver - objective values across 20 restarts:")
print(f"  Best 5: {ana_objectives[:5]}")
print(f"  Worst 5: {ana_objectives[-5:]}")

print(f"\nFEM solver - objective values across 20 restarts:")
print(f"  Best 5: {fem_objectives[:5]}")
print(f"  Worst 5: {fem_objectives[-5:]}")

# Check how many found the global minimum
ana_best = min(ana_objectives)
fem_best = min(fem_objectives)
ana_near_best = sum(1 for o in ana_objectives if o < ana_best * 10)
fem_near_best = sum(1 for o in fem_objectives if o < fem_best * 10)

print(f"\nRestarts finding near-optimal solution (within 10x of best):")
print(f"  Analytical: {ana_near_best}/20")
print(f"  FEM:        {fem_near_best}/20")

# ============================================================
# TEST 2: ELLIPSE DOMAIN - Conformal
# ============================================================
print("\n" + "="*70)
print("TEST 2: ELLIPSE DOMAIN - Conformal solver")
print("="*70)

from conformal_solver import EllipseMap, ConformalForwardSolver, ConformalNonlinearInverseSolver

sources_ellipse = create_sources(8, 'ellipse', seed=42)
true_params_ell = sources_to_params(sources_ellipse)

print("\nTrue sources:")
for i, ((x, y), q) in enumerate(sources_ellipse):
    print(f"  {i+1}: ({x:+.4f}, {y:+.4f}), q={q:+.4f}")

a, b = 1.5, 0.8
emap = EllipseMap(a=a, b=b)
forward_conf = ConformalForwardSolver(emap, 100)
u_conf = forward_conf.solve(sources_ellipse)

inverse_conf = ConformalNonlinearInverseSolver(emap, 8, 100)

print(f"\nObjective at TRUE solution:")
print(f"  Conformal: {inverse_conf._objective_misfit(true_params_ell, u_conf):.2e}")

# Check domain constraint at true solution
conf_constraint = inverse_conf._conformal_constraint(true_params_ell)
print(f"\nDomain constraint at true solution (should be > 0):")
print(f"  Values: {conf_constraint}")
print(f"  All positive: {all(conf_constraint > 0)}")

# Run many restarts
print(f"\nRunning 20 restarts...")

conf_results = []
n = 8

def intensity_sum_ell(params):
    return sum(params[2*n + i] for i in range(n))

for restart in range(20):
    np.random.seed(42 + restart)
    x0 = inverse_conf._generate_valid_initial_guess(
        [(-a, a), (-b, b)]*n + [(-5, 5)]*n, 
        restart, 
        'spread' if restart < 3 else 'random'
    )
    
    try:
        result = minimize(
            lambda p: inverse_conf._objective_misfit(p, u_conf),
            x0, method='SLSQP',
            bounds=[(-a, a), (-b, b)]*n + [(-5, 5)]*n,
            constraints=[
                {'type': 'eq', 'fun': intensity_sum_ell},
                {'type': 'ineq', 'fun': inverse_conf._conformal_constraint}
            ],
            options={'maxiter': 10000, 'ftol': 1e-14}
        )
        conf_results.append((result.fun, result.x))
    except Exception as e:
        print(f"  Restart {restart} failed: {e}")

conf_objectives = sorted([r[0] for r in conf_results])
print(f"\nConformal solver - objective values across {len(conf_results)} restarts:")
print(f"  Best 5: {conf_objectives[:5]}")
if len(conf_objectives) > 5:
    print(f"  Worst 5: {conf_objectives[-5:]}")

# Find the best solution and compute RMSE
if conf_results:
    best_idx = np.argmin([r[0] for r in conf_results])
    best_params = conf_results[best_idx][1]
    
    # Convert to sources
    positions = best_params[:2*n].reshape(n, 2)
    intensities = best_params[2*n:]
    intensities = intensities - np.mean(intensities)
    rec_sources = [((positions[k, 0], positions[k, 1]), intensities[k]) for k in range(n)]
    
    rmse = compute_rmse(sources_ellipse, rec_sources)
    print(f"\nBest solution RMSE: {rmse:.2e}")
    print(f"Best objective: {conf_results[best_idx][0]:.2e}")
    
    if rmse > 0.01:
        print("\nRecovered vs True:")
        for i in range(n):
            tx, ty = sources_ellipse[i][0]
            rx, ry = positions[i]
            print(f"  {i+1}: True=({tx:+.3f},{ty:+.3f}) Rec=({rx:+.3f},{ry:+.3f})")

# ============================================================
# TEST 3: Check if problem is fundamentally harder for ellipse
# ============================================================
print("\n" + "="*70)
print("TEST 3: Is the ellipse problem fundamentally harder?")
print("="*70)

# Check condition number of the problem by looking at gradient magnitudes
print("\nGradient analysis at true solution:")

def numerical_gradient(func, x, eps=1e-7):
    grad = np.zeros_like(x)
    f0 = func(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        grad[i] = (func(x_plus) - f0) / eps
    return grad

grad_ana = numerical_gradient(inverse_ana._objective_misfit, true_params)
grad_conf = numerical_gradient(lambda p: inverse_conf._objective_misfit(p, u_conf), true_params_ell)

print(f"  Analytical disk gradient norm: {np.linalg.norm(grad_ana):.2e}")
print(f"  Conformal ellipse gradient norm: {np.linalg.norm(grad_conf):.2e}")

# Check Hessian eigenvalues (approximation)
print("\nHessian analysis (diagonal approximation):")

def approx_hessian_diag(func, x, eps=1e-5):
    """Approximate diagonal of Hessian."""
    n = len(x)
    diag = np.zeros(n)
    f0 = func(x)
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        diag[i] = (func(x_plus) - 2*f0 + func(x_minus)) / eps**2
    return diag

hess_ana = approx_hessian_diag(inverse_ana._objective_misfit, true_params)
hess_conf = approx_hessian_diag(lambda p: inverse_conf._objective_misfit(p, u_conf), true_params_ell)

print(f"  Analytical disk - Hessian diag range: [{hess_ana.min():.2e}, {hess_ana.max():.2e}]")
print(f"  Conformal ellipse - Hessian diag range: [{hess_conf.min():.2e}, {hess_conf.max():.2e}]")
print(f"  Analytical disk - condition (max/min positive): {hess_ana[hess_ana>0].max()/hess_ana[hess_ana>0].min():.2e}")
print(f"  Conformal ellipse - condition (max/min positive): {hess_conf[hess_conf>0].max()/hess_conf[hess_conf>0].min():.2e}")
