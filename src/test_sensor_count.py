#!/usr/bin/env python3
"""
Test: Effect of number of sensors on inverse problem accuracy.

MATLAB likely used many more sensors. Let's test 100 vs 500 vs 1000.
More sensors = better conditioned problem = easier optimization.
"""

import numpy as np
from time import time
from scipy.optimize import linear_sum_assignment

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


def create_sources(n_sources, seed=42):
    """Create non-symmetric sources."""
    np.random.seed(seed)
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
        r = np.random.uniform(0.7, 0.9)
        x, y = r * np.cos(theta), r * np.sin(theta)
        S = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), S))
    
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    return sources


def sources_to_params(sources):
    params = []
    for (x, y), q in sources:
        params.extend([x, y])
    for (x, y), q in sources:
        params.append(q)
    return np.array(params)


print("="*70)
print("SENSOR COUNT TEST: n=8 non-symmetric disk")
print("="*70)

sources = create_sources(8, seed=42)
true_params = sources_to_params(sources)

print("\nTrue sources:")
for i, ((x, y), q) in enumerate(sources):
    r = np.sqrt(x**2 + y**2)
    print(f"  {i+1}: r={r:.4f}, q={q:+.4f}")

# ============================================================
# TEST ANALYTICAL SOLVER with different sensor counts
# ============================================================
print("\n" + "="*70)
print("ANALYTICAL SOLVER - Varying sensor count")
print("="*70)

from analytical_solver import AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver

sensor_counts = [100, 500, 1000]
n_restarts = 5

for n_sensors in sensor_counts:
    print(f"\n--- {n_sensors} sensors ---")
    
    forward = AnalyticalForwardSolver(n_sensors)
    u = forward.solve(sources)
    
    inverse = AnalyticalNonlinearInverseSolver(8, n_sensors)
    inverse.set_measured_data(u)
    
    obj_at_true = inverse._objective_misfit(true_params)
    print(f"Objective at TRUE params: {obj_at_true:.2e}")
    
    t0 = time()
    result = inverse.solve(method='SLSQP', n_restarts=n_restarts, maxiter=10000)
    elapsed = time() - t0
    
    rmse = compute_rmse(sources, result.sources)
    status = "✓ PASS" if rmse < 0.01 else "✗ FAIL"
    print(f"Result: RMSE={rmse:.2e}, obj={result.residual:.2e}, time={elapsed:.1f}s {status}")


# ============================================================
# TEST FEM SOLVER - n_sensors as parameter (sensors embedded in mesh)
# ============================================================
print("\n" + "="*70)
print("FEM SOLVER - Varying sensor count (embedded in mesh)")
print("="*70)

from fem_solver import FEMNonlinearInverseSolver

for n_sensors in sensor_counts:
    print(f"\n--- {n_sensors} sensors ---")
    
    # Create FEM solver with specified n_sensors
    inverse = FEMNonlinearInverseSolver(8, resolution=0.05, n_sensors=n_sensors,
                                         verbose=False, domain_type='disk')
    
    # Generate measurements at sensor locations (exact, no interpolation)
    u = inverse.forward.solve_at_sensors(sources)
    inverse.set_measured_data(u)
    
    obj_at_true = inverse._objective_misfit(true_params)
    print(f"Objective at TRUE params: {obj_at_true:.2e}")
    print(f"Number of sensors: {inverse.forward.n_sensors}")
    
    t0 = time()
    result = inverse.solve(method='SLSQP', n_restarts=n_restarts, maxiter=10000)
    elapsed = time() - t0
    
    rmse = compute_rmse(sources, result.sources)
    status = "✓ PASS" if rmse < 0.01 else "✗ FAIL"
    print(f"Result: RMSE={rmse:.2e}, obj={result.residual:.2e}, time={elapsed:.1f}s {status}")


# ============================================================
# TEST CONFORMAL SOLVER with different sensor counts
# ============================================================
print("\n" + "="*70)
print("CONFORMAL SOLVER (Ellipse) - Varying sensor count")
print("="*70)

from conformal_solver import EllipseMap, ConformalForwardSolver, ConformalNonlinearInverseSolver

# Create ellipse sources
def create_sources_ellipse(n_sources, a=1.5, b=0.8, seed=42):
    np.random.seed(seed)
    angles = []
    min_sep = np.pi / (n_sources + 1)
    for i in range(n_sources):
        for _ in range(100):
            theta = np.random.uniform(0, 2*np.pi)
            if all(min(abs(theta - aa), 2*np.pi - abs(theta - aa)) > min_sep for aa in angles):
                angles.append(theta)
                break
        else:
            angles.append(np.random.uniform(0, 2*np.pi))
    
    sources = []
    for i, theta in enumerate(angles):
        s = np.random.uniform(0.6, 0.75)
        x, y = a * s * np.cos(theta), b * s * np.sin(theta)
        S = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), S))
    
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    return sources

a, b = 1.5, 0.8
sources_ell = create_sources_ellipse(8, a=a, b=b, seed=42)

print("\nTrue sources (ellipse):")
for i, ((x, y), q) in enumerate(sources_ell):
    print(f"  {i+1}: ({x:+.4f}, {y:+.4f}), q={q:+.4f}")

for n_sensors in sensor_counts:
    print(f"\n--- {n_sensors} sensors ---")
    
    emap = EllipseMap(a=a, b=b)
    forward = ConformalForwardSolver(emap, n_sensors)
    u = forward.solve(sources_ell)
    
    inverse = ConformalNonlinearInverseSolver(emap, 8, n_sensors)
    
    true_params_ell = sources_to_params(sources_ell)
    obj_at_true = inverse._objective_misfit(true_params_ell, u)
    print(f"Objective at TRUE params: {obj_at_true:.2e}")
    
    t0 = time()
    sources_rec, residual = inverse.solve(u, method='SLSQP', n_restarts=n_restarts)
    elapsed = time() - t0
    
    rmse = compute_rmse(sources_ell, sources_rec)
    status = "✓ PASS" if rmse < 0.01 else "✗ FAIL"
    print(f"Result: RMSE={rmse:.2e}, obj={residual:.2e}, time={elapsed:.1f}s {status}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
Key findings:
1. ANALYTICAL solver: sensor count is a direct parameter
2. FEM solver: sensor count is controlled by mesh resolution
3. CONFORMAL solver: sensor count is a direct parameter

More sensors = better conditioned inverse problem = easier optimization

For FEM, finer resolution means:
- More boundary nodes (more sensors)
- More accurate forward model
- But slower computation
""")
