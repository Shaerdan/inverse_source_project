#!/usr/bin/env python3
"""
Minimal test to verify existing nonlinear solvers still work.
This tests the KNOWN WORKING implementation before we touch IPOPT.
"""

import numpy as np
from analytical_solver import AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver
from scipy.optimize import linear_sum_assignment

# Create simple 4-source test
n_sources = 4
n_boundary = 100

# Well-separated sources at r=0.5-0.7
np.random.seed(42)
sources_true = []
angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
for i, theta in enumerate(angles):
    r = 0.5 + 0.2 * np.random.rand()
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    intensity = 1.0 if i % 2 == 0 else -1.0
    sources_true.append(((x, y), intensity))

# Enforce sum = 0
total = sum(s[1] for s in sources_true)
sources_true[-1] = (sources_true[-1][0], sources_true[-1][1] - total)

print("True sources:")
for i, ((x, y), q) in enumerate(sources_true):
    r = np.sqrt(x**2 + y**2)
    print(f"  {i+1}: ({x:+.4f}, {y:+.4f}), r={r:.3f}, q={q:+.4f}")

total_q = sum(s[1] for s in sources_true)
print(f"Sum of intensities: {total_q}")

# Generate measurement
forward = AnalyticalForwardSolver(n_boundary)
u_measured = forward.solve(sources_true)
print(f"\nu_measured range: [{u_measured.min():.4f}, {u_measured.max():.4f}]")

# Test existing solver with L-BFGS-B
print("\n" + "="*60)
print("Testing AnalyticalNonlinearInverseSolver with L-BFGS-B")
print("="*60)

solver = AnalyticalNonlinearInverseSolver(n_sources=n_sources, n_boundary=n_boundary)
solver.set_measured_data(u_measured)

result = solver.solve(method='L-BFGS-B', n_restarts=10, maxiter=2000)

print("\nRecovered sources:")
for s in result.sources:
    r = np.sqrt(s.x**2 + s.y**2)
    print(f"  ({s.x:+.4f}, {s.y:+.4f}), r={r:.3f}, q={s.intensity:+.4f}")

# Compute error
n = len(sources_true)
cost = np.zeros((n, n))
for i, ((tx, ty), _) in enumerate(sources_true):
    for j, s in enumerate(result.sources):
        cost[i, j] = np.sqrt((tx - s.x)**2 + (ty - s.y)**2)

row_ind, col_ind = linear_sum_assignment(cost)
pos_errors = cost[row_ind, col_ind]

print(f"\nMean position error: {pos_errors.mean():.2e}")
print(f"Residual: {result.residual:.2e}")
print(f"Success: {result.success}")

# Test with differential_evolution
print("\n" + "="*60)
print("Testing AnalyticalNonlinearInverseSolver with differential_evolution")
print("="*60)

solver2 = AnalyticalNonlinearInverseSolver(n_sources=n_sources, n_boundary=n_boundary)
solver2.set_measured_data(u_measured)

result2 = solver2.solve(method='differential_evolution', maxiter=2000)

print("\nRecovered sources:")
for s in result2.sources:
    r = np.sqrt(s.x**2 + s.y**2)
    print(f"  ({s.x:+.4f}, {s.y:+.4f}), r={r:.3f}, q={s.intensity:+.4f}")

cost2 = np.zeros((n, n))
for i, ((tx, ty), _) in enumerate(sources_true):
    for j, s in enumerate(result2.sources):
        cost2[i, j] = np.sqrt((tx - s.x)**2 + (ty - s.y)**2)

row_ind2, col_ind2 = linear_sum_assignment(cost2)
pos_errors2 = cost2[row_ind2, col_ind2]

print(f"\nMean position error: {pos_errors2.mean():.2e}")
print(f"Residual: {result2.residual:.2e}")
print(f"Success: {result2.success}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"L-BFGS-B position error: {pos_errors.mean():.2e}")
print(f"DE position error: {pos_errors2.mean():.2e}")

if pos_errors.mean() < 1e-4 or pos_errors2.mean() < 1e-4:
    print("\n✓ At least one existing solver works!")
else:
    print("\n✗ PROBLEM: Existing solvers also broken!")
