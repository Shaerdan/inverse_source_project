#!/usr/bin/env python3
"""
Test: Does finer FEM resolution fix the precision floor?
"""

import numpy as np
from time import time

def create_sources(n_sources, seed=42):
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

from scipy.optimize import linear_sum_assignment

def compute_rmse(sources_true, sources_rec):
    n = len(sources_true)
    cost = np.zeros((n, n))
    for i, (pos_t, _) in enumerate(sources_true):
        for j in range(len(sources_rec)):
            pos_r = sources_rec[j].position
            cost[i, j] = (pos_t[0]-pos_r[0])**2 + (pos_t[1]-pos_r[1])**2
    row_ind, col_ind = linear_sum_assignment(cost)
    return np.sqrt(cost[row_ind, col_ind].mean())

from fem_solver import FEMNonlinearInverseSolver

sources = create_sources(8, seed=42)
true_params = sources_to_params(sources)

print("="*70)
print("FEM RESOLUTION TEST: n=8 non-symmetric disk")
print("="*70)

resolutions = [0.1, 0.05, 0.025]
n_restarts_list = [5, 10, 20]

for resolution in resolutions:
    print(f"\n--- Resolution = {resolution} ---")
    
    inverse = FEMNonlinearInverseSolver(8, resolution=resolution, verbose=False, domain_type='disk')
    u = inverse.forward.solve_at_boundary(sources)
    inverse.set_measured_data(u)
    
    obj_at_true = inverse._objective_misfit(true_params)
    print(f"Objective at TRUE params: {obj_at_true:.2e}")
    print(f"Number of boundary nodes: {len(u)}")
    
    for n_restarts in n_restarts_list:
        t0 = time()
        result = inverse.solve(method='SLSQP', n_restarts=n_restarts, maxiter=10000)
        elapsed = time() - t0
        
        rmse = compute_rmse(sources, result.sources)
        status = "✓ PASS" if rmse < 0.01 else "✗ FAIL"
        print(f"  n_restarts={n_restarts:2d}: RMSE={rmse:.2e}, obj={result.residual:.2e}, time={elapsed:.1f}s {status}")
