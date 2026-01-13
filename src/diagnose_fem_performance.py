#!/usr/bin/env python3
"""
FEM Performance Diagnostic
==========================
Run this to see where time is being spent.
"""

import numpy as np
import time

# Check which version you have
print("=" * 60)
print("FEM PERFORMANCE DIAGNOSTIC")
print("=" * 60)

from fem_solver import FEMForwardSolver, FEMNonlinearInverseSolver

# Check if optimizations are present
print("\n1. Checking for optimizations...")
forward = FEMForwardSolver(resolution=0.05, verbose=True)

has_lu = hasattr(forward, '_K_lu')
has_tree = hasattr(forward, '_centroid_tree')
has_node_elements = hasattr(forward, '_node_to_elements')

print(f"   _K_lu (LU factorization cache): {'YES ✓' if has_lu else 'NO ✗ - MISSING!'}")
print(f"   _centroid_tree (KDTree):        {'YES ✓' if has_tree else 'NO ✗ - MISSING!'}")
print(f"   _node_to_elements (adjacency):  {'YES ✓' if has_node_elements else 'NO ✗ - MISSING!'}")

if not (has_lu and has_tree and has_node_elements):
    print("\n*** YOU NEED TO UPDATE TO v7.33 OR LATER ***")
    print("*** Download inverse_source_v7_33.zip and reinstall ***")

# Test sources
sources = [((0.5, 0.0), 1.0), ((-0.5, 0.0), -1.0)]

# Time individual components
print("\n2. Timing individual forward solve components...")

# Time _build_rhs
t0 = time.time()
for _ in range(100):
    f = forward._build_rhs(sources)
t_rhs = (time.time() - t0) / 100 * 1000
print(f"   _build_rhs (cell lookup + barycentric): {t_rhs:.3f} ms per call")

# Time LU solve
f = forward._build_rhs(sources)
f = forward._project_nullspace(f)
t0 = time.time()
for _ in range(100):
    u = forward._K_lu.solve(f)
t_solve = (time.time() - t0) / 100 * 1000
print(f"   _K_lu.solve (back-substitution):        {t_solve:.3f} ms per call")

# Time full forward solve
t0 = time.time()
for _ in range(100):
    u = forward.solve(sources)
t_full = (time.time() - t0) / 100 * 1000
print(f"   forward.solve (total):                  {t_full:.3f} ms per call")

# Time 1000 forward solves (simulating optimizer)
print("\n3. Timing 1000 forward solves (like optimizer would do)...")
t0 = time.time()
for i in range(1000):
    # Vary source positions like optimizer would
    offset = 0.1 * np.sin(i * 0.01)
    test_sources = [((0.5 + offset, offset), 1.0), ((-0.5 - offset, -offset), -1.0)]
    u = forward.solve(test_sources)
t_1000 = time.time() - t0
print(f"   1000 forward solves: {t_1000:.2f}s ({t_1000*1000/1000:.2f} ms each)")

# Expected vs actual for nonlinear inverse
print("\n4. Quick nonlinear inverse test (2 sources, 3 restarts)...")
inverse = FEMNonlinearInverseSolver(n_sources=2, resolution=0.05, verbose=False)
u_measured = forward.solve(sources)
inverse.set_measured_data(u_measured)

t0 = time.time()
result = inverse.solve(method='L-BFGS-B', n_restarts=3, maxiter=100)
t_inverse = time.time() - t0

print(f"   Time: {t_inverse:.2f}s")
print(f"   Objective evaluations: {len(inverse.history)}")
print(f"   Time per evaluation: {t_inverse/len(inverse.history)*1000:.2f} ms")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Forward solve should be ~{t_full:.1f} ms")
print(f"For 1000 evaluations, expect ~{t_full * 1000 / 1000:.1f} seconds")
print(f"Actual 1000 forward solves took: {t_1000:.1f} seconds")

if t_full > 10:
    print("\n*** FORWARD SOLVE IS SLOW (>10ms) ***")
    print("Check if LU factorization and cell lookup optimizations are active.")
elif t_1000 > 20:
    print("\n*** 1000 SOLVES SLOWER THAN EXPECTED ***")
    print("Something else may be consuming time.")
else:
    print("\n*** PERFORMANCE LOOKS GOOD ***")
