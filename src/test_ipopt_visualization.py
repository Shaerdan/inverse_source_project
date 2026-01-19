#!/usr/bin/env python3
"""
Test real-time visualization during IPOPT optimization.

Run locally where cyipopt is installed:
    conda install -c conda-forge cyipopt
    python test_ipopt_visualization.py
"""

import numpy as np
import sys

# Check dependencies
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Use interactive backend
    import matplotlib.pyplot as plt
    print("✓ matplotlib available")
except ImportError:
    print("✗ matplotlib not found!")
    sys.exit(1)

try:
    import cyipopt
    print(f"✓ cyipopt available (version: {cyipopt.__version__})")
except ImportError:
    print("✗ cyipopt not found!")
    print("Install via: conda install -c conda-forge cyipopt")
    sys.exit(1)

from ipopt_solver import IPOPTNonlinearInverseSolver
from analytical_solver import AnalyticalForwardSolver


def create_test_sources(n_sources=4, seed=42):
    """Create well-separated test sources."""
    np.random.seed(seed)
    
    sources = []
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += 0.1 * np.random.randn(n_sources)
    
    for i, theta in enumerate(angles):
        r = 0.5 + 0.2 * np.random.rand()
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        intensity = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), intensity))
    
    # Enforce sum = 0
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    
    return sources


def main():
    print("\n" + "="*60)
    print("IPOPT Solver with Real-Time Visualization")
    print("="*60)
    
    # Test configuration
    n_sources = 4
    n_boundary = 100
    
    # Create test sources
    sources_true = create_test_sources(n_sources)
    
    print(f"\nTest configuration:")
    print(f"  Sources: {n_sources}")
    print(f"  Boundary points: {n_boundary}")
    
    print("\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources_true):
        r = np.sqrt(x**2 + y**2)
        print(f"  {i+1}: ({x:+.4f}, {y:+.4f}), r={r:.3f}, q={q:+.4f}")
    
    # Verify sum = 0
    total_q = sum(s[1] for s in sources_true)
    print(f"\nSum of intensities: {total_q:.6f}")
    
    # Generate measurement data
    print("\nGenerating measurement data...")
    forward = AnalyticalForwardSolver(n_boundary)
    u_measured = forward.solve(sources_true)
    print(f"  u_measured range: [{u_measured.min():.4f}, {u_measured.max():.4f}]")
    print(f"  u_measured mean: {u_measured.mean():.6f} (should be ~0)")
    
    # Create solver
    solver = IPOPTNonlinearInverseSolver(
        n_sources=n_sources, 
        n_boundary=n_boundary
    )
    solver.set_measured_data(u_measured)
    
    # Solve WITH visualization
    print("\nStarting optimization with visualization...")
    print("(Watch the plots update in real-time)\n")
    
    result = solver.solve_with_visualization(
        sources_true=sources_true,
        update_interval=1,  # Update every iteration
        max_iter=2000,
        tol=1e-10,
        verbose=True,
        print_level=5  # Show IPOPT output to verify convergence
    )
    
    # Final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    print("\nRecovered sources:")
    for i, s in enumerate(result.sources):
        r = np.sqrt(s.x**2 + s.y**2)
        print(f"  {i+1}: ({s.x:+.4f}, {s.y:+.4f}), r={r:.3f}, q={s.intensity:+.4f}")
    
    # Compute position error
    from scipy.optimize import linear_sum_assignment
    
    n = len(sources_true)
    cost = np.zeros((n, n))
    for i, ((tx, ty), _) in enumerate(sources_true):
        for j, s in enumerate(result.sources):
            cost[i, j] = np.sqrt((tx - s.x)**2 + (ty - s.y)**2)
    
    row_ind, col_ind = linear_sum_assignment(cost)
    pos_errors = cost[row_ind, col_ind]
    
    print(f"\nMean position error: {pos_errors.mean():.2e}")
    print(f"Residual (RMS):      {result.residual:.2e}")
    print(f"Iterations:          {result.iterations}")
    print(f"Success:             {result.success}")
    
    if pos_errors.mean() < 1e-5:
        print("\n✓ SUCCESS: Mean position error < 1e-5")
    elif pos_errors.mean() < 1e-3:
        print("\n~ ACCEPTABLE: Mean position error < 1e-3")
    else:
        print("\n✗ NEEDS WORK: Mean position error > 1e-3")


if __name__ == "__main__":
    main()
