#!/usr/bin/env python3
"""
Ellipse-specific diagnostic script.

The main diagnostic showed that ellipse fails BOTH optimizers despite
all other tests passing. This script investigates why.

Hypotheses:
1. Source generation is doing something unexpected
2. Optimization landscape has many local minima for ellipse
3. Ellipse geometry causes numerical issues
4. Initial guess strategy fails for elongated domains

Usage:
    cd src/
    python diagnose_ellipse.py
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from comparison import create_domain_sources, get_conformal_map
from conformal_solver import (ConformalForwardSolver, ConformalNonlinearInverseSolver,
                               EllipseMap)
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def compute_position_rmse(sources_true, sources_rec):
    """Compute position RMSE with optimal matching."""
    pos_true = np.array([[s[0][0], s[0][1]] for s in sources_true])
    pos_rec = np.array([[s[0][0], s[0][1]] for s in sources_rec])
    
    cost = cdist(pos_true, pos_rec)
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_dist = cost[row_ind, col_ind]
    
    return np.sqrt(np.mean(matched_dist**2))


def test_source_generation():
    """
    TEST 1: What sources does create_domain_sources return for ellipse?
    """
    print("="*60)
    print("TEST 1: Source Generation")
    print("="*60)
    
    # Check what create_domain_sources returns
    sources_default = create_domain_sources('ellipse')
    print(f"\ncreate_domain_sources('ellipse') returns:")
    for i, ((x, y), q) in enumerate(sources_default):
        print(f"  {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
    
    # Check with explicit params
    sources_explicit = create_domain_sources('ellipse', {'a': 2.0, 'b': 1.0})
    print(f"\ncreate_domain_sources('ellipse', {{'a': 2.0, 'b': 1.0}}) returns:")
    for i, ((x, y), q) in enumerate(sources_explicit):
        print(f"  {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
    
    # Check if sources are inside ellipse
    a, b = 2.0, 1.0
    print(f"\nAre sources inside ellipse (a={a}, b={b})?")
    for i, ((x, y), q) in enumerate(sources_default):
        inside = (x/a)**2 + (y/b)**2 < 1
        param = (x/a)**2 + (y/b)**2
        print(f"  {i+1}: ({x:.4f}, {y:.4f}) -> (x/a)²+(y/b)² = {param:.4f}, inside={inside}")


def test_simple_sources():
    """
    TEST 2: Try with very simple, well-separated sources
    """
    print("\n" + "="*60)
    print("TEST 2: Simple Sources")
    print("="*60)
    
    a, b = 2.0, 1.0
    cmap = EllipseMap(a=a, b=b)
    
    # Very simple sources: on axes, well inside ellipse
    simple_sources = [
        ((0.8, 0.0), 1.0),
        ((-0.8, 0.0), -1.0),
    ]
    
    print(f"\nSimple 2-source test:")
    print(f"  Sources: {simple_sources}")
    
    # Generate data
    forward = ConformalForwardSolver(cmap, n_boundary=100)
    u_data = forward.solve(simple_sources)
    print(f"  Data ||u|| = {np.linalg.norm(u_data):.6f}")
    
    # Solve
    inverse = ConformalNonlinearInverseSolver(cmap, n_sources=2, n_boundary=100)
    
    print(f"\n  L-BFGS-B (n_restarts=10):")
    sources_rec, residual = inverse.solve(u_data, method='L-BFGS-B', n_restarts=10, seed=42)
    rmse = compute_position_rmse(simple_sources, sources_rec)
    print(f"    Recovered: {sources_rec}")
    print(f"    Residual: {residual:.6e}, RMSE: {rmse:.6f}")
    
    print(f"\n  differential_evolution:")
    sources_rec, residual = inverse.solve(u_data, method='differential_evolution', seed=42)
    rmse = compute_position_rmse(simple_sources, sources_rec)
    print(f"    Recovered: {sources_rec}")
    print(f"    Residual: {residual:.6e}, RMSE: {rmse:.6f}")
    
    # 4 sources on axes
    print(f"\n4-source test (on axes):")
    four_sources = [
        ((1.0, 0.0), 1.0),
        ((-1.0, 0.0), -1.0),
        ((0.0, 0.5), 1.0),
        ((0.0, -0.5), -1.0),
    ]
    print(f"  Sources: {four_sources}")
    
    u_data = forward.solve(four_sources)
    inverse = ConformalNonlinearInverseSolver(cmap, n_sources=4, n_boundary=100)
    
    print(f"\n  L-BFGS-B (n_restarts=10):")
    sources_rec, residual = inverse.solve(u_data, method='L-BFGS-B', n_restarts=10, seed=42)
    rmse = compute_position_rmse(four_sources, sources_rec)
    print(f"    Residual: {residual:.6e}, RMSE: {rmse:.6f}")
    for i, ((x, y), q) in enumerate(sources_rec):
        print(f"      {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")
    
    print(f"\n  differential_evolution:")
    sources_rec, residual = inverse.solve(u_data, method='differential_evolution', seed=42)
    rmse = compute_position_rmse(four_sources, sources_rec)
    print(f"    Residual: {residual:.6e}, RMSE: {rmse:.6f}")
    for i, ((x, y), q) in enumerate(sources_rec):
        print(f"      {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}")


def test_ellipse_map_details():
    """
    TEST 3: Check EllipseMap behavior in detail
    """
    print("\n" + "="*60)
    print("TEST 3: EllipseMap Details")
    print("="*60)
    
    a, b = 2.0, 1.0
    cmap = EllipseMap(a=a, b=b)
    
    print(f"\nEllipseMap parameters:")
    print(f"  a = {cmap.a}, b = {cmap.b}")
    print(f"  c (focal distance) = {cmap.c}")
    print(f"  R (Joukowsky param) = {cmap.R}")
    
    # Test some interior points
    test_points = [
        0.0 + 0.0j,  # center
        1.0 + 0.0j,  # on x-axis
        0.0 + 0.5j,  # on y-axis
        0.8 + 0.3j,  # interior
    ]
    
    print(f"\nInterior point mapping:")
    print(f"  {'z':<20} {'w = to_disk(z)':<25} {'|w|':<10}")
    print("  " + "-"*55)
    for z in test_points:
        w = cmap.to_disk(z)
        print(f"  {z!s:<20} {w!s:<25} {abs(w):.6f}")
    
    # Check boundary mapping more carefully
    print(f"\nBoundary mapping (should give |w|=1):")
    theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
    z_bdy = a * np.cos(theta) + 1j * b * np.sin(theta)
    w_bdy = cmap.to_disk(z_bdy)
    
    for i in range(len(theta)):
        print(f"  θ={theta[i]:.2f}: z={z_bdy[i]:.4f}, w={w_bdy[i]:.4f}, |w|={abs(w_bdy[i]):.6f}")


def test_objective_landscape():
    """
    TEST 4: Examine objective function landscape around truth
    """
    print("\n" + "="*60)
    print("TEST 4: Objective Landscape")
    print("="*60)
    
    a, b = 2.0, 1.0
    cmap = EllipseMap(a=a, b=b)
    
    # Simple 2-source case
    sources = [
        ((0.8, 0.0), 1.0),
        ((-0.8, 0.0), -1.0),
    ]
    
    forward = ConformalForwardSolver(cmap, n_boundary=100)
    u_data = forward.solve(sources)
    
    inverse = ConformalNonlinearInverseSolver(cmap, n_sources=2, n_boundary=100)
    
    # True params: [x1, y1, x2, y2, q1, q2]
    true_params = np.array([0.8, 0.0, -0.8, 0.0, 1.0, -1.0])
    
    obj_true = inverse._objective(true_params, u_data)
    print(f"\nObjective at truth: {obj_true:.10f}")
    
    # Scan along x1 direction
    print(f"\nScanning x1 around truth (x1_true = 0.8):")
    print(f"  {'x1':<10} {'Objective':<15}")
    print("  " + "-"*25)
    for dx in [-0.5, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.5]:
        params = true_params.copy()
        params[0] = 0.8 + dx
        obj = inverse._objective(params, u_data)
        marker = " <-- TRUE" if dx == 0 else ""
        print(f"  {params[0]:<10.4f} {obj:<15.6e}{marker}")
    
    # Scan along y1 direction
    print(f"\nScanning y1 around truth (y1_true = 0.0):")
    print(f"  {'y1':<10} {'Objective':<15}")
    print("  " + "-"*25)
    for dy in [-0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3]:
        params = true_params.copy()
        params[1] = 0.0 + dy
        obj = inverse._objective(params, u_data)
        marker = " <-- TRUE" if dy == 0 else ""
        print(f"  {params[1]:<10.4f} {obj:<15.6e}{marker}")
    
    # Random perturbations
    print(f"\nRandom perturbations from truth:")
    np.random.seed(42)
    for i in range(10):
        perturb = 0.1 * np.random.randn(6)
        params = true_params + perturb
        obj = inverse._objective(params, u_data)
        print(f"  Perturbation {i+1}: obj = {obj:.6e}")


def test_initial_guess():
    """
    TEST 5: Check initial guess generation
    """
    print("\n" + "="*60)
    print("TEST 5: Initial Guess Generation")
    print("="*60)
    
    a, b = 2.0, 1.0
    cmap = EllipseMap(a=a, b=b)
    
    inverse = ConformalNonlinearInverseSolver(cmap, n_sources=4, n_boundary=100)
    
    # Get domain bounds
    boundary = cmap.boundary_physical(100)
    x_min, x_max = np.real(boundary).min(), np.real(boundary).max()
    y_min, y_max = np.imag(boundary).min(), np.imag(boundary).max()
    
    print(f"\nDomain bounds from boundary:")
    print(f"  x: [{x_min:.4f}, {x_max:.4f}]")
    print(f"  y: [{y_min:.4f}, {y_max:.4f}]")
    
    # Check what bounds the solver uses
    n = 4
    x_margin = 0.05 * (x_max - x_min)
    y_margin = 0.05 * (y_max - y_min)
    solver_x_range = (x_min + x_margin, x_max - x_margin)
    solver_y_range = (y_min + y_margin, y_max - y_margin)
    
    print(f"\nSolver bounds (after 5% margin):")
    print(f"  x: [{solver_x_range[0]:.4f}, {solver_x_range[1]:.4f}]")
    print(f"  y: [{solver_y_range[0]:.4f}, {solver_y_range[1]:.4f}]")
    
    # Generate some initial guesses and check if inside ellipse
    print(f"\nGenerated initial guesses (10 trials):")
    np.random.seed(42)
    for trial in range(10):
        x = np.random.uniform(solver_x_range[0], solver_x_range[1])
        y = np.random.uniform(solver_y_range[0], solver_y_range[1])
        z = complex(x, y)
        inside = cmap.is_inside(z)
        param = (x/a)**2 + (y/b)**2
        status = "INSIDE" if inside else "OUTSIDE"
        print(f"  Trial {trial+1}: ({x:.4f}, {y:.4f}), (x/a)²+(y/b)²={param:.4f}, {status}")


def test_compare_disk_vs_ellipse():
    """
    TEST 6: Compare disk and ellipse with equivalent setups
    """
    print("\n" + "="*60)
    print("TEST 6: Disk vs Ellipse Comparison")
    print("="*60)
    
    # Use identical source positions (scaled appropriately)
    # Disk: unit disk
    # Ellipse: a=2, b=1 (so x is scaled by 2, y by 1)
    
    disk_sources = [
        ((0.5, 0.0), 1.0),
        ((-0.5, 0.0), -1.0),
    ]
    
    # Equivalent ellipse sources (same relative position)
    ellipse_sources = [
        ((1.0, 0.0), 1.0),   # 0.5 * 2 = 1.0
        ((-1.0, 0.0), -1.0),
    ]
    
    print(f"\nDisk sources: {disk_sources}")
    print(f"Ellipse sources (scaled): {ellipse_sources}")
    
    # Test disk
    from conformal_solver import DiskMap
    disk_map = DiskMap(radius=1.0)
    forward_disk = ConformalForwardSolver(disk_map, n_boundary=100)
    u_disk = forward_disk.solve(disk_sources)
    
    inverse_disk = ConformalNonlinearInverseSolver(disk_map, n_sources=2, n_boundary=100)
    rec_disk, res_disk = inverse_disk.solve(u_disk, method='L-BFGS-B', n_restarts=5, seed=42)
    rmse_disk = compute_position_rmse(disk_sources, rec_disk)
    
    print(f"\nDisk result:")
    print(f"  Residual: {res_disk:.6e}, RMSE: {rmse_disk:.6f}")
    
    # Test ellipse
    ellipse_map = EllipseMap(a=2.0, b=1.0)
    forward_ellipse = ConformalForwardSolver(ellipse_map, n_boundary=100)
    u_ellipse = forward_ellipse.solve(ellipse_sources)
    
    inverse_ellipse = ConformalNonlinearInverseSolver(ellipse_map, n_sources=2, n_boundary=100)
    rec_ellipse, res_ellipse = inverse_ellipse.solve(u_ellipse, method='L-BFGS-B', n_restarts=5, seed=42)
    rmse_ellipse = compute_position_rmse(ellipse_sources, rec_ellipse)
    
    print(f"\nEllipse result:")
    print(f"  Residual: {res_ellipse:.6e}, RMSE: {rmse_ellipse:.6f}")
    print(f"  Recovered: {rec_ellipse}")


def main():
    test_source_generation()
    test_ellipse_map_details()
    test_simple_sources()
    test_objective_landscape()
    test_initial_guess()
    test_compare_disk_vs_ellipse()
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
