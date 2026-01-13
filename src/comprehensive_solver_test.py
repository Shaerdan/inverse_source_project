#!/usr/bin/env python3
"""
Comprehensive test of all solvers across all domains.

Run this after the ellipse MFS fix to identify any remaining issues.

Usage:
    cd src/
    python comprehensive_solver_test.py

This will:
1. Run all solvers on all domains
2. Flag any solver with RMSE > 0.05 as FAILED
3. Generate a summary table for easy diagnosis
"""

import numpy as np
import warnings
import sys
from time import time
warnings.filterwarnings('ignore')

from comparison import (create_domain_sources, get_conformal_map, 
                        compute_metrics)
from conformal_solver import ConformalForwardSolver, ConformalNonlinearInverseSolver
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


# Thresholds for pass/fail
RMSE_THRESHOLD = 0.05  # Position RMSE above this is a failure
RMSE_EXCELLENT = 0.01  # RMSE below this is excellent


def compute_position_rmse(sources_true, sources_rec):
    """Compute position RMSE with optimal matching."""
    pos_true = np.array([[s[0][0], s[0][1]] for s in sources_true])
    pos_rec = np.array([[s[0][0], s[0][1]] for s in sources_rec])
    
    cost = cdist(pos_true, pos_rec)
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_dist = cost[row_ind, col_ind]
    
    return np.sqrt(np.mean(matched_dist**2))


def test_conformal_nonlinear(domain, seed=42):
    """Test conformal nonlinear solver on a domain."""
    sources = create_domain_sources(domain)
    cmap = get_conformal_map(domain)
    n_sources = len(sources)
    
    # Generate data
    forward = ConformalForwardSolver(cmap, n_boundary=100)
    u_data = forward.solve(sources)
    
    results = {}
    
    # L-BFGS-B
    try:
        inverse = ConformalNonlinearInverseSolver(cmap, n_sources=n_sources, n_boundary=100)
        t0 = time()
        sources_rec, residual = inverse.solve(u_data, method='L-BFGS-B', n_restarts=5, seed=seed)
        elapsed = time() - t0
        rmse = compute_position_rmse(sources, sources_rec)
        results['Conformal L-BFGS-B'] = {'rmse': rmse, 'time': elapsed, 'residual': residual}
    except Exception as e:
        results['Conformal L-BFGS-B'] = {'rmse': float('inf'), 'time': 0, 'error': str(e)}
    
    # Differential evolution
    try:
        inverse = ConformalNonlinearInverseSolver(cmap, n_sources=n_sources, n_boundary=100)
        t0 = time()
        sources_rec, residual = inverse.solve(u_data, method='differential_evolution', seed=seed)
        elapsed = time() - t0
        rmse = compute_position_rmse(sources, sources_rec)
        results['Conformal DiffEvol'] = {'rmse': rmse, 'time': elapsed, 'residual': residual}
    except Exception as e:
        results['Conformal DiffEvol'] = {'rmse': float('inf'), 'time': 0, 'error': str(e)}
    
    return results


def test_fem_nonlinear(domain, seed=42):
    """Test FEM nonlinear solver on a domain."""
    try:
        from fem_solver import FEMNonlinearInverseSolver, FEMForwardSolver
        from mesh import create_disk_mesh, create_ellipse_mesh, create_polygon_mesh, create_brain_mesh
    except ImportError:
        return {'FEM L-BFGS-B': {'rmse': float('inf'), 'error': 'Import failed'},
                'FEM DiffEvol': {'rmse': float('inf'), 'error': 'Import failed'}}
    
    sources = create_domain_sources(domain)
    n_sources = len(sources)
    
    # Create mesh based on domain
    try:
        if domain == 'disk':
            mesh_data = create_disk_mesh(1.0, 0.1)
        elif domain == 'ellipse':
            mesh_data = create_ellipse_mesh(2.0, 1.0, 0.1)
        elif domain == 'star':
            # Star as polygon
            n_pts = 100
            theta = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
            r = 1.0 + 0.3 * np.cos(5 * theta)
            vertices = [(r[i]*np.cos(theta[i]), r[i]*np.sin(theta[i])) for i in range(n_pts)]
            mesh_data = create_polygon_mesh(vertices, 0.1)
        elif domain == 'square':
            vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            mesh_data = create_polygon_mesh(vertices, 0.1)
        elif domain == 'brain':
            mesh_data = create_brain_mesh(0.1)
        else:
            return {'FEM L-BFGS-B': {'rmse': float('inf'), 'error': f'Unknown domain: {domain}'},
                    'FEM DiffEvol': {'rmse': float('inf'), 'error': f'Unknown domain: {domain}'}}
    except Exception as e:
        return {'FEM L-BFGS-B': {'rmse': float('inf'), 'error': f'Mesh creation failed: {e}'},
                'FEM DiffEvol': {'rmse': float('inf'), 'error': f'Mesh creation failed: {e}'}}
    
    # Generate data
    try:
        forward = FEMForwardSolver(resolution=0.1, verbose=False, mesh_data=mesh_data)
        u_data = forward.solve(sources)
    except Exception as e:
        return {'FEM L-BFGS-B': {'rmse': float('inf'), 'error': f'Forward solve failed: {e}'},
                'FEM DiffEvol': {'rmse': float('inf'), 'error': f'Forward solve failed: {e}'}}
    
    results = {}
    
    # L-BFGS-B
    try:
        inverse = FEMNonlinearInverseSolver(n_sources=n_sources, resolution=0.1, 
                                             verbose=False, mesh_data=mesh_data)
        inverse.set_measured_data(u_data)
        t0 = time()
        result = inverse.solve(method='L-BFGS-B', n_restarts=5, maxiter=1000)
        elapsed = time() - t0
        sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
        rmse = compute_position_rmse(sources, sources_rec)
        results['FEM L-BFGS-B'] = {'rmse': rmse, 'time': elapsed, 'residual': result.residual}
    except Exception as e:
        results['FEM L-BFGS-B'] = {'rmse': float('inf'), 'time': 0, 'error': str(e)}
    
    # Differential evolution
    try:
        inverse = FEMNonlinearInverseSolver(n_sources=n_sources, resolution=0.1, 
                                             verbose=False, mesh_data=mesh_data)
        inverse.set_measured_data(u_data)
        t0 = time()
        result = inverse.solve(method='differential_evolution', maxiter=500)
        elapsed = time() - t0
        sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
        rmse = compute_position_rmse(sources, sources_rec)
        results['FEM DiffEvol'] = {'rmse': rmse, 'time': elapsed, 'residual': result.residual}
    except Exception as e:
        results['FEM DiffEvol'] = {'rmse': float('inf'), 'time': 0, 'error': str(e)}
    
    return results


def run_all_tests():
    """Run all tests and collect results."""
    domains = ['disk', 'ellipse', 'star', 'brain', 'square']
    
    all_results = {}
    
    for domain in domains:
        print(f"\n{'='*60}")
        print(f"Testing {domain.upper()}")
        print('='*60)
        
        all_results[domain] = {}
        
        # Conformal nonlinear
        print(f"  Running Conformal Nonlinear...")
        conf_results = test_conformal_nonlinear(domain)
        all_results[domain].update(conf_results)
        
        for solver, data in conf_results.items():
            if 'error' in data:
                print(f"    {solver}: ERROR - {data['error']}")
            else:
                status = "✅" if data['rmse'] < RMSE_THRESHOLD else "❌"
                print(f"    {solver}: RMSE={data['rmse']:.6f}, Time={data['time']:.1f}s {status}")
        
        # FEM nonlinear
        print(f"  Running FEM Nonlinear...")
        fem_results = test_fem_nonlinear(domain)
        all_results[domain].update(fem_results)
        
        for solver, data in fem_results.items():
            if 'error' in data:
                print(f"    {solver}: ERROR - {data['error']}")
            else:
                status = "✅" if data['rmse'] < RMSE_THRESHOLD else "❌"
                print(f"    {solver}: RMSE={data['rmse']:.6f}, Time={data['time']:.1f}s {status}")
    
    return all_results


def print_summary_table(all_results):
    """Print a summary table of all results."""
    domains = ['disk', 'ellipse', 'star', 'brain', 'square']
    solvers = ['Conformal L-BFGS-B', 'Conformal DiffEvol', 'FEM L-BFGS-B', 'FEM DiffEvol']
    
    print("\n\n" + "="*90)
    print("SUMMARY TABLE - Position RMSE")
    print("="*90)
    
    # Header
    header = f"{'Domain':<12}"
    for solver in solvers:
        header += f"{solver:<18}"
    print(header)
    print("-"*90)
    
    # Data rows
    failures = []
    for domain in domains:
        row = f"{domain:<12}"
        for solver in solvers:
            if solver in all_results[domain]:
                data = all_results[domain][solver]
                if 'error' in data:
                    row += f"{'ERROR':<18}"
                    failures.append((domain, solver, 'ERROR'))
                else:
                    rmse = data['rmse']
                    if rmse < RMSE_EXCELLENT:
                        row += f"{rmse:<10.4f} ✅      "
                    elif rmse < RMSE_THRESHOLD:
                        row += f"{rmse:<10.4f} ⚠️      "
                    else:
                        row += f"{rmse:<10.4f} ❌      "
                        failures.append((domain, solver, rmse))
            else:
                row += f"{'N/A':<18}"
        print(row)
    
    print("-"*90)
    print(f"Thresholds: ✅ < {RMSE_EXCELLENT}, ⚠️ < {RMSE_THRESHOLD}, ❌ >= {RMSE_THRESHOLD}")
    
    # List failures
    if failures:
        print("\n" + "="*60)
        print("FAILURES (RMSE >= 0.05 or ERROR)")
        print("="*60)
        for domain, solver, rmse in failures:
            if rmse == 'ERROR':
                print(f"  ❌ {domain} / {solver}: ERROR")
            else:
                print(f"  ❌ {domain} / {solver}: RMSE = {rmse:.4f}")
    else:
        print("\n✅ ALL SOLVERS PASSED!")
    
    return failures


def print_timing_table(all_results):
    """Print timing comparison."""
    domains = ['disk', 'ellipse', 'star', 'brain', 'square']
    solvers = ['Conformal L-BFGS-B', 'Conformal DiffEvol', 'FEM L-BFGS-B', 'FEM DiffEvol']
    
    print("\n\n" + "="*90)
    print("TIMING TABLE (seconds)")
    print("="*90)
    
    # Header
    header = f"{'Domain':<12}"
    for solver in solvers:
        header += f"{solver:<18}"
    print(header)
    print("-"*90)
    
    for domain in domains:
        row = f"{domain:<12}"
        for solver in solvers:
            if solver in all_results[domain]:
                data = all_results[domain][solver]
                if 'time' in data and data['time'] > 0:
                    row += f"{data['time']:<18.1f}"
                else:
                    row += f"{'--':<18}"
            else:
                row += f"{'N/A':<18}"
        print(row)


def main():
    print("="*60)
    print("COMPREHENSIVE SOLVER TEST")
    print("="*60)
    print(f"Testing domains: disk, ellipse, star, brain, square")
    print(f"Testing solvers: Conformal (L-BFGS-B, DiffEvol), FEM (L-BFGS-B, DiffEvol)")
    print(f"Pass threshold: RMSE < {RMSE_THRESHOLD}")
    
    all_results = run_all_tests()
    
    failures = print_summary_table(all_results)
    print_timing_table(all_results)
    
    # Return exit code
    if failures:
        print(f"\n\n❌ {len(failures)} solver(s) failed!")
        return 1
    else:
        print(f"\n\n✅ All solvers passed!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
