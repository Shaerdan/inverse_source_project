#!/usr/bin/env python3
"""
Comprehensive test for FEM nonlinear solvers on disk, ellipse, and square.

Tests multiple source configurations to ensure the fix is robust and not overfit.

Run from src/ directory:
    python test_fem_comprehensive.py
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from comparison import create_domain_sources, get_sensor_locations
from fem_solver import FEMForwardSolver, FEMNonlinearInverseSolver
from mesh import create_disk_mesh, create_ellipse_mesh, create_polygon_mesh
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def compute_position_rmse(sources_true, sources_rec):
    """Compute RMSE between true and recovered source positions."""
    pos_true = np.array([[s[0][0], s[0][1]] for s in sources_true])
    pos_rec = np.array([[(s.x, s.y)] for s in sources_rec]).reshape(-1, 2)
    cost = cdist(pos_true, pos_rec)
    row_ind, col_ind = linear_sum_assignment(cost)
    return np.sqrt(np.mean(cost[row_ind, col_ind]**2))


def test_disk_configurations():
    """Test various source configurations on disk domain."""
    print("\n" + "="*70)
    print("DISK DOMAIN TESTS")
    print("="*70)
    
    sensor_locations = get_sensor_locations('disk', None, 100)
    mesh_data = create_disk_mesh(0.1, sensor_locations=sensor_locations)
    
    tests = {
        "Default (r=0.75, axes)": create_domain_sources('disk'),
        
        "Inner (r=0.4)": [
            ((0.4, 0.0), 1.0), ((0.0, 0.4), -1.0),
            ((-0.4, 0.0), 1.0), ((0.0, -0.4), -1.0),
        ],
        
        "45° rotated": [
            ((0.53, 0.53), 1.0), ((-0.53, 0.53), -1.0),
            ((-0.53, -0.53), 1.0), ((0.53, -0.53), -1.0),
        ],
        
        "Asymmetric": [
            ((0.3, 0.5), 1.5), ((-0.6, 0.2), -0.8),
            ((-0.2, -0.7), 1.2), ((0.5, -0.3), -1.9),
        ],
        
        "2 sources": [
            ((0.5, 0.3), 1.0), ((-0.4, -0.5), -1.0),
        ],
        
        "6 sources": [
            ((0.6, 0.0), 1.0), ((0.3, 0.52), -1.0), ((-0.3, 0.52), 1.0),
            ((-0.6, 0.0), -1.0), ((-0.3, -0.52), 1.0), ((0.3, -0.52), -1.0),
        ],
    }
    
    results = {}
    for name, sources in tests.items():
        n_sources = len(sources)
        inverse = FEMNonlinearInverseSolver(n_sources=n_sources, resolution=0.1, 
                                             mesh_data=mesh_data)
        u_data = inverse.forward.solve(sources)
        inverse.set_measured_data(u_data)
        
        successes = 0
        for trial in range(5):
            np.random.seed(trial * 123)
            result = inverse.solve(method='L-BFGS-B', n_restarts=5, maxiter=2000, 
                                   init_from='random', intensity_bounds=(-3.0, 3.0))
            rmse = compute_position_rmse(sources, result.sources)
            if rmse < 0.05:
                successes += 1
        
        status = "✅" if successes >= 4 else "❌"
        results[name] = successes
        print(f"  {name:<25}: {successes}/5 {status}")
    
    return results


def test_ellipse_configurations():
    """Test various source configurations on ellipse domain."""
    print("\n" + "="*70)
    print("ELLIPSE DOMAIN TESTS")
    print("="*70)
    
    a, b = 2.0, 1.0
    sensor_locations = get_sensor_locations('ellipse', {'a': a, 'b': b}, 100)
    mesh_data = create_ellipse_mesh(a, b, 0.1, sensor_locations=sensor_locations)
    
    tests = {
        "Default": create_domain_sources('ellipse'),
        
        "Inner": [
            ((0.8, 0.0), 1.0), ((0.0, 0.4), -1.0),
            ((-0.8, 0.0), 1.0), ((0.0, -0.4), -1.0),
        ],
        
        "Asymmetric": [
            ((1.2, 0.3), 1.2), ((-0.5, 0.6), -0.9),
            ((-1.0, -0.2), 1.5), ((0.3, -0.5), -1.8),
        ],
        
        "2 sources": [
            ((1.0, 0.3), 1.0), ((-0.8, -0.4), -1.0),
        ],
    }
    
    results = {}
    for name, sources in tests.items():
        n_sources = len(sources)
        inverse = FEMNonlinearInverseSolver.from_ellipse(
            a, b, n_sources=n_sources, resolution=0.1,
            sensor_locations=sensor_locations, mesh_data=mesh_data
        )
        u_data = inverse.forward.solve(sources)
        inverse.set_measured_data(u_data)
        
        successes = 0
        for trial in range(5):
            np.random.seed(trial * 456)
            result = inverse.solve(method='L-BFGS-B', n_restarts=5, maxiter=2000,
                                   init_from='random', intensity_bounds=(-3.0, 3.0))
            rmse = compute_position_rmse(sources, result.sources)
            if rmse < 0.05:
                successes += 1
        
        status = "✅" if successes >= 4 else "❌"
        results[name] = successes
        print(f"  {name:<25}: {successes}/5 {status}")
    
    return results


def test_square_configurations():
    """Test various source configurations on square domain."""
    print("\n" + "="*70)
    print("SQUARE DOMAIN TESTS")
    print("="*70)
    
    vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    sensor_locations = get_sensor_locations('square', None, 100)
    mesh_data = create_polygon_mesh(vertices, 0.1, sensor_locations=sensor_locations)
    
    tests = {
        "Default (corners)": create_domain_sources('square'),
        
        "Edge centers": [
            ((0.7, 0.0), 1.0), ((0.0, 0.7), -1.0),
            ((-0.7, 0.0), 1.0), ((0.0, -0.7), -1.0),
        ],
        
        "Inner": [
            ((0.4, 0.4), 1.0), ((-0.4, 0.4), -1.0),
            ((-0.4, -0.4), 1.0), ((0.4, -0.4), -1.0),
        ],
        
        "Asymmetric": [
            ((0.5, 0.6), 1.3), ((-0.7, 0.3), -0.8),
            ((-0.3, -0.6), 1.1), ((0.6, -0.4), -1.6),
        ],
        
        "2 sources": [
            ((0.5, 0.5), 1.0), ((-0.5, -0.5), -1.0),
        ],
    }
    
    results = {}
    for name, sources in tests.items():
        n_sources = len(sources)
        inverse = FEMNonlinearInverseSolver.from_polygon(
            vertices, n_sources=n_sources, resolution=0.1,
            sensor_locations=sensor_locations, mesh_data=mesh_data
        )
        u_data = inverse.forward.solve(sources)
        inverse.set_measured_data(u_data)
        
        successes = 0
        for trial in range(5):
            np.random.seed(trial * 789)
            result = inverse.solve(method='L-BFGS-B', n_restarts=5, maxiter=2000,
                                   init_from='random', intensity_bounds=(-3.0, 3.0))
            rmse = compute_position_rmse(sources, result.sources)
            if rmse < 0.05:
                successes += 1
        
        status = "✅" if successes >= 4 else "❌"
        results[name] = successes
        print(f"  {name:<25}: {successes}/5 {status}")
    
    return results


def test_polygon_configurations():
    """Test L-shaped polygon domain."""
    print("\n" + "="*70)
    print("L-SHAPED POLYGON DOMAIN TESTS")
    print("="*70)
    
    # L-shaped polygon
    vertices = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    sensor_locations = get_sensor_locations('polygon', {'vertices': vertices}, 100)
    mesh_data = create_polygon_mesh(vertices, 0.1, sensor_locations=sensor_locations)
    
    tests = {
        "Default": create_domain_sources('polygon', domain_params={'vertices': vertices}),
        
        "Different positions": [
            ((0.5, 0.5), 1.0), ((1.5, 0.5), -1.0),
            ((0.5, 1.5), 1.0), ((0.8, 0.8), -1.0),
        ],
        
        "2 sources": [
            ((0.5, 0.5), 1.0), ((1.5, 0.5), -1.0),
        ],
    }
    
    results = {}
    for name, sources in tests.items():
        n_sources = len(sources)
        inverse = FEMNonlinearInverseSolver.from_polygon(
            vertices, n_sources=n_sources, resolution=0.1,
            sensor_locations=sensor_locations, mesh_data=mesh_data
        )
        u_data = inverse.forward.solve(sources)
        inverse.set_measured_data(u_data)
        
        successes = 0
        for trial in range(5):
            np.random.seed(trial * 999)
            result = inverse.solve(method='L-BFGS-B', n_restarts=5, maxiter=2000,
                                   init_from='random', intensity_bounds=(-3.0, 3.0))
            rmse = compute_position_rmse(sources, result.sources)
            if rmse < 0.05:
                successes += 1
        
        status = "✅" if successes >= 4 else "❌"
        results[name] = successes
        print(f"  {name:<25}: {successes}/5 {status}")
    
    return results


def main():
    print("="*70)
    print("FEM NONLINEAR SOLVER - COMPREHENSIVE TEST")
    print("="*70)
    print("""
Tests multiple source configurations per domain to verify robustness.
Each configuration runs 5 trials with n_restarts=5.
Pass criterion: >= 4/5 trials succeed (RMSE < 0.05)
""")
    
    disk_results = test_disk_configurations()
    ellipse_results = test_ellipse_configurations()
    square_results = test_square_configurations()
    polygon_results = test_polygon_configurations()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_results = {
        'disk': disk_results,
        'ellipse': ellipse_results,
        'square': square_results,
        'polygon': polygon_results,
    }
    
    total_tests = 0
    total_passed = 0
    
    for domain, results in all_results.items():
        passed = sum(1 for v in results.values() if v >= 4)
        total = len(results)
        total_tests += total
        total_passed += passed
        status = "✅" if passed == total else "⚠️"
        print(f"  {domain.upper():<10}: {passed}/{total} configurations passed {status}")
    
    print(f"\n  OVERALL: {total_passed}/{total_tests} configurations passed")
    
    if total_passed == total_tests:
        print("\n  ✅ ALL TESTS PASSED - FEM solvers are working correctly!")
    else:
        print("\n  ⚠️ SOME TESTS FAILED - Review failing configurations above")


if __name__ == '__main__':
    main()
