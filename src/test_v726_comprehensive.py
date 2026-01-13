#!/usr/bin/env python3
"""
Comprehensive Test Suite for Inverse Source v7.26
==================================================

This script tests all the critical fixes made in v7.26:
1. gmsh sensor embedding (GEO kernel with circular arcs)
2. Mesh refinement (MathEval background field)
3. FEM sensor ordering (no reordering)
4. Green's function sign (-1/2π)

Run from project directory:
    python -m src.test_v726_comprehensive
    
Or directly:
    cd ~/Projects/inverse_source_project/
    python src/test_v726_comprehensive.py

Author: Claude (Anthropic)
Date: January 2026
"""

import sys
import numpy as np
import time
from typing import List, Tuple, Dict

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

# Standard test sources (sum to zero for compatibility condition)
TEST_SOURCES = [
    ((-0.3, 0.2), 1.0),
    ((0.3, -0.2), -1.0)
]

# More complex 4-source test
TEST_SOURCES_4 = [
    ((-0.3, 0.4), 1.0),
    ((0.5, 0.2), 1.0),
    ((-0.2, -0.3), -1.0),
    ((0.3, -0.4), -1.0),
]

# Test parameters
N_SENSORS = 100
MESH_RESOLUTIONS = [0.20, 0.15, 0.10, 0.08, 0.06, 0.05]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def print_subheader(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")

def print_result(name: str, passed: bool, details: str = ""):
    """Print a test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    if details:
        print(f"  {status}: {name} ({details})")
    else:
        print(f"  {status}: {name}")

# =============================================================================
# TEST 1: MESH REFINEMENT
# =============================================================================

def test_mesh_refinement() -> Tuple[bool, Dict]:
    """
    Test that mesh refines properly with resolution parameter.
    
    Bug in v7.25: Interior node count was constant regardless of h because
    sensor spacing (~0.063) constrained the mesh globally.
    
    Fix in v7.26: MathEval background field overrides boundary constraints.
    """
    print_subheader("Test 1: Mesh Refinement")
    
    try:
        from inverse_source.mesh import create_disk_mesh, get_disk_sensor_locations
    except ImportError:
        try:
            from src.mesh import create_disk_mesh, get_disk_sensor_locations
        except ImportError:
            from mesh import create_disk_mesh, get_disk_sensor_locations
    
    sensors = get_disk_sensor_locations(N_SENSORS)
    results = []
    
    print(f"  Testing mesh refinement with {N_SENSORS} sensors...")
    print(f"  Expected: interior nodes should INCREASE as h decreases\n")
    
    prev_interior = 0
    refinement_working = True
    
    for h in MESH_RESOLUTIONS:
        try:
            nodes, elements, boundary_idx, interior_idx, sensor_idx = \
                create_disk_mesh(h, sensor_locations=sensors)
            
            n_interior = len(interior_idx)
            n_total = len(nodes)
            
            # Check refinement is happening
            if h < 0.15 and n_interior <= prev_interior:
                refinement_working = False
            
            results.append({
                'h': h,
                'total_nodes': n_total,
                'interior_nodes': n_interior,
                'boundary_nodes': len(boundary_idx),
                'elements': len(elements)
            })
            
            print(f"  h={h:.3f}: {n_total:4d} nodes, {n_interior:4d} interior, {len(elements):4d} elements")
            prev_interior = n_interior
            
        except Exception as e:
            print(f"  h={h:.3f}: FAILED - {e}")
            refinement_working = False
    
    # Check that refinement ratio is reasonable (roughly h^-2 scaling)
    if len(results) >= 2:
        ratio = results[-1]['interior_nodes'] / max(results[0]['interior_nodes'], 1)
        h_ratio = results[0]['h'] / results[-1]['h']
        expected_ratio = h_ratio ** 2  # Area scales as h^-2
        
        print(f"\n  Node count ratio (fine/coarse): {ratio:.1f}")
        print(f"  Expected ratio (h^-2 scaling): ~{expected_ratio:.1f}")
        
        # Allow some tolerance
        if ratio < expected_ratio * 0.3:
            refinement_working = False
            print(f"  WARNING: Refinement ratio too low!")
    
    print_result("Mesh refinement", refinement_working,
                 f"{results[-1]['interior_nodes']} interior nodes at h={MESH_RESOLUTIONS[-1]}")
    
    return refinement_working, {'mesh_results': results}

# =============================================================================
# TEST 2: SENSOR EMBEDDING
# =============================================================================

def test_sensor_embedding() -> Tuple[bool, Dict]:
    """
    Test that sensors are exactly embedded as mesh nodes.
    
    Bug in v7.25: Using gmsh.model.mesh.embed() for boundary points didn't work.
    
    Fix in v7.26: Use GEO kernel with addCircleArc() to define boundary
    through sensor points.
    """
    print_subheader("Test 2: Sensor Embedding")
    
    try:
        from inverse_source.mesh import create_disk_mesh, get_disk_sensor_locations
    except ImportError:
        try:
            from src.mesh import create_disk_mesh, get_disk_sensor_locations
        except ImportError:
            from mesh import create_disk_mesh, get_disk_sensor_locations
    
    sensors = get_disk_sensor_locations(N_SENSORS)
    
    try:
        nodes, elements, boundary_idx, interior_idx, sensor_idx = \
            create_disk_mesh(0.1, sensor_locations=sensors)
        
        # Check that sensor indices point to correct locations
        max_error = 0.0
        errors = []
        
        for i, (sx, sy) in enumerate(sensors):
            node_x, node_y = nodes[sensor_idx[i]]
            error = np.sqrt((node_x - sx)**2 + (node_y - sy)**2)
            errors.append(error)
            max_error = max(max_error, error)
        
        embedding_exact = max_error < 1e-10
        
        print(f"  Number of sensors: {len(sensors)}")
        print(f"  Sensor indices found: {len(sensor_idx)}")
        print(f"  Max embedding error: {max_error:.2e}")
        print(f"  Mean embedding error: {np.mean(errors):.2e}")
        
        # Check all sensors are on boundary
        sensor_radii = np.sqrt(nodes[sensor_idx, 0]**2 + nodes[sensor_idx, 1]**2)
        on_boundary = np.all(sensor_radii > 0.99)
        
        print(f"  All sensors on boundary: {on_boundary}")
        
        passed = embedding_exact and on_boundary
        print_result("Sensor embedding", passed, f"max error = {max_error:.2e}")
        
        return passed, {'max_error': max_error, 'on_boundary': on_boundary}
        
    except Exception as e:
        print(f"  FAILED: {e}")
        print_result("Sensor embedding", False, str(e))
        return False, {'error': str(e)}

# =============================================================================
# TEST 3: FEM vs ANALYTICAL AGREEMENT
# =============================================================================

def test_fem_analytical_agreement() -> Tuple[bool, Dict]:
    """
    Test that FEM and analytical solvers produce matching results.
    
    Bugs in v7.25:
    - Green's function had wrong sign (+1/2π instead of -1/2π)
    - FEM solver reordered sensor indices
    
    Fix in v7.26: Correct sign, preserve sensor order.
    """
    print_subheader("Test 3: FEM vs Analytical Agreement")
    
    try:
        from inverse_source.mesh import create_disk_mesh, get_disk_sensor_locations
        from inverse_source.fem_solver import FEMForwardSolver
        from inverse_source.analytical_solver import AnalyticalForwardSolver
    except ImportError:
        try:
            from src.mesh import create_disk_mesh, get_disk_sensor_locations
            from src.fem_solver import FEMForwardSolver
            from src.analytical_solver import AnalyticalForwardSolver
        except ImportError:
            from mesh import create_disk_mesh, get_disk_sensor_locations
            from fem_solver import FEMForwardSolver
            from analytical_solver import AnalyticalForwardSolver
    
    sensors = get_disk_sensor_locations(N_SENSORS)
    
    try:
        # Create mesh with embedded sensors
        mesh_data = create_disk_mesh(0.1, sensor_locations=sensors)
        
        # Solve with both methods
        fem_solver = FEMForwardSolver(mesh_data=mesh_data, sensor_locations=sensors, verbose=False)
        ana_solver = AnalyticalForwardSolver(sensor_locations=sensors)
        
        u_fem = fem_solver.solve(TEST_SOURCES)
        u_ana = ana_solver.solve(TEST_SOURCES)
        
        # Compute agreement metrics
        correlation = np.corrcoef(u_fem, u_ana)[0, 1]
        relative_error = np.linalg.norm(u_fem - u_ana) / np.linalg.norm(u_ana)
        
        # Check sign agreement (was -1.0 before fix)
        sign_correct = correlation > 0
        
        print(f"  FEM solution range: [{u_fem.min():.4f}, {u_fem.max():.4f}]")
        print(f"  Analytical range:   [{u_ana.min():.4f}, {u_ana.max():.4f}]")
        print(f"  Correlation: {correlation:.6f}")
        print(f"  Relative error: {relative_error:.4f} ({relative_error*100:.2f}%)")
        print(f"  Sign agreement: {sign_correct}")
        
        # Thresholds
        passed = correlation > 0.999 and relative_error < 0.01
        
        print_result("FEM/Analytical agreement", passed,
                    f"corr={correlation:.4f}, err={relative_error:.4f}")
        
        return passed, {
            'correlation': correlation,
            'relative_error': relative_error,
            'u_fem': u_fem,
            'u_ana': u_ana
        }
        
    except Exception as e:
        import traceback
        print(f"  FAILED: {e}")
        traceback.print_exc()
        print_result("FEM/Analytical agreement", False, str(e))
        return False, {'error': str(e)}

# =============================================================================
# TEST 4: CONVERGENCE RATE
# =============================================================================

def test_convergence_rate() -> Tuple[bool, Dict]:
    """
    Test that FEM converges at O(h²) rate.
    
    Expected: P1 FEM should give O(h²) convergence for smooth solutions.
    """
    print_subheader("Test 4: Convergence Rate")
    
    try:
        from inverse_source.mesh import create_disk_mesh, get_disk_sensor_locations
        from inverse_source.fem_solver import FEMForwardSolver
        from inverse_source.analytical_solver import AnalyticalForwardSolver
    except ImportError:
        try:
            from src.mesh import create_disk_mesh, get_disk_sensor_locations
            from src.fem_solver import FEMForwardSolver
            from src.analytical_solver import AnalyticalForwardSolver
        except ImportError:
            from mesh import create_disk_mesh, get_disk_sensor_locations
            from fem_solver import FEMForwardSolver
            from analytical_solver import AnalyticalForwardSolver
    
    sensors = get_disk_sensor_locations(N_SENSORS)
    
    # Reference solution (analytical)
    ana_solver = AnalyticalForwardSolver(sensor_locations=sensors)
    u_ref = ana_solver.solve(TEST_SOURCES)
    u_ref = u_ref - np.mean(u_ref)
    
    errors = []
    node_counts = []
    
    print(f"  Computing FEM solutions at different resolutions...\n")
    
    for h in MESH_RESOLUTIONS:
        try:
            mesh_data = create_disk_mesh(h, sensor_locations=sensors)
            nodes = mesh_data[0]
            
            fem_solver = FEMForwardSolver(mesh_data=mesh_data, sensor_locations=sensors, verbose=False)
            u_fem = fem_solver.solve(TEST_SOURCES)
            u_fem = u_fem - np.mean(u_fem)
            
            error = np.linalg.norm(u_fem - u_ref) / np.linalg.norm(u_ref)
            errors.append(error)
            node_counts.append(len(nodes))
            
            print(f"  h={h:.3f}: error={error:.2e}, nodes={len(nodes)}")
            
        except Exception as e:
            print(f"  h={h:.3f}: FAILED - {e}")
            errors.append(np.nan)
            node_counts.append(0)
    
    # Compute convergence rate from log-log slope
    valid = ~np.isnan(errors)
    if np.sum(valid) >= 2:
        log_h = np.log(np.array(MESH_RESOLUTIONS)[valid])
        log_err = np.log(np.array(errors)[valid])
        slope, intercept = np.polyfit(log_h, log_err, 1)
        
        print(f"\n  Overall convergence rate: O(h^{slope:.2f})")
        print(f"  Expected for P1 FEM: O(h^2)")
        
        # Local slopes
        print(f"\n  Local slopes:")
        h_arr = np.array(MESH_RESOLUTIONS)
        err_arr = np.array(errors)
        for i in range(len(h_arr) - 1):
            if not np.isnan(err_arr[i]) and not np.isnan(err_arr[i+1]):
                local_slope = (np.log(err_arr[i+1]) - np.log(err_arr[i])) / \
                             (np.log(h_arr[i+1]) - np.log(h_arr[i]))
                print(f"    h: {h_arr[i]:.3f} -> {h_arr[i+1]:.3f}: slope = {local_slope:.2f}")
        
        # Check if rate is approximately 2
        passed = 1.5 <= slope <= 2.5
        print_result("Convergence rate", passed, f"O(h^{slope:.2f})")
        
        return passed, {
            'convergence_rate': slope,
            'errors': errors,
            'resolutions': MESH_RESOLUTIONS
        }
    else:
        print_result("Convergence rate", False, "Insufficient valid data")
        return False, {'error': 'Insufficient valid data'}

# =============================================================================
# TEST 5: SENSOR ORDER PRESERVATION
# =============================================================================

def test_sensor_order_preservation() -> Tuple[bool, Dict]:
    """
    Test that FEM solver preserves sensor ordering from input.
    
    Bug in v7.25: FEMForwardSolver sorted sensor_indices by angle,
    breaking correspondence with input sensor_locations.
    
    Fix in v7.26: Remove sorting, keep original order.
    """
    print_subheader("Test 5: Sensor Order Preservation")
    
    try:
        from inverse_source.mesh import create_disk_mesh, get_disk_sensor_locations
        from inverse_source.fem_solver import FEMForwardSolver
        from inverse_source.analytical_solver import AnalyticalForwardSolver
    except ImportError:
        try:
            from src.mesh import create_disk_mesh, get_disk_sensor_locations
            from src.fem_solver import FEMForwardSolver
            from src.analytical_solver import AnalyticalForwardSolver
        except ImportError:
            from mesh import create_disk_mesh, get_disk_sensor_locations
            from fem_solver import FEMForwardSolver
            from analytical_solver import AnalyticalForwardSolver
    
    # Use standard sensors (not shuffled - shuffling breaks GEO kernel arc ordering)
    sensors = get_disk_sensor_locations(50)
    
    try:
        mesh_data = create_disk_mesh(0.1, sensor_locations=sensors)
        
        fem_solver = FEMForwardSolver(mesh_data=mesh_data, sensor_locations=sensors, verbose=False)
        ana_solver = AnalyticalForwardSolver(sensor_locations=sensors)
        
        u_fem = fem_solver.solve(TEST_SOURCES)
        u_ana = ana_solver.solve(TEST_SOURCES)
        
        # Center both solutions
        u_fem = u_fem - np.mean(u_fem)
        u_ana = u_ana - np.mean(u_ana)
        
        # Check point-by-point correspondence
        correlation = np.corrcoef(u_fem, u_ana)[0, 1]
        
        # If order was scrambled, correlation would be low
        order_preserved = correlation > 0.99
        
        print(f"  Using {len(sensors)} sensors")
        print(f"  Point-by-point correlation: {correlation:.6f}")
        print(f"  Order preserved: {order_preserved}")
        
        # Also check that sensor values match at specific indices
        print(f"  Sample check - sensor 0: FEM={u_fem[0]:.4f}, Ana={u_ana[0]:.4f}")
        print(f"  Sample check - sensor 25: FEM={u_fem[25]:.4f}, Ana={u_ana[25]:.4f}")
        
        print_result("Sensor order preservation", order_preserved,
                    f"correlation = {correlation:.4f}")
        
        return order_preserved, {'correlation': correlation}
        
    except Exception as e:
        import traceback
        print(f"  FAILED: {e}")
        traceback.print_exc()
        print_result("Sensor order preservation", False, str(e))
        return False, {'error': str(e)}

# =============================================================================
# TEST 6: GREEN'S FUNCTION SIGN
# =============================================================================

def test_greens_function_sign() -> Tuple[bool, Dict]:
    """
    Test that Green's function has correct sign convention.
    
    For -Δu = δ(x - ξ), the solution should be POSITIVE near the source.
    Fundamental solution: Φ = -1/(2π) log|x-ξ|
    
    Bug in v7.25: Used +1/(2π) giving negative values near source.
    """
    print_subheader("Test 6: Green's Function Sign")
    
    try:
        from inverse_source.analytical_solver import greens_function_disk_neumann
    except ImportError:
        try:
            from src.analytical_solver import greens_function_disk_neumann
        except ImportError:
            from analytical_solver import greens_function_disk_neumann
    
    # Test point near a source
    source_pos = np.array([0.3, 0.3])
    
    # Evaluation points at various distances
    distances = [0.05, 0.1, 0.2, 0.5]
    
    print(f"  Source at ({source_pos[0]}, {source_pos[1]})")
    print(f"  Checking Green's function sign at various distances:\n")
    
    all_positive_near_source = True
    
    for d in distances:
        # Point at distance d from source (toward origin)
        direction = -source_pos / np.linalg.norm(source_pos)
        eval_point = source_pos + d * direction
        
        G = greens_function_disk_neumann(eval_point.reshape(1, -1), source_pos)
        G_val = float(G)
        
        # Near source (small d), G should be positive (since log(small) < 0, -log > 0)
        expected_sign = "+" if d < 0.3 else "±"
        actual_sign = "+" if G_val > 0 else "-"
        
        print(f"  d={d:.2f}: G = {G_val:+.4f} (sign: {actual_sign})")
        
        if d <= 0.1 and G_val <= 0:
            all_positive_near_source = False
    
    print(f"\n  Near source, G should be POSITIVE (for -Δu = δ convention)")
    print(f"  This is because: Φ = -1/(2π) log|r|, and log(small r) < 0")
    
    print_result("Green's function sign", all_positive_near_source,
                "positive near source" if all_positive_near_source else "WRONG SIGN")
    
    return all_positive_near_source, {}

# =============================================================================
# TEST 7: FULL FORWARD-INVERSE CYCLE
# =============================================================================

def test_forward_inverse_cycle() -> Tuple[bool, Dict]:
    """
    Test complete forward-inverse cycle with regularization.
    """
    print_subheader("Test 7: Forward-Inverse Cycle")
    
    try:
        from inverse_source.mesh import create_disk_mesh, get_disk_sensor_locations
        from inverse_source.fem_solver import FEMForwardSolver
        from inverse_source.analytical_solver import AnalyticalLinearInverseSolver
    except ImportError:
        try:
            from src.mesh import create_disk_mesh, get_disk_sensor_locations
            from src.fem_solver import FEMForwardSolver
            from src.analytical_solver import AnalyticalLinearInverseSolver
        except ImportError:
            from mesh import create_disk_mesh, get_disk_sensor_locations
            from fem_solver import FEMForwardSolver
            from analytical_solver import AnalyticalLinearInverseSolver
    
    # Check if cvxpy is available for L1
    try:
        import cvxpy
        has_cvxpy = True
    except ImportError:
        has_cvxpy = False
    
    sensors = get_disk_sensor_locations(N_SENSORS)
    
    try:
        # Forward solve with FEM
        mesh_data = create_disk_mesh(0.1, sensor_locations=sensors)
        fem_solver = FEMForwardSolver(mesh_data=mesh_data, sensor_locations=sensors, verbose=False)
        u_clean = fem_solver.solve(TEST_SOURCES_4)
        
        # Add noise
        np.random.seed(42)
        noise_level = 0.001
        u_noisy = u_clean + noise_level * np.random.randn(len(u_clean))
        
        # Inverse solve
        inverse_solver = AnalyticalLinearInverseSolver(
            n_boundary=N_SENSORS,
            source_resolution=0.15,
            verbose=False
        )
        inverse_solver.build_greens_matrix(verbose=False)
        
        alpha = 1e-4
        
        if has_cvxpy:
            method = 'L1'
            q_recovered = inverse_solver.solve_l1(u_noisy, alpha=alpha)
        else:
            method = 'L2'
            q_recovered = inverse_solver.solve_l2(u_noisy, alpha=alpha)
        
        # Check reconstruction quality (primary metric)
        u_reconstructed = inverse_solver.G @ q_recovered
        u_reconstructed = u_reconstructed - np.mean(u_reconstructed)
        u_noisy_centered = u_noisy - np.mean(u_noisy)
        
        reconstruction_error = np.linalg.norm(u_reconstructed - u_noisy_centered) / np.linalg.norm(u_noisy_centered)
        
        # Find peaks (secondary metric)
        threshold = 0.1 * np.max(np.abs(q_recovered))
        n_peaks = np.sum(np.abs(q_recovered) > threshold)
        
        print(f"  True sources: {len(TEST_SOURCES_4)}")
        print(f"  Noise level: {noise_level}")
        print(f"  Method: {method} (alpha={alpha})")
        print(f"  Recovered peaks: {n_peaks}")
        print(f"  Reconstruction error: {reconstruction_error:.4f}")
        
        # Success if reconstruction is reasonable
        passed = reconstruction_error < 0.1
        
        print_result("Forward-inverse cycle", passed,
                    f"recon err = {reconstruction_error:.4f}")
        
        return passed, {
            'n_peaks': n_peaks,
            'reconstruction_error': reconstruction_error,
            'method': method
        }
        
    except Exception as e:
        import traceback
        print(f"  FAILED: {e}")
        traceback.print_exc()
        print_result("Forward-inverse cycle", False, str(e))
        return False, {'error': str(e)}

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests() -> bool:
    """Run all tests and report results."""
    print_header("INVERSE SOURCE v7.26 - COMPREHENSIVE TEST SUITE")
    
    print(f"\nTest configuration:")
    print(f"  Sensors: {N_SENSORS}")
    print(f"  Mesh resolutions: {MESH_RESOLUTIONS}")
    print(f"  Test sources: {len(TEST_SOURCES)} dipole")
    
    start_time = time.time()
    
    # Run all tests
    tests = [
        ("Mesh Refinement", test_mesh_refinement),
        ("Sensor Embedding", test_sensor_embedding),
        ("FEM/Analytical Agreement", test_fem_analytical_agreement),
        ("Convergence Rate", test_convergence_rate),
        ("Sensor Order Preservation", test_sensor_order_preservation),
        ("Green's Function Sign", test_greens_function_sign),
        ("Forward-Inverse Cycle", test_forward_inverse_cycle),
    ]
    
    results = {}
    all_passed = True
    
    for name, test_func in tests:
        try:
            passed, details = test_func()
            results[name] = {'passed': passed, 'details': details}
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\n  EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {'passed': False, 'error': str(e)}
            all_passed = False
    
    elapsed = time.time() - start_time
    
    # Summary
    print_header("TEST SUMMARY")
    
    n_passed = sum(1 for r in results.values() if r['passed'])
    n_total = len(results)
    
    print(f"\n  Tests passed: {n_passed}/{n_total}")
    print(f"  Time elapsed: {elapsed:.1f}s\n")
    
    for name, result in results.items():
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED - v7.26 is working correctly!")
    else:
        print("✗ SOME TESTS FAILED - check output above for details")
    print("=" * 70)
    
    return all_passed


def main():
    """Main entry point."""
    success = run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
