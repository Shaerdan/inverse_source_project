#!/usr/bin/env python
"""
Verify inverse_source installation is correct.

This script tests both the installation and the critical v7.26 fixes:
1. gmsh sensor embedding (GEO kernel with circular arcs)
2. Mesh refinement (MathEval background field)
3. FEM sensor ordering (no reordering)
4. Green's function sign (-1/2π)

Run:
    python -m src.verify_installation
    
Or:
    python src/verify_installation.py
"""

import sys
import numpy as np

# =============================================================================
# IMPORT CHECKS
# =============================================================================

def check_imports():
    """Check all required imports are available."""
    print("Checking imports...")
    
    required = [
        ('numpy', 'np'),
        ('scipy', None),
        ('matplotlib', None),
    ]
    
    optional = [
        ('cvxpy', 'Regularization methods'),
        ('gmsh', 'Quality mesh generation'),
    ]
    
    all_required_ok = True
    
    for module, alias in required:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            all_required_ok = False
    
    print("\nOptional dependencies:")
    for module, purpose in optional:
        try:
            __import__(module)
            print(f"  ✓ {module} ({purpose})")
        except ImportError:
            print(f"  - {module} not installed ({purpose})")
    
    return all_required_ok

def check_package_imports():
    """Check package modules import correctly."""
    print("\nChecking package modules...")
    
    modules = [
        'mesh',
        'fem_solver',
        'analytical_solver',
        'comparison',
        'parameter_selection',
    ]
    
    all_ok = True
    for module in modules:
        # Try inverse_source.xxx first (installed package), then src.xxx, then direct
        try:
            __import__(f'inverse_source.{module}')
            print(f"  ✓ inverse_source.{module}")
        except ImportError:
            try:
                __import__(f'src.{module}')
                print(f"  ✓ src.{module}")
            except ImportError:
                try:
                    __import__(module)
                    print(f"  ✓ {module}")
                except ImportError as e:
                    print(f"  ✗ {module}: {e}")
                    all_ok = False
    
    return all_ok

# =============================================================================
# V7.26 SPECIFIC TESTS
# =============================================================================

def test_mesh_refinement():
    """Test that mesh refines properly (v7.26 fix)."""
    print("\n[v7.26] Testing mesh refinement...")
    
    try:
        from inverse_source.mesh import create_disk_mesh, get_disk_sensor_locations
    except ImportError:
        try:
            from src.mesh import create_disk_mesh, get_disk_sensor_locations
        except ImportError:
            from mesh import create_disk_mesh, get_disk_sensor_locations
    
    sensors = get_disk_sensor_locations(100)
    
    # Test two resolutions
    _, _, _, interior1, _ = create_disk_mesh(0.15, sensor_locations=sensors)
    _, _, _, interior2, _ = create_disk_mesh(0.08, sensor_locations=sensors)
    
    n1, n2 = len(interior1), len(interior2)
    
    # Finer mesh should have more interior nodes
    passed = n2 > n1 * 1.5  # At least 50% more nodes
    
    print(f"  h=0.15: {n1} interior nodes")
    print(f"  h=0.08: {n2} interior nodes")
    print(f"  Refinement working: {'✓' if passed else '✗'}")
    
    return passed

def test_sensor_embedding():
    """Test that sensors are exactly embedded (v7.26 fix)."""
    print("\n[v7.26] Testing sensor embedding...")
    
    try:
        from inverse_source.mesh import create_disk_mesh, get_disk_sensor_locations
    except ImportError:
        try:
            from src.mesh import create_disk_mesh, get_disk_sensor_locations
        except ImportError:
            from mesh import create_disk_mesh, get_disk_sensor_locations
    
    sensors = get_disk_sensor_locations(100)
    nodes, _, _, _, sensor_idx = create_disk_mesh(0.1, sensor_locations=sensors)
    
    # Check embedding accuracy
    max_error = 0.0
    for i, (sx, sy) in enumerate(sensors):
        nx, ny = nodes[sensor_idx[i]]
        error = np.sqrt((nx - sx)**2 + (ny - sy)**2)
        max_error = max(max_error, error)
    
    passed = max_error < 1e-10
    print(f"  Max embedding error: {max_error:.2e}")
    print(f"  Exact embedding: {'✓' if passed else '✗'}")
    
    return passed

def test_fem_analytical_agreement():
    """Test FEM vs analytical agreement (v7.26 fix)."""
    print("\n[v7.26] Testing FEM vs Analytical agreement...")
    
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
    
    sources = [((-0.3, 0.2), 1.0), ((0.3, -0.2), -1.0)]
    sensors = get_disk_sensor_locations(100)
    mesh_data = create_disk_mesh(0.1, sensor_locations=sensors)
    
    fem = FEMForwardSolver(mesh_data=mesh_data, sensor_locations=sensors, verbose=False)
    ana = AnalyticalForwardSolver(sensor_locations=sensors)
    
    u_fem = fem.solve(sources)
    u_ana = ana.solve(sources)
    
    correlation = np.corrcoef(u_fem, u_ana)[0, 1]
    rel_error = np.linalg.norm(u_fem - u_ana) / np.linalg.norm(u_ana)
    
    passed = correlation > 0.999 and rel_error < 0.01
    
    print(f"  Correlation: {correlation:.6f}")
    print(f"  Relative error: {rel_error:.4f}")
    print(f"  Agreement: {'✓' if passed else '✗'}")
    
    return passed

def test_greens_function_sign():
    """Test Green's function has correct sign (v7.26 fix)."""
    print("\n[v7.26] Testing Green's function sign...")
    
    try:
        from inverse_source.analytical_solver import greens_function_disk_neumann
    except ImportError:
        try:
            from src.analytical_solver import greens_function_disk_neumann
        except ImportError:
            from analytical_solver import greens_function_disk_neumann
    
    # Source inside disk
    source = np.array([0.3, 0.3])
    
    # Evaluation point close to source
    eval_point = np.array([[0.35, 0.35]])
    
    G = greens_function_disk_neumann(eval_point, source)
    
    # Near the source, G should be POSITIVE for -Δu = δ convention
    # Because Φ = -1/(2π) log|r| and log(small r) < 0
    passed = float(G) > 0
    
    print(f"  G near source: {float(G):.4f}")
    print(f"  Sign correct (positive): {'✓' if passed else '✗'}")
    
    return passed

def test_convergence_rate():
    """Test O(h²) convergence rate."""
    print("\n[v7.26] Testing convergence rate...")
    
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
    
    sources = [((-0.3, 0.2), 1.0), ((0.3, -0.2), -1.0)]
    sensors = get_disk_sensor_locations(100)
    
    ana = AnalyticalForwardSolver(sensor_locations=sensors)
    u_ref = ana.solve(sources)
    u_ref = u_ref - np.mean(u_ref)
    
    resolutions = [0.15, 0.10, 0.06]
    errors = []
    
    for h in resolutions:
        mesh_data = create_disk_mesh(h, sensor_locations=sensors)
        fem = FEMForwardSolver(mesh_data=mesh_data, sensor_locations=sensors, verbose=False)
        u_fem = fem.solve(sources)
        u_fem = u_fem - np.mean(u_fem)
        errors.append(np.linalg.norm(u_fem - u_ref) / np.linalg.norm(u_ref))
    
    # Compute convergence rate
    log_h = np.log(resolutions)
    log_err = np.log(errors)
    rate, _ = np.polyfit(log_h, log_err, 1)
    
    # Accept rates from 1.5 to 3.0 (superconvergence is OK)
    passed = 1.5 <= rate <= 3.0
    
    print(f"  Convergence rate: O(h^{rate:.2f})")
    print(f"  Expected: O(h^2) or better")
    print(f"  Rate correct: {'✓' if passed else '✗'}")
    
    return passed

# =============================================================================
# L-CURVE / REGULARIZATION TESTS
# =============================================================================

def test_lcurve_analysis():
    """Test L-curve corner detection works."""
    print("\nTesting L-curve analysis...")
    
    try:
        from inverse_source.parameter_selection import find_lcurve_corner
    except ImportError:
        try:
            from src.parameter_selection import find_lcurve_corner
        except ImportError:
            from parameter_selection import find_lcurve_corner
    
    # Create synthetic L-curve data
    alphas = np.logspace(-6, -1, 20)
    residuals = 1.0 / (alphas + 1e-5)  # Decreases with alpha
    regularizers = alphas * 100  # Increases with alpha
    
    idx = find_lcurve_corner(residuals, regularizers)
    
    passed = 5 <= idx <= 15  # Should find corner in middle region
    
    print(f"  L-curve corner index: {idx}")
    print(f"  L-curve detection: {'✓' if passed else '✗'}")
    
    return passed

def test_inverse_solver():
    """Test inverse solver produces reasonable results."""
    print("\nTesting inverse solver...")
    
    try:
        from inverse_source.analytical_solver import AnalyticalForwardSolver, AnalyticalLinearInverseSolver
    except ImportError:
        try:
            from src.analytical_solver import AnalyticalForwardSolver, AnalyticalLinearInverseSolver
        except ImportError:
            from analytical_solver import AnalyticalForwardSolver, AnalyticalLinearInverseSolver
    
    sources = [((-0.3, 0.4), 1.0), ((0.3, -0.4), -1.0)]
    
    forward = AnalyticalForwardSolver(n_boundary_points=100)
    u = forward.solve(sources)
    
    np.random.seed(42)
    u_noisy = u + 0.001 * np.random.randn(len(u))
    
    inverse = AnalyticalLinearInverseSolver(n_boundary=100, source_resolution=0.15, verbose=False)
    inverse.build_greens_matrix(verbose=False)
    
    try:
        q = inverse.solve_l2(u_noisy, alpha=1e-4)
        
        # Check reconstruction quality (more reliable than peak count for L2)
        u_recon = inverse.G @ q
        u_recon = u_recon - np.mean(u_recon)
        u_centered = u_noisy - np.mean(u_noisy)
        recon_error = np.linalg.norm(u_recon - u_centered) / np.linalg.norm(u_centered)
        
        passed = recon_error < 0.5  # Reasonable reconstruction
        print(f"  L2 reconstruction error: {recon_error:.4f}")
        print(f"  Inverse solver: {'✓' if passed else '✗'}")
        
        return passed
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("INVERSE SOURCE v7.26 - INSTALLATION VERIFICATION")
    print("=" * 60)
    
    print(f"\nPython: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")
    
    # Run all checks
    results = {}
    
    # Basic imports
    results['imports'] = check_imports()
    results['package'] = check_package_imports()
    
    if not results['imports'] or not results['package']:
        print("\n" + "=" * 60)
        print("✗ BASIC IMPORTS FAILED - fix dependencies first")
        print("=" * 60)
        return 1
    
    # v7.26 specific tests
    try:
        results['mesh_refinement'] = test_mesh_refinement()
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        results['mesh_refinement'] = False
    
    try:
        results['sensor_embedding'] = test_sensor_embedding()
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        results['sensor_embedding'] = False
    
    try:
        results['fem_analytical'] = test_fem_analytical_agreement()
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        results['fem_analytical'] = False
    
    try:
        results['greens_sign'] = test_greens_function_sign()
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        results['greens_sign'] = False
    
    try:
        results['convergence'] = test_convergence_rate()
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        results['convergence'] = False
    
    # Regularization tests
    try:
        results['lcurve'] = test_lcurve_analysis()
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        results['lcurve'] = False
    
    try:
        results['inverse'] = test_inverse_solver()
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        results['inverse'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    n_passed = sum(1 for v in results.values() if v)
    n_total = len(results)
    
    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
    
    print(f"\nTests passed: {n_passed}/{n_total}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - v7.26 installation verified!")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nTroubleshooting:")
        print("  1. Make sure gmsh is installed: pip install gmsh")
        print("  2. On Linux: apt-get install libglu1-mesa libxft2")
        print("  3. Reinstall: pip install -e . --force-reinstall")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
