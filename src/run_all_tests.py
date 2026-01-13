#!/usr/bin/env python3
"""
Master Test Runner for Inverse Source Localization Package
============================================================

This script runs all validation tests and produces a comprehensive report.
It can be run in quick mode for smoke testing or full mode for thorough
validation.

Usage:
    # Quick smoke test (< 1 minute)
    python run_all_tests.py --quick
    
    # Full validation (may take 10-30 minutes)
    python run_all_tests.py --full
    
    # Specific tests only
    python run_all_tests.py --test validation
    python run_all_tests.py --test polar
    python run_all_tests.py --test scalability
    
    # Save results to specific directory
    python run_all_tests.py --full --save results/

Author: Claude (Anthropic)
Date: January 2026
"""

import numpy as np
import sys
import os
import time
import argparse
import json
from datetime import datetime

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_quick_sanity_check():
    """
    Quick sanity check that basic functionality works.
    Should complete in < 30 seconds.
    """
    print("\n" + "=" * 70)
    print("QUICK SANITY CHECK")
    print("=" * 70)
    
    errors = []
    
    # Test 1: Analytical forward solver
    print("\n[1/5] Testing analytical forward solver...", end=" ", flush=True)
    try:
        from analytical_solver import AnalyticalForwardSolver
        sources = [((0.5, 0.0), 1.0), ((-0.5, 0.0), -1.0)]
        forward = AnalyticalForwardSolver(n_boundary_points=50)
        u = forward.solve(sources)
        assert len(u) == 50, f"Expected 50 boundary values, got {len(u)}"
        assert abs(np.sum(u * 50 / (2*np.pi))) < 1e-6, "Mean should be ~0"  # Rough check
        print("✓")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        errors.append(f"Analytical forward: {e}")
    
    # Test 2: Analytical nonlinear solver (2 sources, few restarts)
    print("[2/5] Testing analytical nonlinear solver...", end=" ", flush=True)
    try:
        from analytical_solver import AnalyticalNonlinearInverseSolver
        sources_true = [((0.7, 0.0), 1.0), ((-0.7, 0.0), -1.0)]
        forward = AnalyticalForwardSolver(n_boundary_points=100)
        u_measured = forward.solve(sources_true)
        
        inverse = AnalyticalNonlinearInverseSolver(n_sources=2, n_boundary=100)
        inverse.set_measured_data(u_measured)
        result = inverse.solve(method='L-BFGS-B', n_restarts=5, maxiter=200)
        
        # Check reasonable result
        pos_errors = []
        for s in result.sources:
            min_dist = min(np.sqrt((s.x - tx)**2 + (s.y - ty)**2) 
                          for (tx, ty), _ in sources_true)
            pos_errors.append(min_dist)
        avg_error = np.mean(pos_errors)
        
        assert avg_error < 0.1, f"Position error too large: {avg_error}"
        print(f"✓ (pos_err={avg_error:.2e})")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        errors.append(f"Analytical nonlinear: {e}")
    
    # Test 3: Polar solver
    print("[3/5] Testing polar coordinate solver...", end=" ", flush=True)
    try:
        from polar_solver import PolarNonlinearInverseSolver
        sources_true = [((0.7, 0.0), 1.0), ((-0.7, 0.0), -1.0)]
        forward = AnalyticalForwardSolver(n_boundary_points=100)
        u_measured = forward.solve(sources_true)
        
        inverse = PolarNonlinearInverseSolver(n_sources=2, n_boundary=100)
        inverse.set_measured_data(u_measured)
        result = inverse.solve(method='L-BFGS-B', n_restarts=5, maxiter=200)
        
        pos_errors = []
        for s in result.sources:
            min_dist = min(np.sqrt((s.x - tx)**2 + (s.y - ty)**2) 
                          for (tx, ty), _ in sources_true)
            pos_errors.append(min_dist)
        avg_error = np.mean(pos_errors)
        
        assert avg_error < 0.1, f"Position error too large: {avg_error}"
        print(f"✓ (pos_err={avg_error:.2e})")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        errors.append(f"Polar solver: {e}")
    
    # Test 4: Conformal solver (ellipse)
    print("[4/5] Testing conformal solver (ellipse)...", end=" ", flush=True)
    try:
        from conformal_solver import EllipseMap, ConformalForwardSolver
        
        ellipse = EllipseMap(a=2.0, b=1.0)
        forward = ConformalForwardSolver(ellipse, n_boundary=50)
        
        sources = [((-0.8, 0.3), 1.0), ((0.8, -0.3), -1.0)]
        u = forward.solve(sources)
        
        assert len(u) == 50, f"Expected 50 values, got {len(u)}"
        print("✓")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        errors.append(f"Conformal: {e}")
    
    # Test 5: FEM solver (optional - may not have FEniCS)
    print("[5/5] Testing FEM solver...", end=" ", flush=True)
    try:
        from fem_solver import FEMForwardSolver
        
        sources = [((0.5, 0.0), 1.0), ((-0.5, 0.0), -1.0)]
        forward = FEMForwardSolver(resolution=0.1, verbose=False)
        u = forward.solve(sources)
        
        assert len(u) > 0, "FEM should return boundary values"
        print("✓")
    except ImportError:
        print("⊘ SKIPPED (FEniCS not installed)")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        errors.append(f"FEM: {e}")
    
    # Summary
    print("\n" + "-" * 70)
    if errors:
        print(f"SANITY CHECK FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("SANITY CHECK PASSED: All basic tests successful")
        return True


def run_validation_tests(quick: bool = False):
    """Run nonlinear solver validation across domains."""
    print("\n" + "=" * 70)
    print("NONLINEAR SOLVER VALIDATION")
    print("=" * 70)
    
    from test_nonlinear_validation import run_validation_suite, print_summary
    
    results = run_validation_suite(
        domains=['disk', 'ellipse', 'star'],
        quick=quick
    )
    print_summary(results)
    
    passed = sum(1 for r in results if r.position_error < 0.01 and r.success)
    return passed, len(results), results


def run_polar_comparison(n_restarts: int = 20):
    """Compare polar vs Cartesian parameterization."""
    print("\n" + "=" * 70)
    print("POLAR vs CARTESIAN COMPARISON")
    print("=" * 70)
    
    from polar_solver import compare_polar_vs_cartesian
    
    # Test with 4 sources
    sources_true = [
        ((0.7, 0.0), 1.0),
        ((0.0, 0.7), -1.0),
        ((-0.7, 0.0), 1.0),
        ((0.0, -0.7), -1.0),
    ]
    
    results = compare_polar_vs_cartesian(sources_true, n_restarts=n_restarts)
    return results


def run_scalability_tests(quick: bool = False, max_sources: int = 10):
    """Run scalability tests."""
    print("\n" + "=" * 70)
    print("SCALABILITY TESTS")
    print("=" * 70)
    
    from test_scalability import run_scalability_study, print_summary_table
    
    n_sources_list = [4, 6, 8] if quick else [4, 6, 8, 10]
    if max_sources < 10:
        n_sources_list = [n for n in n_sources_list if n <= max_sources]
    
    results = run_scalability_study(
        n_sources_list=n_sources_list,
        quick=quick,
        compare_polar=True,
        test_de=not quick
    )
    print_summary_table(results)
    
    converged = sum(1 for r in results if r.converged)
    return converged, len(results), results


def save_results(output_dir: str, validation_results=None, scalability_results=None, 
                 polar_results=None, elapsed_time: float = 0, mode: str = 'full'):
    """Save all results to files including figures."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save summary JSON
    summary = {
        'timestamp': timestamp,
        'mode': mode,
        'elapsed_seconds': elapsed_time,
    }
    
    # Process validation results
    if validation_results:
        summary['validation'] = {
            'total': len(validation_results),
            'passed': sum(1 for r in validation_results if r.position_error < 0.01 and r.success),
            'results': []
        }
        for r in validation_results:
            summary['validation']['results'].append({
                'domain': r.domain,
                'solver': r.solver,
                'optimizer': r.optimizer,
                'n_sources': int(r.n_sources),
                'position_error': float(r.position_error),
                'intensity_error': float(r.intensity_error),
                'time_seconds': float(r.time_seconds),
                'success': bool(r.success)
            })
        
        # Create validation summary figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Position error by domain/solver
        domains = sorted(list(set(r.domain for r in validation_results)))
        solvers = sorted(list(set(r.solver for r in validation_results)))
        
        ax = axes[0]
        x_positions = np.arange(len(domains))
        n_solvers = len(solvers)
        width = 0.8 / n_solvers  # Dynamically adjust width
        
        for i, solver in enumerate(solvers):  # ALL solvers
            errors = []
            for domain in domains:
                matching = [r for r in validation_results if r.domain == domain and r.solver == solver]
                if matching:
                    errors.append(np.mean([r.position_error for r in matching]))
                else:
                    errors.append(np.nan)
            offset = (i - n_solvers/2 + 0.5) * width
            ax.bar(x_positions + offset, errors, width, label=solver)
        ax.set_xlabel('Domain')
        ax.set_ylabel('Position Error')
        ax.set_title('Position Error by Domain and Solver')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(domains)
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.axhline(y=0.01, color='r', linestyle='--', alpha=0.7, label='Threshold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Time by domain/solver
        ax = axes[1]
        for i, solver in enumerate(solvers):  # ALL solvers
            times = []
            for domain in domains:
                matching = [r for r in validation_results if r.domain == domain and r.solver == solver]
                if matching:
                    times.append(np.mean([r.time_seconds for r in matching]))
                else:
                    times.append(np.nan)
            offset = (i - n_solvers/2 + 0.5) * width
            ax.bar(x_positions + offset, times, width, label=solver)
        ax.set_xlabel('Domain')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Computation Time by Domain and Solver')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(domains)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'validation_results.png'), dpi=150)
        plt.close(fig)
    
    # Process scalability results
    if scalability_results:
        summary['scalability'] = {
            'total': len(scalability_results),
            'converged': sum(1 for r in scalability_results if r.converged),
            'results': []
        }
        for r in scalability_results:
            summary['scalability']['results'].append({
                'n_sources': int(r.n_sources),
                'solver': r.solver,
                'optimizer': r.optimizer,
                'position_error': float(r.position_error),
                'intensity_error': float(r.intensity_error),
                'time_seconds': float(r.time_seconds),
                'converged': bool(r.converged)
            })
        
        # Create scalability figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Position error vs N
        ax = axes[0]
        solvers = sorted(list(set(r.solver for r in scalability_results)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h']
        for i, solver in enumerate(solvers):
            matching = sorted([r for r in scalability_results if r.solver == solver], 
                            key=lambda x: x.n_sources)
            if matching:
                ns = [r.n_sources for r in matching]
                errs = [r.position_error for r in matching]
                marker = markers[i % len(markers)]
                ax.semilogy(ns, errs, f'{marker}-', label=solver, markersize=8)
        ax.set_xlabel('Number of Sources')
        ax.set_ylabel('Position Error')
        ax.set_title('Scalability: Position Error vs N')
        ax.axhline(y=0.01, color='r', linestyle='--', alpha=0.7, label='Convergence threshold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([4, 6, 8, 10])
        
        # Time vs N
        ax = axes[1]
        for i, solver in enumerate(solvers):
            matching = sorted([r for r in scalability_results if r.solver == solver],
                            key=lambda x: x.n_sources)
            if matching:
                ns = [r.n_sources for r in matching]
                times = [r.time_seconds for r in matching]
                marker = markers[i % len(markers)]
                ax.plot(ns, times, f'{marker}-', label=solver, markersize=8)
        ax.set_xlabel('Number of Sources')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Scalability: Computation Time vs N')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([4, 6, 8, 10])
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'scalability_results.png'), dpi=150)
        plt.close(fig)
    
    # Save JSON summary
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save text summary
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(f"Inverse Source Localization Test Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Total time: {elapsed_time:.1f} seconds\n\n")
        
        if validation_results:
            passed = sum(1 for r in validation_results if r.position_error < 0.01 and r.success)
            f.write(f"Validation: {passed}/{len(validation_results)} passed\n")
        
        if scalability_results:
            conv = sum(1 for r in scalability_results if r.converged)
            f.write(f"Scalability: {conv}/{len(scalability_results)} converged\n")
    
    print(f"\nResults saved to {output_dir}/")
    print(f"  - results.json (detailed data)")
    print(f"  - summary.txt (text summary)")
    if validation_results:
        print(f"  - validation_results.png")
    if scalability_results:
        print(f"  - scalability_results.png")


def main():
    parser = argparse.ArgumentParser(description="Run all tests for inverse source package")
    parser.add_argument('--quick', action='store_true', 
                        help='Quick mode - faster but less thorough')
    parser.add_argument('--full', action='store_true',
                        help='Full mode - comprehensive testing')
    parser.add_argument('--test', type=str, default=None,
                        choices=['sanity', 'validation', 'polar', 'scalability', 'all'],
                        help='Run specific test only')
    parser.add_argument('--max-sources', type=int, default=10,
                        help='Maximum sources for scalability test')
    parser.add_argument('--save', type=str, default='results',
                        help='Directory to save results (default: results/)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to files')
    
    args = parser.parse_args()
    
    # Default to quick if neither specified
    if not args.full and not args.quick:
        args.quick = True
    
    if args.test is None:
        args.test = 'all'
    
    print("=" * 70)
    print("INVERSE SOURCE LOCALIZATION - TEST SUITE")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"Tests: {args.test}")
    print("=" * 70)
    
    start_time = time.time()
    summary = {}
    
    # Store results for saving
    validation_results = None
    scalability_results = None
    polar_results = None
    
    # Sanity check (always run first)
    if args.test in ['sanity', 'all']:
        sanity_ok = run_quick_sanity_check()
        summary['sanity'] = 'PASS' if sanity_ok else 'FAIL'
        
        if not sanity_ok and args.test == 'all':
            print("\n⚠ Sanity check failed - stopping further tests")
            return
    
    # Validation tests
    if args.test in ['validation', 'all']:
        passed, total, validation_results = run_validation_tests(quick=args.quick)
        summary['validation'] = f'{passed}/{total}'
    
    # Polar comparison
    if args.test in ['polar', 'all']:
        n_restarts = 10 if args.quick else 20
        polar_results = run_polar_comparison(n_restarts=n_restarts)
        summary['polar'] = 'DONE'
    
    # Scalability tests
    if args.test in ['scalability', 'all']:
        converged, total, scalability_results = run_scalability_tests(
            quick=args.quick, 
            max_sources=args.max_sources
        )
        summary['scalability'] = f'{converged}/{total}'
    
    # Final summary
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print()
    for test, result in summary.items():
        print(f"  {test}: {result}")
    print("=" * 70)
    
    # Save results (default: yes)
    if not args.no_save:
        save_results(
            output_dir=args.save,
            validation_results=validation_results,
            scalability_results=scalability_results,
            polar_results=polar_results,
            elapsed_time=elapsed,
            mode='QUICK' if args.quick else 'FULL'
        )


if __name__ == "__main__":
    main()
