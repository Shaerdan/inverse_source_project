#!/usr/bin/env python3
"""
Test Script 4: Scalability Test for 6-10 Source Recovery
=========================================================

This script tests the nonlinear inverse solvers' ability to recover
larger numbers of sources (6, 8, 10). Key questions:

1. How does position error scale with number of sources?
2. How many restarts/iterations are needed for reliable recovery?
3. How does computation time scale?
4. Does polar parameterization help for larger source counts?

For reliable recovery of N sources, we generally need:
- Well-separated sources (angular separation > 2π/N)
- More restarts as N increases
- Potentially global optimizers for N > 6

Usage:
    python test_scalability.py
    python test_scalability.py --quick          # Fewer restarts
    python test_scalability.py --max-sources 8  # Test up to 8 sources
    python test_scalability.py --compare-polar  # Compare polar vs Cartesian

Author: Claude (Anthropic)
Date: January 2026
"""

import numpy as np
import time
import sys
import argparse
from typing import List, Tuple, Dict
from dataclasses import dataclass, asdict
import json

# Add parent to path if running as script
if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ScalabilityResult:
    """Results from a scalability test."""
    n_sources: int
    solver: str
    optimizer: str
    n_restarts: int
    position_error: float
    intensity_error: float
    residual: float
    time_seconds: float
    success: bool
    converged: bool  # Did we achieve error < threshold?
    details: str = ""


def create_well_separated_sources(n_sources: int, radius: float = 0.7, 
                                   seed: int = 42) -> List[Tuple[Tuple[float, float], float]]:
    """
    Create n well-separated sources on a circle.
    
    For N sources, angular separation is 2π/N.
    Intensities alternate +1/-1 with small random perturbation.
    Zero-sum enforced.
    """
    np.random.seed(seed)
    sources = []
    
    # Base angles with small random perturbation
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    max_perturbation = 0.3 * (2*np.pi / n_sources)  # Max 30% of angular spacing
    angles += np.random.uniform(-max_perturbation, max_perturbation, n_sources)
    
    # Radii with small variation
    radii = radius + 0.05 * np.random.randn(n_sources)
    radii = np.clip(radii, 0.5, 0.85)  # Keep in safe region
    
    # Alternating intensities with small variation
    for i in range(n_sources):
        x = radii[i] * np.cos(angles[i])
        y = radii[i] * np.sin(angles[i])
        base_intensity = 1.0 if i % 2 == 0 else -1.0
        intensity = base_intensity * (0.8 + 0.4 * np.random.rand())  # Scale by 0.8-1.2
        sources.append(((x, y), intensity))
    
    # Enforce zero sum
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    
    return sources


def compute_errors(recovered_sources, true_sources) -> Tuple[float, float]:
    """
    Compute position and intensity errors using optimal matching.
    """
    from scipy.optimize import linear_sum_assignment
    
    n_true = len(true_sources)
    n_rec = len(recovered_sources)
    
    if n_rec == 0:
        return float('inf'), float('inf')
    
    # Build cost matrix (position-based)
    cost = np.zeros((n_true, n_rec))
    for i, ((tx, ty), _) in enumerate(true_sources):
        for j, rec in enumerate(recovered_sources):
            if hasattr(rec, 'x'):
                rx, ry = rec.x, rec.y
            else:
                (rx, ry), _ = rec
            cost[i, j] = np.sqrt((tx - rx)**2 + (ty - ry)**2)
    
    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    pos_error = cost[row_ind, col_ind].mean()
    
    # Intensity error for matched pairs
    int_errors = []
    for i, j in zip(row_ind, col_ind):
        true_int = true_sources[i][1]
        if hasattr(recovered_sources[j], 'intensity'):
            rec_int = recovered_sources[j].intensity
        else:
            rec_int = recovered_sources[j][1]
        int_errors.append(abs(true_int - rec_int))
    
    int_error = np.mean(int_errors)
    
    return pos_error, int_error


def test_analytical_cartesian(sources_true: list, optimizer: str = 'L-BFGS-B',
                               n_restarts: int = 20, maxiter: int = 500) -> ScalabilityResult:
    """Test analytical solver with Cartesian parameterization."""
    from analytical_solver import AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver
    
    n_sources = len(sources_true)
    start_time = time.time()
    
    try:
        # Generate measurement data
        forward = AnalyticalForwardSolver(n_boundary_points=100)
        u_measured = forward.solve(sources_true)
        
        # Solve inverse problem
        inverse = AnalyticalNonlinearInverseSolver(n_sources=n_sources, n_boundary=100)
        inverse.set_measured_data(u_measured)
        result = inverse.solve(method=optimizer, maxiter=maxiter, n_restarts=n_restarts)
        
        elapsed = time.time() - start_time
        
        recovered = [((s.x, s.y), s.intensity) for s in result.sources]
        pos_err, int_err = compute_errors(recovered, sources_true)
        
        return ScalabilityResult(
            n_sources=n_sources,
            solver='analytical_cartesian',
            optimizer=optimizer,
            n_restarts=n_restarts,
            position_error=pos_err,
            intensity_error=int_err,
            residual=result.residual,
            time_seconds=elapsed,
            success=result.success,
            converged=pos_err < 0.01,
            details=f"iterations={result.iterations}"
        )
    except Exception as e:
        import traceback
        return ScalabilityResult(
            n_sources=n_sources,
            solver='analytical_cartesian',
            optimizer=optimizer,
            n_restarts=n_restarts,
            position_error=float('inf'),
            intensity_error=float('inf'),
            residual=float('inf'),
            time_seconds=time.time() - start_time,
            success=False,
            converged=False,
            details=f"ERROR: {str(e)}\n{traceback.format_exc()}"
        )


def test_analytical_polar(sources_true: list, optimizer: str = 'L-BFGS-B',
                          n_restarts: int = 20, maxiter: int = 500) -> ScalabilityResult:
    """Test analytical solver with polar parameterization."""
    start_time = time.time()
    
    try:
        from analytical_solver import AnalyticalForwardSolver
        from polar_solver import PolarNonlinearInverseSolver
        
        n_sources = len(sources_true)
        
        # Generate measurement data
        forward = AnalyticalForwardSolver(n_boundary_points=100)
        u_measured = forward.solve(sources_true)
        
        # Solve inverse problem
        inverse = PolarNonlinearInverseSolver(n_sources=n_sources, n_boundary=100)
        inverse.set_measured_data(u_measured)
        result = inverse.solve(method=optimizer, maxiter=maxiter, n_restarts=n_restarts)
        
        elapsed = time.time() - start_time
        
        recovered = [((s.x, s.y), s.intensity) for s in result.sources]
        pos_err, int_err = compute_errors(recovered, sources_true)
        
        return ScalabilityResult(
            n_sources=n_sources,
            solver='analytical_polar',
            optimizer=optimizer,
            n_restarts=n_restarts,
            position_error=pos_err,
            intensity_error=int_err,
            residual=result.residual,
            time_seconds=elapsed,
            success=result.success,
            converged=pos_err < 0.01,
            details=f"iterations={result.iterations}"
        )
    except Exception as e:
        import traceback
        return ScalabilityResult(
            n_sources=n_sources,
            solver='analytical_polar',
            optimizer=optimizer,
            n_restarts=n_restarts,
            position_error=float('inf'),
            intensity_error=float('inf'),
            residual=float('inf'),
            time_seconds=time.time() - start_time,
            success=False,
            converged=False,
            details=f"ERROR: {str(e)}\n{traceback.format_exc()}"
        )


def test_differential_evolution(sources_true: list, solver_type: str = 'cartesian',
                                 maxiter: int = 500) -> ScalabilityResult:
    """Test with differential evolution (global optimizer)."""
    start_time = time.time()
    
    try:
        from analytical_solver import AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver
        from polar_solver import PolarNonlinearInverseSolver
        
        n_sources = len(sources_true)
        
        # Generate measurement data
        forward = AnalyticalForwardSolver(n_boundary_points=100)
        u_measured = forward.solve(sources_true)
        
        # Solve inverse problem
        if solver_type == 'polar':
            inverse = PolarNonlinearInverseSolver(n_sources=n_sources, n_boundary=100)
            solver_name = 'polar_DE'
        else:
            inverse = AnalyticalNonlinearInverseSolver(n_sources=n_sources, n_boundary=100)
            solver_name = 'cartesian_DE'
        
        inverse.set_measured_data(u_measured)
        result = inverse.solve(method='differential_evolution', maxiter=maxiter)
        
        elapsed = time.time() - start_time
        
        recovered = [((s.x, s.y), s.intensity) for s in result.sources]
        pos_err, int_err = compute_errors(recovered, sources_true)
        
        return ScalabilityResult(
            n_sources=n_sources,
            solver=solver_name,
            optimizer='differential_evolution',
            n_restarts=1,
            position_error=pos_err,
            intensity_error=int_err,
            residual=result.residual,
            time_seconds=elapsed,
            success=result.success,
            converged=pos_err < 0.01,
            details=f"maxiter={maxiter}"
        )
    except Exception as e:
        import traceback
        return ScalabilityResult(
            n_sources=n_sources,
            solver=solver_type + '_DE',
            optimizer='differential_evolution',
            n_restarts=1,
            position_error=float('inf'),
            intensity_error=float('inf'),
            residual=float('inf'),
            time_seconds=time.time() - start_time,
            success=False,
            converged=False,
            details=f"ERROR: {str(e)}\n{traceback.format_exc()}"
        )


def run_scalability_study(n_sources_list: list = None,
                          n_restarts_base: int = 20,
                          quick: bool = False,
                          compare_polar: bool = False,
                          test_de: bool = False,
                          seeds: list = None,
                          verbose: bool = True) -> List[ScalabilityResult]:
    """
    Run scalability study across different source counts.
    
    Parameters
    ----------
    n_sources_list : list
        Numbers of sources to test. Default: [4, 6, 8, 10]
    n_restarts_base : int
        Base number of restarts (scaled by n_sources)
    quick : bool
        Use fewer restarts for quick testing
    compare_polar : bool
        Also test polar parameterization
    test_de : bool
        Also test differential evolution
    seeds : list
        Random seeds for multiple trials per n_sources
    verbose : bool
        Print progress
        
    Returns
    -------
    results : list of ScalabilityResult
    """
    if n_sources_list is None:
        n_sources_list = [4, 6, 8, 10]
    
    if seeds is None:
        seeds = [42]  # Single trial by default
    
    results = []
    
    print("=" * 80)
    print("SCALABILITY STUDY: Source Recovery Performance")
    print("=" * 80)
    print(f"Source counts: {n_sources_list}")
    print(f"Base restarts: {n_restarts_base}")
    print(f"Quick mode: {quick}")
    print(f"Compare polar: {compare_polar}")
    print(f"Test diff. evolution: {test_de}")
    print(f"Seeds (trials per config): {seeds}")
    print("=" * 80)
    
    for n_sources in n_sources_list:
        print(f"\n{'='*60}")
        print(f"Testing N = {n_sources} sources")
        print(f"{'='*60}")
        
        # Scale restarts with n_sources
        n_restarts = n_restarts_base
        if not quick:
            n_restarts = max(n_restarts_base, n_sources * 5)  # More restarts for more sources
        else:
            n_restarts = max(5, n_sources)
        
        for seed in seeds:
            # Create test sources
            sources_true = create_well_separated_sources(n_sources, radius=0.7, seed=seed)
            
            if verbose:
                print(f"\nSeed={seed}, {n_sources} sources:")
                for i, ((x, y), q) in enumerate(sources_true):
                    r = np.sqrt(x**2 + y**2)
                    theta = np.arctan2(y, x)
                    print(f"  {i+1}: ({x:+.4f}, {y:+.4f}), r={r:.3f}, θ={np.degrees(theta):+.1f}°, q={q:+.4f}")
            
            # Test 1: Cartesian with L-BFGS-B
            if verbose:
                print(f"\n[1] Cartesian + L-BFGS-B ({n_restarts} restarts)...")
            result = test_analytical_cartesian(sources_true, optimizer='L-BFGS-B',
                                               n_restarts=n_restarts)
            results.append(result)
            
            if verbose:
                status = "✓" if result.converged else "✗"
                print(f"    {status} pos_err={result.position_error:.2e}, "
                      f"int_err={result.intensity_error:.2e}, "
                      f"time={result.time_seconds:.1f}s")
            
            # Test 2: Polar with L-BFGS-B
            if compare_polar:
                if verbose:
                    print(f"\n[2] Polar + L-BFGS-B ({n_restarts} restarts)...")
                result = test_analytical_polar(sources_true, optimizer='L-BFGS-B',
                                               n_restarts=n_restarts)
                results.append(result)
                
                if verbose:
                    status = "✓" if result.converged else "✗"
                    print(f"    {status} pos_err={result.position_error:.2e}, "
                          f"int_err={result.intensity_error:.2e}, "
                          f"time={result.time_seconds:.1f}s")
            
            # Test 3: Differential Evolution (scale maxiter with problem size)
            if test_de and not quick:
                # DE needs more iterations for larger problems
                # Rule of thumb: 200 * n_dimensions = 200 * (3n - 1)
                de_maxiter = max(500, 200 * (3 * n_sources - 1))
                if verbose:
                    print(f"\n[3] Differential Evolution (maxiter={de_maxiter})...")
                result = test_differential_evolution(sources_true, solver_type='cartesian',
                                                     maxiter=de_maxiter)
                results.append(result)
                
                if verbose:
                    status = "✓" if result.converged else "✗"
                    print(f"    {status} pos_err={result.position_error:.2e}, "
                          f"int_err={result.intensity_error:.2e}, "
                          f"time={result.time_seconds:.1f}s")
    
    return results


def print_summary_table(results: List[ScalabilityResult]):
    """Print summary table."""
    print("\n" + "=" * 110)
    print("SCALABILITY SUMMARY")
    print("=" * 110)
    print(f"{'N':<4} {'Solver':<25} {'Optimizer':<15} {'Restarts':<10} "
          f"{'Pos Err':<12} {'Int Err':<12} {'Time (s)':<10} {'Status':<10}")
    print("-" * 110)
    
    for r in sorted(results, key=lambda x: (x.n_sources, x.solver)):
        status = "CONVERGED" if r.converged else ("FAIL" if r.success else "ERROR")
        pos_str = f"{r.position_error:.2e}" if r.position_error < float('inf') else "N/A"
        int_str = f"{r.intensity_error:.2e}" if r.intensity_error < float('inf') else "N/A"
        
        print(f"{r.n_sources:<4} {r.solver:<25} {r.optimizer:<15} {r.n_restarts:<10} "
              f"{pos_str:<12} {int_str:<12} {r.time_seconds:<10.1f} {status:<10}")
    
    print("=" * 110)
    
    # Aggregate statistics by n_sources
    print("\nAGGREGATE BY SOURCE COUNT:")
    print("-" * 60)
    
    for n in sorted(set(r.n_sources for r in results)):
        subset = [r for r in results if r.n_sources == n]
        converged = sum(1 for r in subset if r.converged)
        avg_pos_err = np.mean([r.position_error for r in subset if r.position_error < float('inf')])
        avg_time = np.mean([r.time_seconds for r in subset])
        
        print(f"N={n}: {converged}/{len(subset)} converged, "
              f"avg pos_err={avg_pos_err:.2e}, avg time={avg_time:.1f}s")


def run_restart_sensitivity(n_sources: int = 6, 
                             restart_counts: list = None,
                             n_trials: int = 5,
                             verbose: bool = True) -> Dict:
    """
    Study how recovery improves with more restarts.
    
    For a fixed n_sources, test multiple restart counts and measure:
    - Success rate (converged < 0.01 error)
    - Average position error
    - Computation time
    """
    if restart_counts is None:
        restart_counts = [5, 10, 20, 50, 100]
    
    results = {n: {'errors': [], 'times': [], 'converged': 0} for n in restart_counts}
    
    print(f"\nRESTART SENSITIVITY STUDY (n_sources={n_sources})")
    print("=" * 60)
    
    from analytical_solver import AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver
    
    for trial in range(n_trials):
        # Create test sources
        sources_true = create_well_separated_sources(n_sources, seed=42 + trial)
        
        # Generate measurements
        forward = AnalyticalForwardSolver(n_boundary_points=100)
        u_measured = forward.solve(sources_true)
        
        if verbose:
            print(f"\nTrial {trial+1}/{n_trials}")
        
        for n_restarts in restart_counts:
            t0 = time.time()
            
            inverse = AnalyticalNonlinearInverseSolver(n_sources=n_sources, n_boundary=100)
            inverse.set_measured_data(u_measured)
            result = inverse.solve(method='L-BFGS-B', n_restarts=n_restarts, maxiter=500)
            
            elapsed = time.time() - t0
            
            recovered = [((s.x, s.y), s.intensity) for s in result.sources]
            pos_err, _ = compute_errors(recovered, sources_true)
            
            results[n_restarts]['errors'].append(pos_err)
            results[n_restarts]['times'].append(elapsed)
            if pos_err < 0.01:
                results[n_restarts]['converged'] += 1
            
            if verbose:
                status = "✓" if pos_err < 0.01 else "✗"
                print(f"  {status} restarts={n_restarts}: pos_err={pos_err:.2e}, time={elapsed:.1f}s")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESTART SENSITIVITY SUMMARY")
    print("=" * 60)
    print(f"{'Restarts':<12} {'Success Rate':<15} {'Avg Pos Err':<15} {'Avg Time (s)':<12}")
    print("-" * 60)
    
    for n_restarts in restart_counts:
        data = results[n_restarts]
        success_rate = data['converged'] / n_trials
        avg_err = np.mean(data['errors'])
        avg_time = np.mean(data['times'])
        
        print(f"{n_restarts:<12} {success_rate*100:>6.1f}%{'':<8} {avg_err:<15.2e} {avg_time:<12.1f}")
    
    print("=" * 60)
    
    return results


def save_results(results: List[ScalabilityResult], filename: str = "scalability_results.json"):
    """Save results to JSON."""
    data = [asdict(r) for r in results]
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test scalability of source recovery")
    parser.add_argument('--quick', action='store_true', 
                        help='Fewer restarts for quick testing')
    parser.add_argument('--max-sources', type=int, default=10,
                        help='Maximum number of sources to test')
    parser.add_argument('--min-sources', type=int, default=4,
                        help='Minimum number of sources to test')
    parser.add_argument('--compare-polar', action='store_true',
                        help='Also test polar parameterization')
    parser.add_argument('--test-de', action='store_true',
                        help='Also test differential evolution')
    parser.add_argument('--restart-study', action='store_true',
                        help='Run restart sensitivity study')
    parser.add_argument('--n-trials', type=int, default=3,
                        help='Number of trials per configuration')
    parser.add_argument('--save', type=str, default=None,
                        help='Save results to JSON file')
    
    args = parser.parse_args()
    
    if args.restart_study:
        # Run restart sensitivity study
        run_restart_sensitivity(n_sources=6, n_trials=args.n_trials)
    else:
        # Run main scalability study
        n_sources_list = list(range(args.min_sources, args.max_sources + 1, 2))
        seeds = list(range(42, 42 + args.n_trials))
        
        results = run_scalability_study(
            n_sources_list=n_sources_list,
            quick=args.quick,
            compare_polar=args.compare_polar,
            test_de=args.test_de,
            seeds=seeds
        )
        
        print_summary_table(results)
        
        if args.save:
            save_results(results, args.save)
