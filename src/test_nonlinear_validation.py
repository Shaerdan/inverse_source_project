#!/usr/bin/env python3
"""
Test Script 1: Nonlinear Solver Validation Across Domains
==========================================================

This script validates the nonlinear inverse solvers across different domains:
- Disk (Analytical)
- Disk (FEM) 
- Ellipse (Conformal)
- Ellipse (FEM)
- Star (Conformal)

For each domain, we test with well-separated sources at various configurations
and measure position recovery error.

Expected results (with well-separated sources, no noise):
- Position error < 1e-4 for 2-4 sources
- Position error < 1e-2 for 6 sources (may need more restarts)

Usage:
    python test_nonlinear_validation.py
    python test_nonlinear_validation.py --quick  # Skip slow differential_evolution
    python test_nonlinear_validation.py --domain disk  # Test only disk domain

Author: Claude (Anthropic)
Date: January 2026
"""

import numpy as np
import time
import sys
import argparse
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Add parent to path if running as script
if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ValidationResult:
    """Results from a single validation test."""
    domain: str
    solver: str
    optimizer: str
    n_sources: int
    position_error: float
    intensity_error: float
    residual: float
    time_seconds: float
    success: bool
    details: str = ""


def create_well_separated_sources_disk(n_sources: int, radius: float = 0.7, 
                                        seed: int = 42) -> List[Tuple[Tuple[float, float], float]]:
    """
    Create well-separated sources on a circle for disk domain.
    Intensities alternate +1/-1 with zero sum enforced.
    """
    np.random.seed(seed)
    sources = []
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    # Small random perturbation to angles
    angles += np.random.uniform(-0.1, 0.1, n_sources)
    
    for i, theta in enumerate(angles):
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        intensity = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), intensity))
    
    # Adjust last source for zero sum
    total = sum(s[1] for s in sources)
    if abs(total) > 1e-10:
        sources[-1] = (sources[-1][0], sources[-1][1] - total)
    
    return sources


def create_well_separated_sources_ellipse(n_sources: int, a: float = 2.0, b: float = 1.0,
                                           seed: int = 42) -> List[Tuple[Tuple[float, float], float]]:
    """Create well-separated sources inside ellipse."""
    np.random.seed(seed)
    sources = []
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.1, 0.1, n_sources)
    
    # Place at 60% of ellipse boundary
    for i, theta in enumerate(angles):
        # Ellipse parametric: (a*cos(t), b*sin(t))
        # Scale to be well inside
        scale = 0.6
        x = scale * a * np.cos(theta)
        y = scale * b * np.sin(theta)
        intensity = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), intensity))
    
    total = sum(s[1] for s in sources)
    if abs(total) > 1e-10:
        sources[-1] = (sources[-1][0], sources[-1][1] - total)
    
    return sources


def create_well_separated_sources_star(n_sources: int, seed: int = 42) -> List[Tuple[Tuple[float, float], float]]:
    """Create well-separated sources inside star domain (safe inner region r < 0.4)."""
    np.random.seed(seed)
    sources = []
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.1, 0.1, n_sources)
    
    # Star inner radius is ~0.4, place sources at r=0.25
    radius = 0.25
    for i, theta in enumerate(angles):
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        intensity = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), intensity))
    
    total = sum(s[1] for s in sources)
    if abs(total) > 1e-10:
        sources[-1] = (sources[-1][0], sources[-1][1] - total)
    
    return sources


def compute_position_error(recovered_sources, true_sources) -> float:
    """
    Compute mean position error using optimal matching (Hungarian algorithm).
    """
    from scipy.optimize import linear_sum_assignment
    
    n_true = len(true_sources)
    n_rec = len(recovered_sources)
    
    if n_rec == 0:
        return float('inf')
    
    # Build cost matrix
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
    
    return cost[row_ind, col_ind].mean()


def compute_intensity_error(recovered_sources, true_sources) -> float:
    """Compute intensity error after optimal position matching."""
    from scipy.optimize import linear_sum_assignment
    
    n_true = len(true_sources)
    n_rec = len(recovered_sources)
    
    if n_rec == 0:
        return float('inf')
    
    # Match by position first
    cost = np.zeros((n_true, n_rec))
    for i, ((tx, ty), _) in enumerate(true_sources):
        for j, rec in enumerate(recovered_sources):
            if hasattr(rec, 'x'):
                rx, ry = rec.x, rec.y
            else:
                (rx, ry), _ = rec
            cost[i, j] = np.sqrt((tx - rx)**2 + (ty - ry)**2)
    
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Compute intensity error for matched pairs
    errors = []
    for i, j in zip(row_ind, col_ind):
        true_int = true_sources[i][1]
        if hasattr(recovered_sources[j], 'intensity'):
            rec_int = recovered_sources[j].intensity
        else:
            rec_int = recovered_sources[j][1]
        errors.append(abs(true_int - rec_int))
    
    return np.mean(errors)


# =============================================================================
# DOMAIN-SPECIFIC TEST FUNCTIONS
# =============================================================================

def test_disk_analytical(sources_true: list, optimizer: str = 'L-BFGS-B',
                         n_restarts: int = 10, maxiter: int = 500) -> ValidationResult:
    """Test analytical nonlinear solver on disk domain."""
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
        
        # Convert to list format for error computation
        recovered = [((s.x, s.y), s.intensity) for s in result.sources]
        pos_err = compute_position_error(recovered, sources_true)
        int_err = compute_intensity_error(recovered, sources_true)
        
        return ValidationResult(
            domain='disk',
            solver='analytical',
            optimizer=optimizer,
            n_sources=n_sources,
            position_error=pos_err,
            intensity_error=int_err,
            residual=result.residual,
            time_seconds=elapsed,
            success=result.success,
            details=f"iterations={result.iterations}"
        )
    except Exception as e:
        return ValidationResult(
            domain='disk',
            solver='analytical',
            optimizer=optimizer,
            n_sources=n_sources,
            position_error=float('inf'),
            intensity_error=float('inf'),
            residual=float('inf'),
            time_seconds=time.time() - start_time,
            success=False,
            details=f"ERROR: {str(e)}"
        )


def test_disk_fem(sources_true: list, optimizer: str = 'L-BFGS-B',
                  n_restarts: int = 10, maxiter: int = 500) -> ValidationResult:
    """Test FEM nonlinear solver on disk domain."""
    start_time = time.time()
    
    try:
        from fem_solver import FEMForwardSolver, FEMNonlinearInverseSolver
        
        n_sources = len(sources_true)
        
        # Generate measurement data using FEM forward solver
        forward = FEMForwardSolver(resolution=0.05, verbose=False)
        u_measured = forward.solve(sources_true)
        
        # Solve inverse problem - FEM inverse solver creates its own forward solver
        inverse = FEMNonlinearInverseSolver(
            n_sources=n_sources,
            resolution=0.05,
            verbose=False
        )
        inverse.set_measured_data(u_measured)
        result = inverse.solve(method=optimizer, maxiter=maxiter, n_restarts=n_restarts)
        
        elapsed = time.time() - start_time
        
        recovered = [((s.x, s.y), s.intensity) for s in result.sources]
        pos_err = compute_position_error(recovered, sources_true)
        int_err = compute_intensity_error(recovered, sources_true)
        
        return ValidationResult(
            domain='disk',
            solver='FEM',
            optimizer=optimizer,
            n_sources=n_sources,
            position_error=pos_err,
            intensity_error=int_err,
            residual=result.residual,
            time_seconds=elapsed,
            success=result.success,
            details=f"iterations={result.iterations}"
        )
    except ImportError as e:
        return ValidationResult(
            domain='disk',
            solver='FEM',
            optimizer=optimizer,
            n_sources=len(sources_true),
            position_error=float('inf'),
            intensity_error=float('inf'),
            residual=float('inf'),
            time_seconds=time.time() - start_time,
            success=False,
            details=f"IMPORT ERROR (FEniCS not installed?): {str(e)}"
        )
    except Exception as e:
        return ValidationResult(
            domain='disk',
            solver='FEM',
            optimizer=optimizer,
            n_sources=len(sources_true),
            position_error=float('inf'),
            intensity_error=float('inf'),
            residual=float('inf'),
            time_seconds=time.time() - start_time,
            success=False,
            details=f"ERROR: {str(e)}"
        )


def test_ellipse_conformal(sources_true: list, a: float = 2.0, b: float = 1.0,
                           optimizer: str = 'differential_evolution',
                           maxiter: int = 200) -> ValidationResult:
    """Test conformal nonlinear solver on ellipse domain."""
    start_time = time.time()
    
    try:
        from conformal_solver import EllipseMap, ConformalForwardSolver, ConformalNonlinearInverseSolver
        
        n_sources = len(sources_true)
        
        # Create conformal map
        ellipse_map = EllipseMap(a=a, b=b)
        
        # Generate measurement data
        forward = ConformalForwardSolver(ellipse_map, n_boundary=100)
        u_measured = forward.solve(sources_true)
        
        # Solve inverse problem
        inverse = ConformalNonlinearInverseSolver(ellipse_map, n_sources=n_sources, n_boundary=100)
        sources_rec, residual = inverse.solve(u_measured, method=optimizer)
        
        elapsed = time.time() - start_time
        
        pos_err = compute_position_error(sources_rec, sources_true)
        int_err = compute_intensity_error(sources_rec, sources_true)
        
        return ValidationResult(
            domain='ellipse',
            solver='conformal',
            optimizer=optimizer,
            n_sources=n_sources,
            position_error=pos_err,
            intensity_error=int_err,
            residual=residual,
            time_seconds=elapsed,
            success=True,
            details=f"a={a}, b={b}"
        )
    except Exception as e:
        return ValidationResult(
            domain='ellipse',
            solver='conformal',
            optimizer=optimizer,
            n_sources=len(sources_true),
            position_error=float('inf'),
            intensity_error=float('inf'),
            residual=float('inf'),
            time_seconds=time.time() - start_time,
            success=False,
            details=f"ERROR: {str(e)}"
        )


def test_ellipse_fem(sources_true: list, a: float = 2.0, b: float = 1.0,
                     optimizer: str = 'L-BFGS-B', n_restarts: int = 10,
                     maxiter: int = 500) -> ValidationResult:
    """Test FEM nonlinear solver on ellipse domain."""
    start_time = time.time()
    
    try:
        from fem_solver import FEMNonlinearInverseSolver
        
        n_sources = len(sources_true)
        
        # Create inverse solver using factory method (handles mesh creation)
        inverse = FEMNonlinearInverseSolver.from_ellipse(
            a=a, b=b, 
            n_sources=n_sources,
            resolution=0.1,
            verbose=False
        )
        
        # Generate measurement data using the inverse solver's internal forward solver
        # This ensures consistent sensor locations
        u_measured = inverse.forward.solve(sources_true)
        
        # Solve inverse problem
        inverse.set_measured_data(u_measured)
        result = inverse.solve(method=optimizer, maxiter=maxiter, n_restarts=n_restarts)
        
        elapsed = time.time() - start_time
        
        recovered = [((s.x, s.y), s.intensity) for s in result.sources]
        pos_err = compute_position_error(recovered, sources_true)
        int_err = compute_intensity_error(recovered, sources_true)
        
        return ValidationResult(
            domain='ellipse',
            solver='FEM',
            optimizer=optimizer,
            n_sources=n_sources,
            position_error=pos_err,
            intensity_error=int_err,
            residual=result.residual,
            time_seconds=elapsed,
            success=result.success,
            details=f"a={a}, b={b}"
        )
    except ImportError as e:
        return ValidationResult(
            domain='ellipse',
            solver='FEM',
            optimizer=optimizer,
            n_sources=len(sources_true),
            position_error=float('inf'),
            intensity_error=float('inf'),
            residual=float('inf'),
            time_seconds=time.time() - start_time,
            success=False,
            details=f"IMPORT ERROR: {str(e)}"
        )
    except Exception as e:
        import traceback
        return ValidationResult(
            domain='ellipse',
            solver='FEM',
            optimizer=optimizer,
            n_sources=len(sources_true),
            position_error=float('inf'),
            intensity_error=float('inf'),
            residual=float('inf'),
            time_seconds=time.time() - start_time,
            success=False,
            details=f"ERROR: {str(e)}\n{traceback.format_exc()}"
        )


def test_star_conformal(sources_true: list, optimizer: str = 'differential_evolution',
                        maxiter: int = 200) -> ValidationResult:
    """Test conformal nonlinear solver on star domain."""
    start_time = time.time()
    
    try:
        from conformal_solver import ConformalForwardSolver, ConformalNonlinearInverseSolver
        
        # Try to import StarShapedMap or NumericalConformalMap
        try:
            from conformal_solver import StarShapedMap
            
            def star_radius(theta):
                return 0.7 + 0.3 * np.cos(5 * theta)
            
            star_map = StarShapedMap(star_radius, n_terms=32)
        except ImportError:
            # StarShapedMap not implemented - try NumericalConformalMap
            try:
                from conformal_solver import NumericalConformalMap
                
                def star_boundary(t):
                    """Parametric star boundary: t in [0, 2π]"""
                    r = 0.7 + 0.3 * np.cos(5 * t)
                    return r * np.exp(1j * t)
                
                star_map = NumericalConformalMap(star_boundary, n_boundary=256)
            except Exception as e:
                return ValidationResult(
                    domain='star',
                    solver='conformal',
                    optimizer=optimizer,
                    n_sources=len(sources_true),
                    position_error=float('inf'),
                    intensity_error=float('inf'),
                    residual=float('inf'),
                    time_seconds=time.time() - start_time,
                    success=False,
                    converged=False,
                    details=f"Star domain not supported: {e}"
                )
        
        n_sources = len(sources_true)
        
        # Generate measurement data
        forward = ConformalForwardSolver(star_map, n_boundary=100)
        u_measured = forward.solve(sources_true)
        
        # Solve inverse problem
        inverse = ConformalNonlinearInverseSolver(star_map, n_sources=n_sources, n_boundary=100)
        sources_rec, residual = inverse.solve(u_measured, method=optimizer)
        
        elapsed = time.time() - start_time
        
        pos_err = compute_position_error(sources_rec, sources_true)
        int_err = compute_intensity_error(sources_rec, sources_true)
        
        return ValidationResult(
            domain='star',
            solver='conformal',
            optimizer=optimizer,
            n_sources=n_sources,
            position_error=pos_err,
            intensity_error=int_err,
            residual=residual,
            time_seconds=elapsed,
            success=True,
            details="5-pointed star"
        )
    except Exception as e:
        import traceback
        return ValidationResult(
            domain='star',
            solver='conformal',
            optimizer=optimizer,
            n_sources=len(sources_true),
            position_error=float('inf'),
            intensity_error=float('inf'),
            residual=float('inf'),
            time_seconds=time.time() - start_time,
            success=False,
            details=f"ERROR: {str(e)}\n{traceback.format_exc()}"
        )


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_validation_suite(domains: list = None, quick: bool = False, 
                         verbose: bool = True) -> List[ValidationResult]:
    """
    Run full validation suite across domains.
    
    Parameters
    ----------
    domains : list
        List of domains to test. Default: ['disk', 'ellipse', 'star']
    quick : bool
        If True, skip slow differential_evolution tests
    verbose : bool
        Print progress
    
    Returns
    -------
    results : list of ValidationResult
    """
    if domains is None:
        domains = ['disk', 'ellipse', 'star']
    
    results = []
    
    # Test configurations
    n_sources_list = [2, 4]  # Start with easier cases
    optimizers = ['L-BFGS-B'] if quick else ['L-BFGS-B', 'differential_evolution']
    
    print("=" * 80)
    print("NONLINEAR SOLVER VALIDATION SUITE")
    print("=" * 80)
    print(f"Domains: {domains}")
    print(f"Source counts: {n_sources_list}")
    print(f"Optimizers: {optimizers}")
    print(f"Quick mode: {quick}")
    print("=" * 80)
    
    for domain in domains:
        print(f"\n{'='*40}")
        print(f"DOMAIN: {domain.upper()}")
        print(f"{'='*40}")
        
        for n_sources in n_sources_list:
            # Create appropriate sources for domain
            if domain == 'disk':
                sources_true = create_well_separated_sources_disk(n_sources)
            elif domain == 'ellipse':
                sources_true = create_well_separated_sources_ellipse(n_sources)
            elif domain == 'star':
                sources_true = create_well_separated_sources_star(n_sources)
            else:
                print(f"Unknown domain: {domain}")
                continue
            
            if verbose:
                print(f"\n--- {n_sources} sources ---")
                print("True sources:")
                for i, ((x, y), q) in enumerate(sources_true):
                    print(f"  {i+1}: ({x:+.4f}, {y:+.4f}), q={q:+.4f}")
            
            for optimizer in optimizers:
                if verbose:
                    print(f"\nTesting {domain}/{optimizer}...", end=" ", flush=True)
                
                # Run appropriate test
                if domain == 'disk':
                    # Test both analytical and FEM
                    result = test_disk_analytical(sources_true, optimizer=optimizer,
                                                  n_restarts=10 if optimizer == 'L-BFGS-B' else 1)
                    results.append(result)
                    
                    if verbose:
                        status = "✓" if result.position_error < 0.01 else "✗"
                        print(f"{status} Analytical: pos_err={result.position_error:.2e}, "
                              f"time={result.time_seconds:.1f}s")
                    
                    # FEM test
                    if verbose:
                        print(f"Testing {domain}/FEM/{optimizer}...", end=" ", flush=True)
                    result_fem = test_disk_fem(sources_true, optimizer=optimizer,
                                               n_restarts=10 if optimizer == 'L-BFGS-B' else 1)
                    results.append(result_fem)
                    
                    if verbose:
                        status = "✓" if result_fem.position_error < 0.01 else "✗"
                        print(f"{status} FEM: pos_err={result_fem.position_error:.2e}, "
                              f"time={result_fem.time_seconds:.1f}s")
                
                elif domain == 'ellipse':
                    # Test conformal
                    result = test_ellipse_conformal(sources_true, optimizer=optimizer)
                    results.append(result)
                    
                    if verbose:
                        status = "✓" if result.position_error < 0.01 else "✗"
                        print(f"{status} Conformal: pos_err={result.position_error:.2e}, "
                              f"time={result.time_seconds:.1f}s")
                    
                    # FEM test
                    if verbose:
                        print(f"Testing {domain}/FEM/{optimizer}...", end=" ", flush=True)
                    result_fem = test_ellipse_fem(sources_true, optimizer=optimizer,
                                                  n_restarts=10 if optimizer == 'L-BFGS-B' else 1)
                    results.append(result_fem)
                    
                    if verbose:
                        status = "✓" if result_fem.position_error < 0.01 else "✗"
                        print(f"{status} FEM: pos_err={result_fem.position_error:.2e}, "
                              f"time={result_fem.time_seconds:.1f}s")
                
                elif domain == 'star':
                    result = test_star_conformal(sources_true, optimizer=optimizer)
                    results.append(result)
                    
                    if verbose:
                        status = "✓" if result.position_error < 0.01 else "✗"
                        print(f"{status} Conformal: pos_err={result.position_error:.2e}, "
                              f"time={result.time_seconds:.1f}s")
    
    return results


def print_summary(results: List[ValidationResult]):
    """Print summary table of all results."""
    print("\n" + "=" * 100)
    print("VALIDATION SUMMARY")
    print("=" * 100)
    print(f"{'Domain':<10} {'Solver':<12} {'Optimizer':<25} {'N':<3} "
          f"{'Pos Error':<12} {'Int Error':<12} {'Time (s)':<10} {'Status':<8}")
    print("-" * 100)
    
    for r in results:
        status = "PASS" if r.position_error < 0.01 else "FAIL"
        if not r.success or r.position_error == float('inf'):
            status = "ERROR"
        
        pos_str = f"{r.position_error:.2e}" if r.position_error < float('inf') else "N/A"
        int_str = f"{r.intensity_error:.2e}" if r.intensity_error < float('inf') else "N/A"
        
        print(f"{r.domain:<10} {r.solver:<12} {r.optimizer:<25} {r.n_sources:<3} "
              f"{pos_str:<12} {int_str:<12} {r.time_seconds:<10.1f} {status:<8}")
        
        if "ERROR" in r.details:
            print(f"    └─ {r.details[:80]}")
    
    print("=" * 100)
    
    # Summary statistics
    passed = sum(1 for r in results if r.position_error < 0.01 and r.success)
    failed = sum(1 for r in results if r.position_error >= 0.01 and r.success)
    errors = sum(1 for r in results if not r.success or r.position_error == float('inf'))
    
    print(f"\nTotal: {len(results)} tests | PASSED: {passed} | FAILED: {failed} | ERRORS: {errors}")


def save_results(results: List[ValidationResult], filename: str = "validation_results.json"):
    """Save results to JSON file."""
    import json
    from dataclasses import asdict
    
    data = [asdict(r) for r in results]
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate nonlinear inverse solvers")
    parser.add_argument('--quick', action='store_true', 
                        help='Skip slow differential_evolution tests')
    parser.add_argument('--domain', type=str, default=None,
                        help='Test only specific domain (disk, ellipse, star)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save results to JSON file')
    
    args = parser.parse_args()
    
    domains = [args.domain] if args.domain else None
    
    results = run_validation_suite(domains=domains, quick=args.quick)
    print_summary(results)
    
    if args.save:
        save_results(results, args.save)
