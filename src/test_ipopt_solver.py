#!/usr/bin/env python3
"""
Comprehensive Test Suite for IPOPT Nonlinear Inverse Solver
============================================================

This script validates the IPOPT solver against known test cases and compares
performance with existing solvers (L-BFGS-B, differential evolution).

Tests:
1. Well-separated sources (2, 4, 6 sources) - should achieve RMSE < 1e-5
2. Comparison with existing solvers
3. Noise robustness test
4. Conformal domain test (ellipse)

Run locally with cyipopt installed:
    python test_ipopt_solver.py

Author: Claude (Anthropic)
Date: January 2026
"""

import numpy as np
import time
from typing import List, Tuple
from dataclasses import dataclass

# Check cyipopt availability first
try:
    import cyipopt
    HAS_CYIPOPT = True
except ImportError:
    HAS_CYIPOPT = False
    print("=" * 70)
    print("WARNING: cyipopt not available")
    print("Install via: conda install -c conda-forge cyipopt")
    print("This test script requires cyipopt to run.")
    print("=" * 70)
    exit(0)

# Import solvers
try:
    from ipopt_solver import (
        IPOPTNonlinearInverseSolver,
        IPOPTConformalInverseSolver,
        check_cyipopt_available,
        get_ipopt_version
    )
    from analytical_solver import (
        AnalyticalForwardSolver,
        AnalyticalNonlinearInverseSolver,
        Source,
        InverseResult
    )
    from conformal_solver import create_conformal_map, ConformalForwardSolver
except ImportError:
    from src.ipopt_solver import (
        IPOPTNonlinearInverseSolver,
        IPOPTConformalInverseSolver,
        check_cyipopt_available,
        get_ipopt_version
    )
    from src.analytical_solver import (
        AnalyticalForwardSolver,
        AnalyticalNonlinearInverseSolver,
        Source,
        InverseResult
    )
    from src.conformal_solver import create_conformal_map, ConformalForwardSolver

from scipy.optimize import linear_sum_assignment


# =============================================================================
# TEST UTILITIES
# =============================================================================

def create_well_separated_sources(n_sources: int, r_range: Tuple[float, float] = (0.6, 0.9),
                                   seed: int = 42) -> List[Tuple[Tuple[float, float], float]]:
    """
    Create well-separated test sources.
    
    - Radii in r_range
    - Angles evenly spaced with small perturbation
    - Alternating +1, -1 intensities
    - Last intensity adjusted for sum = 0
    """
    np.random.seed(seed)
    
    sources = []
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.2, 0.2, n_sources)
    
    for i, theta in enumerate(angles):
        r = np.random.uniform(r_range[0], r_range[1])
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        intensity = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), intensity))
    
    # Enforce zero sum
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    
    return sources


def compute_position_error(sources_true: List[Tuple[Tuple[float, float], float]],
                           sources_recovered: List[Source]) -> Tuple[float, float, np.ndarray]:
    """
    Compute position error using optimal matching (Hungarian algorithm).
    
    Returns: (mean_error, max_error, individual_errors)
    """
    n = len(sources_true)
    cost = np.zeros((n, n))
    
    for i, ((tx, ty), _) in enumerate(sources_true):
        for j, s in enumerate(sources_recovered):
            cost[i, j] = np.sqrt((tx - s.x)**2 + (ty - s.y)**2)
    
    row_ind, col_ind = linear_sum_assignment(cost)
    errors = cost[row_ind, col_ind]
    
    return errors.mean(), errors.max(), errors


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    n_sources: int
    solver: str
    mean_pos_error: float
    max_pos_error: float
    residual: float
    time_seconds: float
    success: bool
    message: str = ""


def print_test_result(result: TestResult, target_error: float = 1e-5):
    """Pretty print a test result."""
    status = "✓" if result.mean_pos_error < target_error else "✗"
    print(f"  {status} {result.name}")
    print(f"    Solver:     {result.solver}")
    print(f"    Position:   mean={result.mean_pos_error:.2e}, max={result.max_pos_error:.2e}")
    print(f"    Residual:   {result.residual:.2e}")
    print(f"    Time:       {result.time_seconds:.2f}s")
    if not result.success:
        print(f"    Warning:    {result.message}")


# =============================================================================
# TEST CASES
# =============================================================================

def test_well_separated_sources(n_sources: int, n_restarts: int = 10,
                                 seed: int = 42) -> TestResult:
    """Test IPOPT solver with well-separated sources."""
    
    # Create test sources
    sources_true = create_well_separated_sources(n_sources, seed=seed)
    
    # Generate measurements
    forward = AnalyticalForwardSolver(n_boundary_points=100)
    u_measured = forward.solve(sources_true)
    
    # Solve with IPOPT
    t0 = time.time()
    solver = IPOPTNonlinearInverseSolver(n_sources=n_sources, n_boundary=100)
    solver.set_measured_data(u_measured)
    result = solver.solve(n_restarts=n_restarts, verbose=False)
    elapsed = time.time() - t0
    
    # Compute error
    mean_err, max_err, _ = compute_position_error(sources_true, result.sources)
    
    return TestResult(
        name=f"{n_sources} well-separated sources",
        n_sources=n_sources,
        solver="IPOPT",
        mean_pos_error=mean_err,
        max_pos_error=max_err,
        residual=result.residual,
        time_seconds=elapsed,
        success=result.success,
        message=result.message
    )


def test_comparison_lbfgsb(n_sources: int = 4, n_restarts: int = 10,
                           seed: int = 42) -> Tuple[TestResult, TestResult]:
    """Compare IPOPT with L-BFGS-B solver."""
    
    sources_true = create_well_separated_sources(n_sources, seed=seed)
    
    forward = AnalyticalForwardSolver(n_boundary_points=100)
    u_measured = forward.solve(sources_true)
    
    # IPOPT solver
    t0 = time.time()
    ipopt_solver = IPOPTNonlinearInverseSolver(n_sources=n_sources, n_boundary=100)
    ipopt_solver.set_measured_data(u_measured)
    ipopt_result = ipopt_solver.solve(n_restarts=n_restarts, verbose=False)
    ipopt_time = time.time() - t0
    
    ipopt_mean, ipopt_max, _ = compute_position_error(sources_true, ipopt_result.sources)
    
    # L-BFGS-B solver
    t0 = time.time()
    lbfgsb_solver = AnalyticalNonlinearInverseSolver(n_sources=n_sources, n_boundary=100)
    lbfgsb_solver.set_measured_data(u_measured)
    lbfgsb_result = lbfgsb_solver.solve(method='L-BFGS-B', n_restarts=n_restarts)
    lbfgsb_time = time.time() - t0
    
    lbfgsb_mean, lbfgsb_max, _ = compute_position_error(sources_true, lbfgsb_result.sources)
    
    return (
        TestResult(
            name=f"{n_sources} sources comparison",
            n_sources=n_sources,
            solver="IPOPT",
            mean_pos_error=ipopt_mean,
            max_pos_error=ipopt_max,
            residual=ipopt_result.residual,
            time_seconds=ipopt_time,
            success=ipopt_result.success
        ),
        TestResult(
            name=f"{n_sources} sources comparison",
            n_sources=n_sources,
            solver="L-BFGS-B",
            mean_pos_error=lbfgsb_mean,
            max_pos_error=lbfgsb_max,
            residual=lbfgsb_result.residual,
            time_seconds=lbfgsb_time,
            success=lbfgsb_result.success
        )
    )


def test_noise_robustness(n_sources: int = 4, noise_levels: List[float] = [0.0, 0.01, 0.05],
                          n_restarts: int = 10, seed: int = 42) -> List[TestResult]:
    """Test solver robustness to measurement noise."""
    
    sources_true = create_well_separated_sources(n_sources, seed=seed)
    
    forward = AnalyticalForwardSolver(n_boundary_points=100)
    u_clean = forward.solve(sources_true)
    
    results = []
    
    for noise_level in noise_levels:
        np.random.seed(seed)
        u_noisy = u_clean + noise_level * np.random.randn(len(u_clean))
        
        t0 = time.time()
        solver = IPOPTNonlinearInverseSolver(n_sources=n_sources, n_boundary=100)
        solver.set_measured_data(u_noisy)
        result = solver.solve(n_restarts=n_restarts, verbose=False)
        elapsed = time.time() - t0
        
        mean_err, max_err, _ = compute_position_error(sources_true, result.sources)
        
        results.append(TestResult(
            name=f"noise σ={noise_level}",
            n_sources=n_sources,
            solver="IPOPT",
            mean_pos_error=mean_err,
            max_pos_error=max_err,
            residual=result.residual,
            time_seconds=elapsed,
            success=result.success
        ))
    
    return results


def test_ellipse_domain(n_sources: int = 4, a: float = 2.0, b: float = 1.0,
                        n_restarts: int = 10, seed: int = 42) -> TestResult:
    """Test IPOPT conformal solver on ellipse domain."""
    
    # Create conformal map
    conf_map = create_conformal_map('ellipse', a=a, b=b)
    
    # Create sources inside ellipse
    np.random.seed(seed)
    sources_true = []
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.2, 0.2, n_sources)
    
    for i, theta in enumerate(angles):
        # Scale radius to fit inside ellipse
        r_max = 0.7 * min(a, b)  # Conservative bound
        r = np.random.uniform(0.3 * r_max, 0.8 * r_max)
        x = r * np.cos(theta)
        y = r * np.sin(theta) * (b / a)  # Scale y to fit ellipse
        intensity = 1.0 if i % 2 == 0 else -1.0
        sources_true.append(((x, y), intensity))
    
    # Enforce zero sum
    total = sum(s[1] for s in sources_true)
    sources_true[-1] = (sources_true[-1][0], sources_true[-1][1] - total)
    
    # Generate measurements
    forward = ConformalForwardSolver(conf_map, n_boundary=100)
    u_measured = forward.solve(sources_true)
    
    # Solve with IPOPT conformal solver
    t0 = time.time()
    solver = IPOPTConformalInverseSolver(n_sources=n_sources, conformal_map=conf_map,
                                          n_boundary=100)
    solver.set_measured_data(u_measured)
    result = solver.solve(n_restarts=n_restarts, verbose=False)
    elapsed = time.time() - t0
    
    # Compute error
    mean_err, max_err, _ = compute_position_error(sources_true, result.sources)
    
    return TestResult(
        name=f"Ellipse ({a}x{b}) {n_sources} sources",
        n_sources=n_sources,
        solver="IPOPT-Conformal",
        mean_pos_error=mean_err,
        max_pos_error=max_err,
        residual=result.residual,
        time_seconds=elapsed,
        success=result.success
    )


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all tests and print summary."""
    
    print("=" * 70)
    print("IPOPT NONLINEAR INVERSE SOLVER - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"\ncyipopt version: {get_ipopt_version()}")
    print(f"cyipopt available: {check_cyipopt_available()}")
    
    all_results = []
    
    # Test 1: Well-separated sources (various counts)
    print("\n" + "-" * 70)
    print("TEST 1: Well-separated sources on unit disk")
    print("-" * 70)
    
    for n in [2, 4, 6]:
        result = test_well_separated_sources(n_sources=n, n_restarts=10)
        all_results.append(result)
        print_test_result(result)
    
    # Test 2: Comparison with L-BFGS-B
    print("\n" + "-" * 70)
    print("TEST 2: IPOPT vs L-BFGS-B comparison")
    print("-" * 70)
    
    ipopt_result, lbfgsb_result = test_comparison_lbfgsb(n_sources=4, n_restarts=10)
    all_results.extend([ipopt_result, lbfgsb_result])
    
    print("  IPOPT:")
    print(f"    Position:   mean={ipopt_result.mean_pos_error:.2e}, max={ipopt_result.max_pos_error:.2e}")
    print(f"    Time:       {ipopt_result.time_seconds:.2f}s")
    print("  L-BFGS-B:")
    print(f"    Position:   mean={lbfgsb_result.mean_pos_error:.2e}, max={lbfgsb_result.max_pos_error:.2e}")
    print(f"    Time:       {lbfgsb_result.time_seconds:.2f}s")
    
    if ipopt_result.mean_pos_error < lbfgsb_result.mean_pos_error:
        improvement = (lbfgsb_result.mean_pos_error - ipopt_result.mean_pos_error) / lbfgsb_result.mean_pos_error * 100
        print(f"  → IPOPT is {improvement:.1f}% better in position accuracy")
    else:
        improvement = (ipopt_result.mean_pos_error - lbfgsb_result.mean_pos_error) / ipopt_result.mean_pos_error * 100
        print(f"  → L-BFGS-B is {improvement:.1f}% better in position accuracy")
    
    # Test 3: Noise robustness
    print("\n" + "-" * 70)
    print("TEST 3: Noise robustness")
    print("-" * 70)
    
    noise_results = test_noise_robustness(n_sources=4, noise_levels=[0.0, 0.01, 0.05])
    all_results.extend(noise_results)
    
    for result in noise_results:
        print_test_result(result, target_error=0.1)  # Relaxed target for noisy data
    
    # Test 4: Ellipse domain
    print("\n" + "-" * 70)
    print("TEST 4: Conformal domain (ellipse)")
    print("-" * 70)
    
    ellipse_result = test_ellipse_domain(n_sources=4, a=2.0, b=1.0)
    all_results.append(ellipse_result)
    print_test_result(ellipse_result, target_error=1e-3)  # Slightly relaxed for conformal
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in all_results if r.mean_pos_error < 1e-3)
    total = len(all_results)
    
    print(f"\nTests passed (pos error < 1e-3): {passed}/{total}")
    
    # Check primary success criteria
    disk_results = [r for r in all_results if "well-separated" in r.name and r.solver == "IPOPT"]
    disk_passed = all(r.mean_pos_error < 1e-5 for r in disk_results)
    
    if disk_passed:
        print("\n✓ PRIMARY SUCCESS: All well-separated disk tests achieved RMSE < 1e-5")
    else:
        print("\n✗ PRIMARY FAILURE: Some well-separated disk tests did not achieve RMSE < 1e-5")
        for r in disk_results:
            if r.mean_pos_error >= 1e-5:
                print(f"    - {r.name}: {r.mean_pos_error:.2e}")
    
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    results = run_all_tests()
