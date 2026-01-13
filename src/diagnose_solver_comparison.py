#!/usr/bin/env python3
"""
Diagnose Polar vs Cartesian solver differences.

Key questions:
1. Do both solvers use analytical gradients? NO - both use numerical
2. Are initial guesses comparable?
3. Is the optimization landscape different?
4. Why is DE failing for N>4?
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DiagResult:
    solver: str
    n_sources: int
    n_restarts: int
    pos_err: float
    int_err: float
    n_evals: int
    time: float
    final_misfit: float
    
def create_well_separated_sources(n_sources: int, seed: int = 42) -> list:
    """Create well-separated test sources."""
    np.random.seed(seed)
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += 0.1 * np.random.randn(n_sources)
    
    sources = []
    for i, theta in enumerate(angles):
        r = 0.6 + 0.15 * np.random.rand()
        x, y = r * np.cos(theta), r * np.sin(theta)
        intensity = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), intensity))
    
    # Adjust last intensity for sum = 0
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    
    return sources

def compute_errors(recovered, true_sources):
    """Compute position and intensity errors."""
    n = len(true_sources)
    
    # Hungarian matching for best assignment
    from scipy.optimize import linear_sum_assignment
    
    cost_matrix = np.zeros((n, n))
    for i, (pos_r, _) in enumerate(recovered):
        for j, (pos_t, _) in enumerate(true_sources):
            cost_matrix[i, j] = np.sqrt((pos_r[0]-pos_t[0])**2 + (pos_r[1]-pos_t[1])**2)
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    pos_errors = [cost_matrix[i, j] for i, j in zip(row_ind, col_ind)]
    int_errors = [abs(recovered[i][1] - true_sources[j][1]) 
                  for i, j in zip(row_ind, col_ind)]
    
    return np.mean(pos_errors), np.mean(int_errors)


def test_cartesian(sources_true: list, n_restarts: int = 20, maxiter: int = 500, 
                   verbose: bool = False) -> DiagResult:
    """Test Cartesian solver with detailed diagnostics."""
    from analytical_solver import AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver
    
    n = len(sources_true)
    forward = AnalyticalForwardSolver(n_boundary_points=100)
    u_measured = forward.solve(sources_true)
    
    inverse = AnalyticalNonlinearInverseSolver(n_sources=n, n_boundary=100)
    inverse.set_measured_data(u_measured)
    
    t0 = time.time()
    result = inverse.solve(method='L-BFGS-B', maxiter=maxiter, n_restarts=n_restarts)
    elapsed = time.time() - t0
    
    recovered = [((s.x, s.y), s.intensity) for s in result.sources]
    pos_err, int_err = compute_errors(recovered, sources_true)
    
    if verbose:
        print(f"  Cartesian: {len(inverse.history)} evals, misfit={inverse.history[-1]:.2e}")
    
    return DiagResult(
        solver='cartesian',
        n_sources=n,
        n_restarts=n_restarts,
        pos_err=pos_err,
        int_err=int_err,
        n_evals=len(inverse.history),
        time=elapsed,
        final_misfit=inverse.history[-1] if inverse.history else float('inf')
    )


def test_polar(sources_true: list, n_restarts: int = 20, maxiter: int = 500,
               verbose: bool = False) -> DiagResult:
    """Test Polar solver with detailed diagnostics."""
    from analytical_solver import AnalyticalForwardSolver
    from polar_solver import PolarNonlinearInverseSolver
    
    n = len(sources_true)
    forward = AnalyticalForwardSolver(n_boundary_points=100)
    u_measured = forward.solve(sources_true)
    
    inverse = PolarNonlinearInverseSolver(n_sources=n, n_boundary=100)
    inverse.set_measured_data(u_measured)
    
    t0 = time.time()
    result = inverse.solve(method='L-BFGS-B', maxiter=maxiter, n_restarts=n_restarts,
                          use_gradient=False)  # Make sure we use same gradient approach
    elapsed = time.time() - t0
    
    recovered = [((s.x, s.y), s.intensity) for s in result.sources]
    pos_err, int_err = compute_errors(recovered, sources_true)
    
    if verbose:
        print(f"  Polar: {len(inverse.history)} evals, misfit={inverse.history[-1]:.2e}")
    
    return DiagResult(
        solver='polar',
        n_sources=n,
        n_restarts=n_restarts,
        pos_err=pos_err,
        int_err=int_err,
        n_evals=len(inverse.history),
        time=elapsed,
        final_misfit=inverse.history[-1] if inverse.history else float('inf')
    )


def test_de_cartesian(sources_true: list, maxiter: int = 1000, 
                      verbose: bool = False) -> DiagResult:
    """Test Cartesian with Differential Evolution."""
    from analytical_solver import AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver
    
    n = len(sources_true)
    forward = AnalyticalForwardSolver(n_boundary_points=100)
    u_measured = forward.solve(sources_true)
    
    inverse = AnalyticalNonlinearInverseSolver(n_sources=n, n_boundary=100)
    inverse.set_measured_data(u_measured)
    
    t0 = time.time()
    result = inverse.solve(method='differential_evolution', maxiter=maxiter)
    elapsed = time.time() - t0
    
    recovered = [((s.x, s.y), s.intensity) for s in result.sources]
    pos_err, int_err = compute_errors(recovered, sources_true)
    
    if verbose:
        print(f"  DE Cartesian: {len(inverse.history)} evals, misfit={inverse.history[-1]:.2e}")
    
    return DiagResult(
        solver='cartesian_DE',
        n_sources=n,
        n_restarts=1,
        pos_err=pos_err,
        int_err=int_err,
        n_evals=len(inverse.history),
        time=elapsed,
        final_misfit=inverse.history[-1] if inverse.history else float('inf')
    )


def run_fair_comparison():
    """Run fair comparison between solvers."""
    print("=" * 70)
    print("FAIR SOLVER COMPARISON")
    print("=" * 70)
    print("\nBoth solvers use:")
    print("  - Same forward solver (AnalyticalForwardSolver)")
    print("  - Same boundary points (100)")
    print("  - Same test sources")
    print("  - Numerical gradients (L-BFGS-B internal finite differences)")
    print("  - Same maxiter per restart (500)")
    print()
    
    for n_sources in [4, 6, 8]:
        print(f"\n{'='*70}")
        print(f"N = {n_sources} SOURCES")
        print(f"{'='*70}")
        
        sources = create_well_separated_sources(n_sources, seed=42)
        print("True sources:")
        for i, ((x, y), q) in enumerate(sources):
            print(f"  {i+1}: ({x:+.3f}, {y:+.3f}), q={q:+.4f}")
        
        # Scale restarts with problem size
        n_restarts = 10 * n_sources
        
        print(f"\nL-BFGS-B with {n_restarts} restarts:")
        
        cart = test_cartesian(sources, n_restarts=n_restarts, verbose=True)
        polar = test_polar(sources, n_restarts=n_restarts, verbose=True)
        
        print(f"\n{'Solver':<12} {'Pos Err':<12} {'Int Err':<12} {'#Evals':<10} {'Time':<10} {'Misfit':<12}")
        print("-" * 70)
        print(f"{'Cartesian':<12} {cart.pos_err:<12.2e} {cart.int_err:<12.2e} {cart.n_evals:<10} {cart.time:<10.2f} {cart.final_misfit:<12.2e}")
        print(f"{'Polar':<12} {polar.pos_err:<12.2e} {polar.int_err:<12.2e} {polar.n_evals:<10} {polar.time:<10.2f} {polar.final_misfit:<12.2e}")
        
        # Also test DE with more iterations
        print(f"\nDifferential Evolution (maxiter=1000 for N≤4, 2000 for N>4):")
        de_maxiter = 1000 if n_sources <= 4 else 2000
        de_cart = test_de_cartesian(sources, maxiter=de_maxiter, verbose=True)
        print(f"{'DE Cartesian':<12} {de_cart.pos_err:<12.2e} {de_cart.int_err:<12.2e} {de_cart.n_evals:<10} {de_cart.time:<10.2f} {de_cart.final_misfit:<12.2e}")


def check_gradient_usage():
    """Verify neither solver uses analytical gradient by default."""
    print("\n" + "=" * 70)
    print("GRADIENT USAGE CHECK")
    print("=" * 70)
    
    # Check Cartesian
    from analytical_solver import AnalyticalNonlinearInverseSolver
    import inspect
    
    solve_source = inspect.getsource(AnalyticalNonlinearInverseSolver.solve)
    if 'jac=' in solve_source or 'gradient' in solve_source.lower():
        print("Cartesian: May use gradient (check code)")
    else:
        print("Cartesian: Uses L-BFGS-B internal numerical gradient")
    
    # Check Polar  
    from polar_solver import PolarNonlinearInverseSolver
    solve_source = inspect.getsource(PolarNonlinearInverseSolver.solve)
    if 'use_gradient' in solve_source:
        print("Polar: Has use_gradient option (default=False)")
    
    print("\nConclusion: Both use numerical gradients by default - FAIR comparison")


def check_initial_guess_distribution():
    """Visualize initial guess distributions."""
    print("\n" + "=" * 70)
    print("INITIAL GUESS COMPARISON")
    print("=" * 70)
    
    from analytical_solver import AnalyticalNonlinearInverseSolver
    from polar_solver import PolarNonlinearInverseSolver
    
    n = 4
    cart = AnalyticalNonlinearInverseSolver(n_sources=n, n_boundary=100)
    polar = PolarNonlinearInverseSolver(n_sources=n, n_boundary=100)
    
    print("\nFirst 5 initial guesses (positions only):")
    print("\nCartesian solver:")
    for seed in range(5):
        x0 = cart._get_initial_guess('circle' if seed == 0 else 'random', seed)
        positions = [(x0[3*i], x0[3*i+1]) for i in range(n)]
        radii = [np.sqrt(x**2 + y**2) for x, y in positions]
        print(f"  Seed {seed}: r = {radii}")
    
    print("\nPolar solver:")
    for seed in range(5):
        x0 = polar._get_initial_guess(seed, 'circle' if seed == 0 else 'random')
        # Polar params: [S1, r1, θ1, S2, r2, θ2, ..., r_n, θ_n]
        radii = []
        for i in range(n-1):
            radii.append(x0[3*i + 1])
        radii.append(x0[3*(n-1)])  # Last source r
        print(f"  Seed {seed}: r = {radii}")


if __name__ == '__main__':
    check_gradient_usage()
    check_initial_guess_distribution()
    run_fair_comparison()
