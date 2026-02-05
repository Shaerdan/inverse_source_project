#!/usr/bin/env python3
"""
Truncated RMSE Test (Test B): Inverse Recovery on Truncated Fourier System
===========================================================================

Tests whether source recovery succeeds/fails at the predicted N_max boundary
when using the TRUNCATED forward map (first n* Fourier modes only).

This is the correct system for validating the theoretical bound: the theorem
says the truncated system h: R^{3N} -> R^{2n*} loses injectivity at
N > (2/3)*n*. Test A showed the null space exists; Test B shows this null
space causes actual recovery failure.

CRITICAL CONSISTENCY: Both the signal and the noise are truncated to the
same n* modes. The data vector is d_trunc = h(s_true) + eta_trunc, where
eta_trunc contains only the first n* Fourier noise components.

The solver minimises ||h(s) - d_trunc||^2 over source parameters s,
using SLSQP with multistart (matching the existing inverse solver pattern).

Usage:
    python test_rmse_truncated.py --seed 0 --test-case general_random_intensity
"""

import numpy as np
import sys
import time
from typing import List, Tuple
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')

# Import shared functions from existing modules
from test_statistical_validation import (
    compute_noise_fourier_coeffs,
    compute_n_star_actual,
    compute_N_max_for_test_case,
    compute_sigma_four,
    compute_n_star_predicted,
    generate_sources_for_test_case,
    compute_N_values_around_prediction,
)

from test_perturbation import (
    forward_truncated,
    compute_noise_fourier_components,
    sources_to_polar,
)

from test_bound_theory import compute_position_rmse


def compute_all_rmse(true_sources: list, recovered_sources: list) -> dict:
    """
    Compute position, intensity, and total RMSE using Hungarian matching.
    
    Matching is based on position (same as compute_position_rmse), then
    the same matching is used for intensity and total RMSE.
    
    Returns
    -------
    dict with keys:
        rmse_position : float
        rmse_intensity : float  
        rmse_total : float (all 3 components: x, y, I)
        matching : list of (true_idx, rec_idx) pairs
    """
    from scipy.optimize import linear_sum_assignment
    
    n = len(true_sources)
    if len(recovered_sources) != n:
        return {
            'rmse_position': np.inf,
            'rmse_intensity': np.inf,
            'rmse_total': np.inf,
            'matching': [],
        }
    
    # Extract positions and intensities
    true_pos = []
    true_I = []
    for s in true_sources:
        if hasattr(s, 'x'):
            true_pos.append((s.x, s.y))
            true_I.append(s.intensity)
        else:
            true_pos.append(s[0])
            true_I.append(s[1])
    
    rec_pos = []
    rec_I = []
    for s in recovered_sources:
        if hasattr(s, 'x'):
            rec_pos.append((s.x, s.y))
            rec_I.append(s.intensity)
        else:
            rec_pos.append(s[0])
            rec_I.append(s[1])
    
    # Build position cost matrix for Hungarian matching
    cost = np.zeros((n, n))
    for i, (x1, y1) in enumerate(true_pos):
        for j, (x2, y2) in enumerate(rec_pos):
            cost[i, j] = (x1 - x2)**2 + (y1 - y2)**2
    
    # Optimal matching based on position
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Compute all three RMSE metrics using this matching
    sum_sq_pos = 0.0
    sum_sq_I = 0.0
    sum_sq_total = 0.0
    
    for i, j in zip(row_ind, col_ind):
        x1, y1 = true_pos[i]
        x2, y2 = rec_pos[j]
        I1 = true_I[i]
        I2 = rec_I[j]
        
        dx2 = (x1 - x2)**2
        dy2 = (y1 - y2)**2
        dI2 = (I1 - I2)**2
        
        sum_sq_pos += dx2 + dy2
        sum_sq_I += dI2
        sum_sq_total += dx2 + dy2 + dI2
    
    return {
        'rmse_position': float(np.sqrt(sum_sq_pos / n)),
        'rmse_intensity': float(np.sqrt(sum_sq_I / n)),
        'rmse_total': float(np.sqrt(sum_sq_total / n)),
        'matching': list(zip(row_ind.tolist(), col_ind.tolist())),
    }


# =============================================================================
# TRUNCATED INVERSE SOLVER
# =============================================================================

class TruncatedInverseSolver:
    """
    Nonlinear inverse solver operating on the truncated Fourier system.

    Minimises ||h(s) - d_trunc||^2 where h maps source parameters to
    the first n* Fourier coefficients [a_1, b_1, ..., a_n*, b_n*].

    Parameters are laid out as:
        [x_0, y_0, x_1, y_1, ..., x_{N-1}, y_{N-1}, q_0, q_1, ..., q_{N-1}]
    (Cartesian positions + intensities, matching existing solver convention)

    Zero-sum constraint: sum(q_k) = 0, enforced via SLSQP equality constraint.
    Disk constraint: x_k^2 + y_k^2 < 1, enforced via SLSQP inequality constraint.
    """

    def __init__(self, n_sources: int, n_star: int):
        self.n_sources = n_sources
        self.n_star = n_star
        self.d_trunc = None  # Truncated data vector (2*n_star,)

    def set_data(self, d_trunc: np.ndarray):
        """Set the truncated Fourier data to fit."""
        assert len(d_trunc) == 2 * self.n_star
        self.d_trunc = d_trunc.copy()

    def _params_to_polar(self, params: np.ndarray):
        """Convert Cartesian params to (r, theta, I) for forward_truncated."""
        n = self.n_sources
        x = np.array([params[2 * i] for i in range(n)])
        y = np.array([params[2 * i + 1] for i in range(n)])
        q = np.array([params[2 * n + i] for i in range(n)])

        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        return r, theta, q

    def _objective(self, params: np.ndarray) -> float:
        """||h(s) - d_trunc||^2"""
        r, theta, q = self._params_to_polar(params)
        h = forward_truncated(r, theta, q, self.n_star)
        return float(np.sum((h - self.d_trunc) ** 2))

    def _objective_gradient(self, params: np.ndarray) -> np.ndarray:
        """Numerical gradient via central finite differences."""
        eps = 1e-7
        grad = np.zeros_like(params)
        for j in range(len(params)):
            p_plus = params.copy(); p_plus[j] += eps
            p_minus = params.copy(); p_minus[j] -= eps
            grad[j] = (self._objective(p_plus) - self._objective(p_minus)) / (2 * eps)
        return grad

    def _get_initial_guess(self, strategy: str, restart_idx: int) -> np.ndarray:
        """Generate initial guess for optimisation."""
        n = self.n_sources
        rng = np.random.RandomState(restart_idx * 1000 + 7)

        if strategy == 'spread':
            # Sources evenly spread on a circle at r=0.5
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            r0 = 0.5
            x0 = r0 * np.cos(angles)
            y0 = r0 * np.sin(angles)
        elif strategy == 'circle':
            # Different radius
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n
            r0 = 0.3
            x0 = r0 * np.cos(angles)
            y0 = r0 * np.sin(angles)
        else:  # random
            r0 = rng.uniform(0.1, 0.85, n)
            angles = rng.uniform(0, 2 * np.pi, n)
            x0 = r0 * np.cos(angles)
            y0 = r0 * np.sin(angles)

        # Alternating intensities, centred to sum=0
        q0 = np.array([(-1) ** i for i in range(n)], dtype=float)
        q0 = q0 - np.mean(q0)
        # Add small noise
        q0 += 0.1 * rng.randn(n)
        q0 = q0 - np.mean(q0)

        params = np.zeros(3 * n)
        for i in range(n):
            params[2 * i] = x0[i]
            params[2 * i + 1] = y0[i]
        params[2 * n:] = q0
        return params

    def solve(self, n_restarts: int = 15, maxiter: int = 10000,
              seed: int = 42) -> Tuple[list, float]:
        """
        Solve via SLSQP with multistart.

        Returns
        -------
        sources : list of ((x, y), intensity)
        residual : float
            Best objective value
        """
        if self.d_trunc is None:
            raise ValueError("Call set_data() first")

        np.random.seed(seed)
        n = self.n_sources

        # Bounds
        box_bounds = [(-0.95, 0.95)] * (2 * n) + [(-5.0, 5.0)] * n

        # Constraints
        def intensity_sum(params):
            return sum(params[2 * n + i] for i in range(n))

        def disk_ineq(params):
            return np.array([1.0 - params[2 * i] ** 2 - params[2 * i + 1] ** 2
                             for i in range(n)])

        constraints = [
            {'type': 'eq', 'fun': intensity_sum},
            {'type': 'ineq', 'fun': disk_ineq},
        ]

        # Multistart
        init_strategies = ['spread', 'circle'] + ['random'] * max(0, n_restarts - 2)
        best_result = None
        best_fun = np.inf

        for restart, init_type in enumerate(init_strategies[:n_restarts]):
            x0 = self._get_initial_guess(init_type, restart + seed)

            try:
                result = minimize(
                    self._objective,
                    x0,
                    method='SLSQP',
                    bounds=box_bounds,
                    constraints=constraints,
                    options={'maxiter': maxiter, 'ftol': 1e-14, 'disp': False}
                )
                if result.fun < best_fun:
                    best_fun = result.fun
                    best_result = result
            except Exception:
                pass

        if best_result is None:
            # Return worst case
            return [((0.0, 0.0), 0.0)] * n, np.inf

        # Extract sources from best result
        params = best_result.x
        sources = []
        q_raw = np.array([params[2 * n + i] for i in range(n)])
        # NOT centering here: SLSQP enforces sum=0 via constraint
        for i in range(n):
            sources.append(((params[2 * i], params[2 * i + 1]), q_raw[i]))

        return sources, best_fun


# =============================================================================
# MAIN VALIDATION FOR ONE SEED
# =============================================================================

def run_single_seed_rmse_truncated(
    seed: int,
    test_case: str = 'general_random_intensity',
    rho: float = 0.7,
    r_min: float = 0.5,
    r_max: float = 0.9,
    theta_0: float = 0.0,
    sigma_noise: float = 0.001,
    n_sensors: int = 100,
    n_restarts: int = 15,
    N_values: List[int] = None,
    use_dynamic_N: bool = True,
    random_rho_min: bool = False,
    rho_min_low: float = 0.5,
    rho_min_high: float = 0.7,
    intensity_low: float = 0.5,
    intensity_high: float = 2.0,
) -> dict:
    """
    Run truncated RMSE test (Test B) for ONE seed.

    For each N, generates sources, builds the truncated data vector
    (signal_trunc + noise_trunc, both to same n*), then runs the
    truncated inverse solver and computes position RMSE.

    Returns
    -------
    result : dict
        Complete results for this seed
    """
    start_time = time.time()

    # -----------------------------------------------------------------
    # Determine rho_min (same logic as test_statistical_validation)
    # -----------------------------------------------------------------
    if random_rho_min:
        rng_rho = np.random.RandomState(seed + 99999)
        rho_min = rng_rho.uniform(rho_min_low, rho_min_high)
        if test_case == 'same_radius':
            rho = rho_min
        else:
            r_min = rho_min
            r_max = min(r_min + 0.4, 0.95)
    else:
        if test_case == 'same_radius':
            rho_min = rho
        else:
            rho_min = r_min

    # -----------------------------------------------------------------
    # Generate noise ONCE for this seed (same as other pipelines)
    # -----------------------------------------------------------------
    np.random.seed(seed)
    noise = sigma_noise * np.random.randn(n_sensors)

    # Compute noise Fourier magnitudes for n*
    noise_fourier_magnitudes = compute_noise_fourier_coeffs(noise, n_modes=50)

    # Compute n*
    n_star_max, K, usable_modes = compute_n_star_actual(
        noise_fourier_magnitudes, rho_min, n_cutoff=50
    )

    if n_star_max == 0:
        print(f"  WARNING: n_star = 0 for seed {seed}")
        return {
            'seed': int(seed), 'test_case': test_case,
            'rho_min': float(rho_min), 'n_star': 0, 'N_max': 0.0,
            'results_by_N': [], 'time_seconds': float(time.time() - start_time),
            'error': 'n_star is zero',
        }

    # N_max for this test case
    N_max = compute_N_max_for_test_case(float(n_star_max), test_case)

    # Truncated noise components (same n* for noise and signal)
    noise_trunc = compute_noise_fourier_components(noise, n_star_max, n_sensors)

    # Reference values
    sigma_four = compute_sigma_four(sigma_noise, n_sensors)
    n_star_sigma_four = compute_n_star_predicted(sigma_four, rho_min)
    N_max_sigma_four = compute_N_max_for_test_case(n_star_sigma_four, test_case)

    # Determine N values to test
    if N_values is None:
        if use_dynamic_N:
            N_values = compute_N_values_around_prediction(N_max, delta=6, step=2)
        else:
            N_values = [2, 4, 6, 8, 10, 12, 14, 16]

    print(f"  Seed {seed}: rho_min={rho_min:.4f}, n*={n_star_max}, "
          f"N_max({test_case})={N_max:.2f}")
    print(f"    N values: {N_values}")

    # -----------------------------------------------------------------
    # Test each N value
    # -----------------------------------------------------------------
    results_by_N = []

    for N in N_values:
        t0 = time.time()

        # Generate sources (same seeding as other pipelines)
        source_seed = seed + 10000 + N
        sources, _ = generate_sources_for_test_case(
            test_case, N, source_seed,
            rho=rho, r_min=r_min, r_max=r_max, theta_0=theta_0,
            intensity_low=intensity_low, intensity_high=intensity_high,
        )

        # Convert to polar for truncated forward map
        r_true, theta_true, I_true = sources_to_polar(sources)

        # Truncated signal at truth
        signal_trunc = forward_truncated(r_true, theta_true, I_true, n_star_max)

        # Truncated data = signal + noise (both truncated to same n*)
        data_trunc = signal_trunc + noise_trunc

        # --- Solve truncated inverse problem ---
        solver = TruncatedInverseSolver(n_sources=N, n_star=n_star_max)
        solver.set_data(data_trunc)

        recovered, residual = solver.solve(
            n_restarts=n_restarts,
            seed=seed + 20000 + N,
        )

        # Compute all RMSE metrics (position, intensity, total)
        rmse_dict = compute_all_rmse(sources, recovered)
        rmse_pos = rmse_dict['rmse_position']
        rmse_int = rmse_dict['rmse_intensity']
        rmse_tot = rmse_dict['rmse_total']

        elapsed_N = time.time() - t0
        status = "SUCCESS" if rmse_pos < 0.1 else "FAILED"
        print(f"    N={N:2d}: RMSE_pos={rmse_pos:.4f} RMSE_int={rmse_int:.4f} "
              f"RMSE_tot={rmse_tot:.4f} res={residual:.2e} ({elapsed_N:.1f}s) [{status}]")

        results_by_N.append({
            'N': int(N),
            'n_star': int(n_star_max),
            'N_max': float(N_max),
            'N_minus_Nmax': float(N - N_max),
            'rmse_position': float(rmse_pos),
            'rmse_intensity': float(rmse_int),
            'rmse_total': float(rmse_tot),
            'residual': float(residual),
            'time_seconds': float(elapsed_N),
            'true_sources': [{'pos': list(s[0]), 'intensity': float(s[1])}
                             for s in sources],
            'recovered_sources': [{'pos': list(s[0]), 'intensity': float(s[1])}
                                  for s in recovered],
        })

    elapsed = time.time() - start_time

    result = {
        'seed': int(seed),
        'test_case': test_case,
        'rho_min': float(rho_min),
        'n_star': int(n_star_max),
        'K': int(K),
        'usable_modes': [int(m) for m in usable_modes],
        'N_max': float(N_max),
        'N_values_tested': [int(n) for n in N_values],

        # Reference values
        'sigma_noise': float(sigma_noise),
        'n_sensors': int(n_sensors),
        'n_restarts': int(n_restarts),
        'sigma_four': float(sigma_four),
        'n_star_sigma_four': float(n_star_sigma_four),
        'N_max_sigma_four': float(N_max_sigma_four),

        # Random rho_min info
        'random_rho_min': random_rho_min,
        'rho_min_low': float(rho_min_low) if random_rho_min else None,
        'rho_min_high': float(rho_min_high) if random_rho_min else None,

        # Intensity range
        'intensity_low': float(intensity_low),
        'intensity_high': float(intensity_high),

        # Per-N results
        'results_by_N': results_by_N,

        'time_seconds': float(elapsed),
    }

    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Truncated RMSE test (Test B) for one seed"
    )
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--test-case', type=str, default='general_random_intensity',
                        choices=['same_radius', 'same_angle', 'general',
                                 'general_random_intensity'])
    parser.add_argument('--rho', type=float, default=0.7)
    parser.add_argument('--r-min', type=float, default=0.5)
    parser.add_argument('--r-max', type=float, default=0.9)
    parser.add_argument('--theta-0', type=float, default=0.0)
    parser.add_argument('--sigma-noise', type=float, default=0.001)
    parser.add_argument('--sensors', type=int, default=100)
    parser.add_argument('--n-restarts', type=int, default=15)
    parser.add_argument('--random-rho-min', action='store_true')
    parser.add_argument('--rho-min-low', type=float, default=0.5)
    parser.add_argument('--rho-min-high', type=float, default=0.7)
    parser.add_argument('--intensity-low', type=float, default=0.5)
    parser.add_argument('--intensity-high', type=float, default=2.0)

    args = parser.parse_args()

    print(f"Running truncated RMSE test (Test B)")
    print(f"  Seed: {args.seed}, Test case: {args.test_case}")

    result = run_single_seed_rmse_truncated(
        seed=args.seed,
        test_case=args.test_case,
        rho=args.rho,
        r_min=args.r_min,
        r_max=args.r_max,
        theta_0=args.theta_0,
        sigma_noise=args.sigma_noise,
        n_sensors=args.sensors,
        n_restarts=args.n_restarts,
        random_rho_min=args.random_rho_min,
        rho_min_low=args.rho_min_low,
        rho_min_high=args.rho_min_high,
        intensity_low=args.intensity_low,
        intensity_high=args.intensity_high,
    )

    print(f"\n{'='*60}")
    print(f"RESULTS: Seed {args.seed}")
    print(f"  n* = {result['n_star']}, N_max = {result['N_max']:.2f}")
    print(f"  Time: {result['time_seconds']:.1f}s")
