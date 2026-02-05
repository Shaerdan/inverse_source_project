#!/usr/bin/env python3
"""
Statistical Validation of Source Number Bounds - Multi-Domain
==============================================================

Extends test_statistical_validation.py to support:
  - disk (analytical solver, baseline)
  - ellipse (conformal map, a=1.5, b=0.8)
  - brain (conformal map via MFS)

Solver patterns follow EXACTLY the tested flow in test_bound_theory.py:
  - Ellipse: create_conformal_map('ellipse', a=1.5, b=0.8)
  - Brain: MFSConformalMap with boundary from mesh.get_brain_boundary
  - Source generation: cmap.from_disk(rho * exp(i*theta))
  - Forward: ConformalForwardSolver(cmap, n_sensors)
  - Sensor locations: fwd_solver.boundary_points
  - Inverse: ConformalNonlinearInverseSolver(cmap, n_sources,
             n_boundary=len(sensor_locations), sensor_locations=sensor_locations)

Usage:
    python test_statistical_validation_multidomain.py --seed 0 --test-case general --domain disk
    python test_statistical_validation_multidomain.py --seed 0 --test-case general --domain ellipse
    python test_statistical_validation_multidomain.py --seed 0 --test-case general --domain brain
"""

import numpy as np
import sys
import time
from typing import List, Tuple, Dict, Callable, Optional
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')

# =============================================================================
# IMPORTS — same as test_bound_theory.py
# =============================================================================

from analytical_solver import (
    AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver,
)
from conformal_solver import (
    MFSConformalMap, ConformalForwardSolver, ConformalNonlinearInverseSolver,
    create_conformal_map
)
# test_bound_theory.py functions we reuse for disk
from test_bound_theory import solve_inverse_disk, compute_position_rmse


SUPPORTED_DOMAINS = ['disk', 'ellipse', 'brain']


# =============================================================================
# CONFORMAL MAP CREATION — copied from test_bound_theory.py run_test()
# =============================================================================

def create_ellipse_cmap(a: float = 1.5, b: float = 0.8):
    """Create ellipse conformal map — same as test_bound_theory.py line 609."""
    return create_conformal_map('ellipse', a=a, b=b)


def create_brain_cmap():
    """Create brain conformal map — same as test_bound_theory.py lines 611-621."""
    from mesh import get_brain_boundary
    from scipy.interpolate import interp1d

    boundary_pts = get_brain_boundary(200)
    z_boundary = boundary_pts[:, 0] + 1j * boundary_pts[:, 1]

    t_vals = np.linspace(0, 2*np.pi, len(z_boundary), endpoint=False)
    real_interp = interp1d(t_vals, z_boundary.real, kind='cubic', fill_value='extrapolate')
    imag_interp = interp1d(t_vals, z_boundary.imag, kind='cubic', fill_value='extrapolate')

    def brain_boundary(t):
        t = t % (2*np.pi)
        return real_interp(t) + 1j * imag_interp(t)

    return MFSConformalMap(brain_boundary, n_boundary=200, n_charge=150)


# =============================================================================
# FORWARD / INVERSE SOLVER WRAPPERS — following test_bound_theory.py run_test()
# =============================================================================

def forward_solve_disk(sources, n_sensors):
    """Disk forward solve — test_bound_theory.py lines 601-605."""
    theta = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
    solver = AnalyticalForwardSolver(n_boundary_points=n_sensors)
    u_true = solver.solve(sources)
    sensor_locations = np.column_stack([np.cos(theta), np.sin(theta)])
    return u_true, theta, sensor_locations


def forward_solve_conformal(sources, cmap, n_sensors):
    """Conformal forward solve — test_bound_theory.py lines 623-625."""
    fwd_solver = ConformalForwardSolver(cmap, n_sensors)
    u_true = fwd_solver.solve(sources)
    sensor_locations = fwd_solver.boundary_points
    return u_true, sensor_locations


def inverse_solve_disk(u_measured, theta, n_sources, n_restarts, seed):
    """Disk inverse solve — test_bound_theory.py lines 638-641."""
    return solve_inverse_disk(u_measured, theta, n_sources,
                               n_restarts=n_restarts, seed=seed)


def inverse_solve_conformal(u_measured, cmap, sensor_locations, n_sources, n_restarts, seed):
    """Conformal inverse solve — test_bound_theory.py lines 302-317."""
    solver = ConformalNonlinearInverseSolver(
        cmap,
        n_sources=n_sources,
        n_boundary=len(sensor_locations),
        sensor_locations=sensor_locations
    )
    sources, residual = solver.solve(u_measured, method='SLSQP',
                                      n_restarts=n_restarts, seed=seed)
    return sources, residual


# =============================================================================
# SOURCE GENERATION — following test_bound_theory.py patterns
# =============================================================================

def generate_zero_sum_intensities_flexible(n_sources: int, seed: int = None) -> np.ndarray:
    """
    Generate intensities that sum to zero.
    Alternating ±1 with small perturbation.

    Unlike test_bound_theory.py which requires even N, this handles odd N too
    (needed for the statistical validation which sweeps a range of N values).
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    intensities = np.empty(n_sources)
    for i in range(n_sources):
        intensities[i] = 1.0 if i % 2 == 0 else -1.0

    # Small random perturbation for variety
    intensities += 0.1 * rng.randn(n_sources)

    # Enforce zero sum via centering
    intensities -= np.mean(intensities)

    return intensities


def generate_random_intensities(n_sources: int, seed: int = None,
                                intensity_low: float = 0.5,
                                intensity_high: float = 2.0) -> np.ndarray:
    """
    Generate random-magnitude intensities that sum to zero.

    I_k = U(intensity_low, intensity_high) * (-1)^k, then centered to enforce Σ I_k = 0.
    This tests whether the bound holds with non-uniform intensities.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    magnitudes = rng.uniform(intensity_low, intensity_high, n_sources)
    signs = np.array([(-1)**k for k in range(n_sources)], dtype=float)
    intensities = magnitudes * signs
    intensities -= np.mean(intensities)

    return intensities


def generate_sources_conformal(test_case: str, n_sources: int, seed: int,
                                cmap, rho: float = 0.7,
                                r_min: float = 0.5, r_max: float = 0.9,
                                theta_0: float = 0.0,
                                intensity_low: float = 0.5,
                                intensity_high: float = 2.0) -> Tuple[List, float]:
    """
    Generate sources in conformal coordinates, then map to physical domain.

    For disk: cmap=None, conformal coords ARE physical coords.
    For ellipse/brain: uses cmap.from_disk(w) to map to physical domain.

    Parameters
    ----------
    test_case : str
        'same_radius', 'same_angle', 'general', or 'general_random_intensity'
    cmap : conformal map object or None (for disk)
    intensity_low, intensity_high : float
        Magnitude range for general_random_intensity case (default: 0.5, 2.0)
    """
    np.random.seed(seed)

    if test_case == 'same_radius':
        radii = np.full(n_sources, rho)
        base_angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
        perturbation = 0.1 * (2*np.pi / n_sources) * np.random.randn(n_sources)
        angles = base_angles + perturbation
        rho_min = rho

    elif test_case == 'same_angle':
        angles = np.full(n_sources, theta_0)
        radii = np.linspace(r_min, r_max, n_sources)
        rho_min = r_min

    elif test_case == 'general':
        base_angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
        perturbation = 0.1 * (2*np.pi / n_sources) * np.random.randn(n_sources)
        angles = base_angles + perturbation
        radii = np.linspace(r_min, r_max, n_sources)
        np.random.shuffle(radii)
        rho_min = r_min

    elif test_case == 'general_random_intensity':
        # Same geometry as 'general', but random-magnitude intensities
        base_angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
        perturbation = 0.1 * (2*np.pi / n_sources) * np.random.randn(n_sources)
        angles = base_angles + perturbation
        radii = np.linspace(r_min, r_max, n_sources)
        np.random.shuffle(radii)
        rho_min = r_min

    else:
        raise ValueError(f"Unknown test_case: {test_case}")

    # Choose intensity generation based on test case
    if test_case == 'general_random_intensity':
        intensities = generate_random_intensities(n_sources,
                                                  intensity_low=intensity_low,
                                                  intensity_high=intensity_high)
    else:
        intensities = generate_zero_sum_intensities_flexible(n_sources)

    sources = []
    for i in range(n_sources):
        if cmap is None:
            # Disk: conformal coords = physical coords
            x = radii[i] * np.cos(angles[i])
            y = radii[i] * np.sin(angles[i])
        else:
            # Ellipse/Brain: map from disk to physical via inverse conformal map
            # Same pattern as test_bound_theory.py lines 237-239
            w_target = radii[i] * np.exp(1j * angles[i])
            z = cmap.from_disk(w_target)
            x, y = float(z.real), float(z.imag)

        sources.append(((x, y), intensities[i]))

    return sources, rho_min


# =============================================================================
# RMSE COMPUTATION
# =============================================================================

def compute_intensity_rmse(true_sources, recovered_sources) -> float:
    """Compute RMSE for intensities after matching sources by position."""
    from scipy.optimize import linear_sum_assignment

    n_true = len(true_sources)
    n_rec = len(recovered_sources)
    if n_rec == 0:
        return float('inf')

    # Extract positions
    true_pos = [(s[0] if isinstance(s, tuple) else (s.x, s.y)) for s in true_sources]
    rec_pos = [(s[0] if isinstance(s, tuple) else (s.x, s.y)) for s in recovered_sources]

    cost = np.zeros((n_true, n_rec))
    for i, (x1, y1) in enumerate(true_pos):
        for j, (x2, y2) in enumerate(rec_pos):
            cost[i, j] = (x1 - x2)**2 + (y1 - y2)**2

    row_ind, col_ind = linear_sum_assignment(cost)

    true_int = [(s[1] if isinstance(s, tuple) else s.intensity) for s in true_sources]
    rec_int = [(s[1] if isinstance(s, tuple) else s.intensity) for s in recovered_sources]

    intensity_errors = []
    for i, j in zip(row_ind, col_ind):
        intensity_errors.append((true_int[i] - rec_int[j])**2)

    return np.sqrt(np.mean(intensity_errors))


def compute_total_rmse(true_sources, recovered_sources) -> float:
    """Compute RMSE over ALL unknown parameters (x, y, S) per matched source."""
    from scipy.optimize import linear_sum_assignment

    n_true = len(true_sources)
    n_rec = len(recovered_sources)
    if n_rec == 0:
        return float('inf')

    true_pos = [(s[0] if isinstance(s, tuple) else (s.x, s.y)) for s in true_sources]
    rec_pos = [(s[0] if isinstance(s, tuple) else (s.x, s.y)) for s in recovered_sources]

    cost = np.zeros((n_true, n_rec))
    for i, (x1, y1) in enumerate(true_pos):
        for j, (x2, y2) in enumerate(rec_pos):
            cost[i, j] = (x1 - x2)**2 + (y1 - y2)**2

    row_ind, col_ind = linear_sum_assignment(cost)

    true_int = [(s[1] if isinstance(s, tuple) else s.intensity) for s in true_sources]
    rec_int = [(s[1] if isinstance(s, tuple) else s.intensity) for s in recovered_sources]

    all_sq = []
    for i, j in zip(row_ind, col_ind):
        x1, y1 = true_pos[i]
        x2, y2 = rec_pos[j]
        all_sq.append((x1 - x2)**2)
        all_sq.append((y1 - y2)**2)
        all_sq.append((true_int[i] - rec_int[j])**2)

    return np.sqrt(np.mean(all_sq))


# =============================================================================
# THEORETICAL COMPUTATIONS (same as disk version)
# =============================================================================

def compute_sigma_four(sigma_noise: float, n_sensors: int) -> float:
    return sigma_noise / np.sqrt(n_sensors)


def compute_n_star_predicted(sigma_four: float, rho_min: float) -> float:
    return np.log(sigma_four) / np.log(rho_min)


def compute_N_max_for_test_case(n_star: float, test_case: str) -> float:
    if test_case == 'same_radius':
        return n_star
    elif test_case == 'same_angle':
        return 0.5 * n_star
    elif test_case in ('general', 'general_random_intensity'):
        # Same DOF: 3N unknowns, 2n* equations → (2/3) n*
        return (2.0 / 3.0) * n_star
    else:
        raise ValueError(f"Unknown test_case: {test_case}")


# =============================================================================
# NOISE FOURIER ANALYSIS (same as disk version)
# =============================================================================

def compute_noise_fourier_coeffs(noise: np.ndarray, n_modes: int = 50) -> np.ndarray:
    M = len(noise)
    fft_coeffs = np.fft.fft(noise)
    eta_abs = (2.0 / M) * np.abs(fft_coeffs[1:n_modes + 1])
    return eta_abs


def compute_n_star_actual(noise_fourier_coeffs: np.ndarray, rho_min: float,
                          n_cutoff: int = 50) -> Tuple[int, int, List[int]]:
    usable_modes = []
    n_star_max = 0

    for n in range(1, min(n_cutoff, len(noise_fourier_coeffs)) + 1):
        signal_level = rho_min**n / (np.pi * n)
        noise_level = noise_fourier_coeffs[n - 1]

        if signal_level > noise_level:
            usable_modes.append(n)
            n_star_max = max(n_star_max, n)

    K = len(usable_modes)
    return n_star_max, K, usable_modes


# =============================================================================
# DYNAMIC N VALUES
# =============================================================================

def compute_N_values_around_prediction(N_predicted: float, delta: int = 6, step: int = 2) -> List[int]:
    N_center = max(2, int(round(N_predicted)))
    N_min = max(2, N_center - delta)
    N_max = N_center + delta
    N_values = list(range(N_min, N_max + 1, step))
    if 2 not in N_values and N_min > 2:
        N_values = [2] + N_values
    return N_values


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def run_single_seed_validation(
    seed: int,
    test_case: str = 'general',
    domain: str = 'disk',
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
    rho_min_high: float = 0.6,
    intensity_low: float = 0.5,
    intensity_high: float = 2.0,
) -> dict:
    """
    Run statistical validation for a single seed on any domain.

    Sources are generated in conformal coordinates and mapped to physical
    domain. Theory is identical across domains in conformal coords.
    """
    start_time = time.time()

    # =========================================================================
    # 1. Create conformal map (None for disk)
    #    Following test_bound_theory.py run_test() lines 601-625
    # =========================================================================
    print(f"  Setting up domain: {domain}")

    if domain == 'disk':
        cmap = None
    elif domain == 'ellipse':
        cmap = create_ellipse_cmap(a=1.5, b=0.8)
    elif domain == 'brain':
        cmap = create_brain_cmap()
    else:
        raise ValueError(f"Unsupported domain: {domain}. Choose from {SUPPORTED_DOMAINS}")

    # =========================================================================
    # 2. Determine rho_min
    # =========================================================================
    if random_rho_min:
        rng = np.random.RandomState(seed + 99999)
        rho_min = rng.uniform(rho_min_low, rho_min_high)
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

    # =========================================================================
    # 3. Generate noise ONCE for this seed
    # =========================================================================
    np.random.seed(seed)
    noise = sigma_noise * np.random.randn(n_sensors)

    # Noise Fourier analysis
    noise_fourier_coeffs = compute_noise_fourier_coeffs(noise, n_modes=50)
    n_star_max, K, usable_modes = compute_n_star_actual(noise_fourier_coeffs, rho_min, n_cutoff=50)

    # N_max predictions
    N_max_predicted = compute_N_max_for_test_case(n_star_max, test_case)
    N_max_same_radius = float(n_star_max)
    N_max_same_angle = 0.5 * n_star_max
    N_max_general = (2.0 / 3.0) * n_star_max

    sigma_four = compute_sigma_four(sigma_noise, n_sensors)
    n_star_sigma_four = compute_n_star_predicted(sigma_four, rho_min)
    N_max_sigma_four = compute_N_max_for_test_case(n_star_sigma_four, test_case)

    # Determine N values
    if N_values is None:
        if use_dynamic_N:
            N_values = compute_N_values_around_prediction(N_max_predicted, delta=6, step=2)
        else:
            N_values = [10, 12, 14, 16, 18, 20, 22, 24]

    print(f"  Domain: {domain}")
    print(f"  Test case: {test_case}")
    print(f"  rho_min = {rho_min:.3f}")
    print(f"  n*_actual = {n_star_max}")
    print(f"  N_max_predicted ({test_case}) = {N_max_predicted:.2f}")
    print(f"  Testing N values: {N_values}")

    # =========================================================================
    # 4. Test each N value
    # =========================================================================
    rmse_position_results = {}
    rmse_intensity_results = {}
    rmse_total_results = {}

    for N in N_values:
        source_seed = seed + 10000 + N

        # Generate sources in conformal coords, mapped to physical domain
        sources, _ = generate_sources_conformal(
            test_case, N, source_seed, cmap=cmap,
            rho=rho, r_min=r_min, r_max=r_max, theta_0=theta_0,
            intensity_low=intensity_low, intensity_high=intensity_high
        )

        # Forward solve — following test_bound_theory.py run_test()
        if domain == 'disk':
            u_true, theta_boundary, sensor_locations = forward_solve_disk(sources, n_sensors)
        else:
            u_true, sensor_locations = forward_solve_conformal(sources, cmap, n_sensors)

        # Add noise (same noise realization for all N)
        u_measured = u_true + noise

        # Inverse solve — following test_bound_theory.py run_test()
        if domain == 'disk':
            recovered, residual = inverse_solve_disk(
                u_measured, theta_boundary, n_sources=N,
                n_restarts=n_restarts, seed=seed + 20000 + N
            )
        else:
            recovered, residual = inverse_solve_conformal(
                u_measured, cmap, sensor_locations, n_sources=N,
                n_restarts=n_restarts, seed=seed + 20000 + N
            )

        # Convert to tuples if needed (Source objects)
        if hasattr(recovered[0], 'to_tuple'):
            recovered = [s.to_tuple() for s in recovered]

        # Compute RMSE
        rmse_pos = compute_position_rmse(sources, recovered)
        rmse_int = compute_intensity_rmse(sources, recovered)
        rmse_tot = compute_total_rmse(sources, recovered)

        rmse_position_results[N] = float(rmse_pos)
        rmse_intensity_results[N] = float(rmse_int)
        rmse_total_results[N] = float(rmse_tot)

        print(f"    N={N}: RMSE_pos={rmse_pos:.6f}, RMSE_int={rmse_int:.6f}, RMSE_total={rmse_tot:.6f}")

    elapsed = time.time() - start_time

    # =========================================================================
    # 5. Build result dict (same structure as disk version for aggregation)
    # =========================================================================
    result = {
        'seed': int(seed),
        'test_case': test_case,
        'domain': domain,
        'rho_min': float(rho_min),

        # Config
        'rho': float(rho),
        'r_min': float(r_min),
        'r_max': float(r_max),
        'theta_0': float(theta_0),

        # RMSE results
        'rmse_position': rmse_position_results,
        'rmse_intensity': rmse_intensity_results,
        'rmse_total': rmse_total_results,
        'N_values_tested': [int(n) for n in N_values],

        # Noise analysis
        'noise_fourier_coeffs': [float(c) for c in noise_fourier_coeffs],
        'n_star_max': int(n_star_max),
        'K': int(K),
        'usable_modes': [int(m) for m in usable_modes],

        # N_max predictions
        'N_max_predicted': float(N_max_predicted),
        'N_max_same_radius': float(N_max_same_radius),
        'N_max_same_angle': float(N_max_same_angle),
        'N_max_general': float(N_max_general),

        'n_star_sigma_four': float(n_star_sigma_four),
        'N_max_sigma_four': float(N_max_sigma_four),

        'n_star_actual': int(n_star_max),
        'N_max_actual': float(N_max_predicted),
        'n_star_predicted': float(n_star_sigma_four),

        'random_rho_min': random_rho_min,
        'rho_min_low': float(rho_min_low) if random_rho_min else None,
        'rho_min_high': float(rho_min_high) if random_rho_min else None,

        # Intensity range (for general_random_intensity)
        'intensity_low': float(intensity_low),
        'intensity_high': float(intensity_high),

        'time_seconds': float(elapsed)
    }

    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-domain statistical validation")
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--test-case', type=str, default='general',
                       choices=['same_radius', 'same_angle', 'general', 'general_random_intensity'])
    parser.add_argument('--domain', type=str, default='disk',
                       choices=SUPPORTED_DOMAINS,
                       help='Domain geometry (default: disk)')
    parser.add_argument('--rho', type=float, default=0.7)
    parser.add_argument('--r-min', type=float, default=0.5)
    parser.add_argument('--r-max', type=float, default=0.9)
    parser.add_argument('--theta-0', type=float, default=0.0)
    parser.add_argument('--sigma-noise', type=float, default=0.001)
    parser.add_argument('--sensors', type=int, default=100)
    parser.add_argument('--restarts', type=int, default=15)
    parser.add_argument('--no-dynamic-N', action='store_true')
    parser.add_argument('--random-rho-min', action='store_true')
    parser.add_argument('--rho-min-low', type=float, default=0.5)
    parser.add_argument('--rho-min-high', type=float, default=0.6)
    parser.add_argument('--intensity-low', type=float, default=0.5,
                       help='Min intensity magnitude for general_random_intensity (default: 0.5)')
    parser.add_argument('--intensity-high', type=float, default=2.0,
                       help='Max intensity magnitude for general_random_intensity (default: 2.0)')

    args = parser.parse_args()

    print(f"Running multi-domain statistical validation")
    print(f"  Seed: {args.seed}")
    print(f"  Domain: {args.domain}")
    print(f"  Test case: {args.test_case}")
    print(f"  sigma_noise: {args.sigma_noise}, sensors: {args.sensors}")
    if args.random_rho_min:
        print(f"  Random rho_min: [{args.rho_min_low}, {args.rho_min_high}]")
    if args.test_case == 'general_random_intensity':
        print(f"  Intensity range: [{args.intensity_low}, {args.intensity_high}]")

    result = run_single_seed_validation(
        seed=args.seed,
        test_case=args.test_case,
        domain=args.domain,
        rho=args.rho,
        r_min=args.r_min,
        r_max=args.r_max,
        theta_0=args.theta_0,
        sigma_noise=args.sigma_noise,
        n_sensors=args.sensors,
        n_restarts=args.restarts,
        use_dynamic_N=not args.no_dynamic_N,
        random_rho_min=args.random_rho_min,
        rho_min_low=args.rho_min_low,
        rho_min_high=args.rho_min_high,
        intensity_low=args.intensity_low,
        intensity_high=args.intensity_high
    )

    print(f"\n{'='*60}")
    print(f"RESULTS: Seed {args.seed}, Domain: {args.domain}, Test Case: {args.test_case}")
    print(f"{'='*60}")
    print(f"  rho_min = {result['rho_min']:.4f}")
    print(f"  n*_max = {result['n_star_max']}")
    print(f"  K = {result['K']}")
    print(f"  N_max_predicted ({args.test_case}) = {result['N_max_predicted']:.2f}")
    print(f"\n  Time: {result['time_seconds']:.1f}s")
    print(f"\n  RMSE_position:  {result['rmse_position']}")
    print(f"  RMSE_intensity: {result['rmse_intensity']}")
    print(f"  RMSE_total:     {result['rmse_total']}")
