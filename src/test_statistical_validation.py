#!/usr/bin/env python3
"""
Statistical Validation of Source Number Bounds (v2)
====================================================

Tests different source configurations to validate the bound formulas:

TEST CASES:
-----------
1. same_radius: All sources at same r, random angles
   - Unknowns: 2N (θ_k, I_k)
   - Equations: 2n*
   - Expected: N_max = n*

2. same_angle: All sources at same θ, random radii
   - Unknowns: 2N (r_k, I_k)  
   - Equations: n* (only a_n informative, b_n redundant)
   - Expected: N_max = (1/2)n*

3. general: Random radii and angles
   - Unknowns: 3N (r_k, θ_k, I_k)
   - Equations: 2n*
   - Expected: N_max = (2/3)n*

Usage:
    python test_statistical_validation.py --seed 0 --test-case same_radius
"""

import numpy as np
import sys
import time
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')

from analytical_solver import AnalyticalForwardSolver
from test_bound_theory import solve_inverse_disk, compute_position_rmse


# =============================================================================
# RMSE COMPUTATION
# =============================================================================

def compute_intensity_rmse(true_sources, recovered_sources) -> float:
    """
    Compute RMSE for intensities after matching sources by position.
    """
    from scipy.optimize import linear_sum_assignment
    
    n_true = len(true_sources)
    n_rec = len(recovered_sources)
    
    if n_rec == 0:
        return float('inf')
    
    cost = np.zeros((n_true, n_rec))
    for i, ((x1, y1), _) in enumerate(true_sources):
        for j, ((x2, y2), _) in enumerate(recovered_sources):
            cost[i, j] = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    
    row_ind, col_ind = linear_sum_assignment(cost)
    
    intensity_errors = []
    for i, j in zip(row_ind, col_ind):
        intensity_errors.append((true_sources[i][1] - recovered_sources[j][1])**2)
    
    return np.sqrt(np.mean(intensity_errors))


# =============================================================================
# THEORETICAL COMPUTATIONS
# =============================================================================

def compute_sigma_four(sigma_noise: float, n_sensors: int) -> float:
    """σ_Four = σ_noise / √M"""
    return sigma_noise / np.sqrt(n_sensors)


def compute_n_star_predicted(sigma_four: float, rho_min: float) -> float:
    """
    n* from theory (σ_Four based, doesn't include πn factor)
    """
    return np.log(sigma_four) / np.log(rho_min)


def compute_N_max_for_test_case(n_star: float, test_case: str) -> float:
    """
    Compute N_max using the appropriate formula for each test case.
    
    Parameters
    ----------
    n_star : float
        The actual n* computed from noise realization
    test_case : str
        One of: 'same_radius', 'same_angle', 'general'
    
    Returns
    -------
    N_max : float
        The predicted maximum recoverable sources
    """
    if test_case == 'same_radius':
        # Unknowns: 2N (θ, I), Equations: 2n* → N_max = n*
        return n_star
    elif test_case == 'same_angle':
        # Unknowns: 2N (r, I), Equations: n* → N_max = (1/2)n*
        return 0.5 * n_star
    elif test_case == 'general':
        # Unknowns: 3N (r, θ, I), Equations: 2n* → N_max = (2/3)n*
        return (2.0 / 3.0) * n_star
    else:
        raise ValueError(f"Unknown test_case: {test_case}")


# =============================================================================
# NOISE FOURIER ANALYSIS
# =============================================================================

def compute_noise_fourier_coeffs(noise: np.ndarray, n_modes: int = 50) -> np.ndarray:
    """
    Compute |η̂_n| for n = 1, ..., n_modes using correct normalization.
    
    Uses: |η̂_n| = (2/M) * |FFT[n]|
    """
    M = len(noise)
    fft_coeffs = np.fft.fft(noise)  # Unnormalized DFT
    eta_abs = (2.0 / M) * np.abs(fft_coeffs[1:n_modes + 1])
    return eta_abs


def compute_n_star_actual(noise_fourier_coeffs: np.ndarray, rho_min: float, 
                          n_cutoff: int = 50) -> Tuple[int, int, List[int]]:
    """
    Compute n*_actual using CORRECT SNR condition.
    
    Signal level at mode n: ρ^n / (π*n)
    Noise level at mode n: |η̂_n|
    
    Condition: signal > noise → ρ^n / (π*n) > |η̂_n|
    Rearranged: ρ^n > π * n * |η̂_n|
    
    Returns
    -------
    n_star_max : int
        Max mode index where condition holds
    K : int
        Count of modes where condition holds
    usable_modes : list
        List of mode indices that pass
    """
    usable_modes = []
    
    n_max_check = min(n_cutoff, len(noise_fourier_coeffs))
    
    for n in range(1, n_max_check + 1):
        rho_power = rho_min ** n
        eta_n = noise_fourier_coeffs[n - 1]
        
        # Correct condition: ρ^n > π * n * |η̂_n|
        threshold = np.pi * n * eta_n
        
        if rho_power > threshold:
            usable_modes.append(n)
    
    n_star_max = max(usable_modes) if usable_modes else 0
    K = len(usable_modes)
    
    return n_star_max, K, usable_modes


# =============================================================================
# SOURCE GENERATION FOR DIFFERENT TEST CASES
# =============================================================================

def generate_zero_sum_intensities(n_sources: int) -> np.ndarray:
    """Generate intensities summing to zero: alternating +1/-1."""
    intensities = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n_sources)])
    intensities = intensities - np.mean(intensities)  # Ensure exact sum = 0
    return intensities


def generate_sources_same_radius(n_sources: int, rho: float, seed: int) -> Tuple[List, float]:
    """
    TEST CASE 1: Same radius, random angles.
    
    All sources at r_k = ρ, angles evenly spaced with small perturbation.
    
    Returns
    -------
    sources : list of ((x, y), intensity)
    rho_min : float (= rho for this case)
    """
    np.random.seed(seed)
    
    # Evenly spaced angles with small perturbation
    base_angles = np.linspace(0, 2 * np.pi, n_sources, endpoint=False)
    perturbation = 0.1 * (2 * np.pi / n_sources) * np.random.randn(n_sources)
    angles = base_angles + perturbation
    
    # All same radius
    radii = np.full(n_sources, rho)
    
    # Intensities
    intensities = generate_zero_sum_intensities(n_sources)
    
    sources = []
    for i in range(n_sources):
        x = radii[i] * np.cos(angles[i])
        y = radii[i] * np.sin(angles[i])
        sources.append(((x, y), intensities[i]))
    
    rho_min = rho  # All at same radius
    return sources, rho_min


def generate_sources_same_angle(n_sources: int, theta_0: float, r_min: float, 
                                 r_max: float, seed: int) -> Tuple[List, float]:
    """
    TEST CASE 2: Same angle, random radii.
    
    All sources at θ_k = θ_0, radii spread in [r_min, r_max].
    
    Returns
    -------
    sources : list of ((x, y), intensity)
    rho_min : float (= r_min for this case)
    """
    np.random.seed(seed)
    
    # All same angle
    angles = np.full(n_sources, theta_0)
    
    # Radii evenly spread in range
    radii = np.linspace(r_min, r_max, n_sources)
    
    # Intensities
    intensities = generate_zero_sum_intensities(n_sources)
    
    sources = []
    for i in range(n_sources):
        x = radii[i] * np.cos(angles[i])
        y = radii[i] * np.sin(angles[i])
        sources.append(((x, y), intensities[i]))
    
    rho_min = r_min  # Smallest radius
    return sources, rho_min


def generate_sources_general(n_sources: int, r_min: float, r_max: float, 
                              seed: int) -> Tuple[List, float]:
    """
    TEST CASE 3: General configuration (random radii, random angles).
    
    Returns
    -------
    sources : list of ((x, y), intensity)
    rho_min : float (= r_min for this case)
    """
    np.random.seed(seed)
    
    # Evenly spaced angles with perturbation
    base_angles = np.linspace(0, 2 * np.pi, n_sources, endpoint=False)
    perturbation = 0.1 * (2 * np.pi / n_sources) * np.random.randn(n_sources)
    angles = base_angles + perturbation
    
    # Radii evenly spread in range
    radii = np.linspace(r_min, r_max, n_sources)
    
    # Intensities
    intensities = generate_zero_sum_intensities(n_sources)
    
    sources = []
    for i in range(n_sources):
        x = radii[i] * np.cos(angles[i])
        y = radii[i] * np.sin(angles[i])
        sources.append(((x, y), intensities[i]))
    
    rho_min = r_min  # Smallest radius
    return sources, rho_min


def generate_sources_for_test_case(test_case: str, n_sources: int, seed: int,
                                    rho: float = 0.7, r_min: float = 0.5, 
                                    r_max: float = 0.9, theta_0: float = 0.0) -> Tuple[List, float]:
    """
    Generate sources for the specified test case.
    
    Parameters
    ----------
    test_case : str
        One of: 'same_radius', 'same_angle', 'general'
    n_sources : int
        Number of sources
    seed : int
        Random seed
    rho : float
        Common radius for same_radius case
    r_min, r_max : float
        Radius range for same_angle and general cases
    theta_0 : float
        Common angle for same_angle case
    
    Returns
    -------
    sources : list of ((x, y), intensity)
    rho_min : float
        Minimum radius (for n* computation)
    """
    if test_case == 'same_radius':
        return generate_sources_same_radius(n_sources, rho, seed)
    elif test_case == 'same_angle':
        return generate_sources_same_angle(n_sources, theta_0, r_min, r_max, seed)
    elif test_case == 'general':
        return generate_sources_general(n_sources, r_min, r_max, seed)
    else:
        raise ValueError(f"Unknown test_case: {test_case}")


# =============================================================================
# TRANSITION DETECTION
# =============================================================================

def detect_N_transition_ratio(rmse_dict: Dict[int, float], ratio_threshold: float = 2.0) -> int:
    """
    Detect N_transition using ratio method.
    Returns first N where RMSE jumps by > ratio_threshold.
    """
    N_values = sorted(rmse_dict.keys())
    
    for i in range(len(N_values) - 1):
        N_curr = N_values[i]
        N_next = N_values[i + 1]
        
        rmse_curr = rmse_dict[N_curr]
        rmse_next = rmse_dict[N_next]
        
        if rmse_curr > 0:
            ratio = rmse_next / rmse_curr
            if ratio > ratio_threshold:
                return N_next
    
    return N_values[-1]


def detect_N_transition_derivative(rmse_dict: Dict[int, float]) -> int:
    """
    Detect N_transition using max derivative method.
    """
    N_values = sorted(rmse_dict.keys())
    
    if len(N_values) < 2:
        return N_values[0]
    
    max_deriv = -np.inf
    N_transition = N_values[-1]
    
    for i in range(len(N_values) - 1):
        N_curr = N_values[i]
        N_next = N_values[i + 1]
        
        rmse_curr = rmse_dict[N_curr]
        rmse_next = rmse_dict[N_next]
        
        if rmse_curr > 0 and rmse_next > 0:
            deriv = (np.log(rmse_next) - np.log(rmse_curr)) / (N_next - N_curr)
            if deriv > max_deriv:
                max_deriv = deriv
                N_transition = N_next
    
    return N_transition


# =============================================================================
# DYNAMIC N VALUES
# =============================================================================

def compute_N_values_around_prediction(N_predicted: float, delta: int = 6, step: int = 2) -> List[int]:
    """
    Generate N values centered around N_predicted with range ±delta.
    
    Parameters
    ----------
    N_predicted : float
        The predicted N_max
    delta : int
        Range: test from N_predicted - delta to N_predicted + delta
    step : int
        Step size between N values
    
    Returns
    -------
    N_values : list of int
        N values to test, all even and >= 2
    """
    N_center = int(round(N_predicted))
    
    N_values = []
    for offset in range(-delta, delta + 1, step):
        N = N_center + offset
        # Ensure N is even (required for zero-sum intensities) and >= 2
        if N % 2 != 0:
            N += 1
        if N >= 2:
            N_values.append(N)
    
    # Remove duplicates and sort
    N_values = sorted(set(N_values))
    
    return N_values


# =============================================================================
# MAIN VALIDATION FOR ONE SEED
# =============================================================================

def run_single_seed_validation(
    seed: int,
    test_case: str = 'same_radius',
    rho: float = 0.7,
    r_min: float = 0.5,
    r_max: float = 0.9,
    theta_0: float = 0.0,
    sigma_noise: float = 0.001,
    n_sensors: int = 100,
    n_restarts: int = 15,
    N_values: List[int] = None,
    use_dynamic_N: bool = True
) -> dict:
    """
    Run statistical validation for ONE seed with specified test case.
    
    Parameters
    ----------
    seed : int
        Random seed for this experiment
    test_case : str
        One of: 'same_radius', 'same_angle', 'general'
    rho : float
        Common radius for same_radius case
    r_min, r_max : float
        Radius range for same_angle and general cases
    theta_0 : float
        Common angle for same_angle case
    sigma_noise : float
        Noise standard deviation
    n_sensors : int
        Number of boundary sensors
    n_restarts : int
        Optimizer restarts
    N_values : list
        If provided, use these N values instead of dynamic
    use_dynamic_N : bool
        If True and N_values is None, compute N values around prediction
    
    Returns
    -------
    result : dict
        Contains all results and statistics
    """
    start_time = time.time()
    
    # Determine rho_min for n* computation based on test case
    if test_case == 'same_radius':
        rho_min = rho
    else:
        rho_min = r_min
    
    # Setup forward solver
    forward_solver = AnalyticalForwardSolver(n_boundary_points=n_sensors)
    theta_boundary = np.linspace(0, 2 * np.pi, n_sensors, endpoint=False)
    
    # =========================================================================
    # CRITICAL: Generate noise ONCE for this seed
    # =========================================================================
    np.random.seed(seed)
    noise = sigma_noise * np.random.randn(n_sensors)
    
    # Compute noise Fourier coefficients
    noise_fourier_coeffs = compute_noise_fourier_coeffs(noise, n_modes=50)
    
    # Compute n*_actual
    n_star_max, K, usable_modes = compute_n_star_actual(noise_fourier_coeffs, rho_min, n_cutoff=50)
    
    # Compute N_max using the CORRECT formula for this test case
    N_max_predicted = compute_N_max_for_test_case(n_star_max, test_case)
    
    # Also compute N_max using other formulas for comparison
    N_max_same_radius = float(n_star_max)
    N_max_same_angle = 0.5 * n_star_max
    N_max_general = (2.0 / 3.0) * n_star_max
    
    # Compute σ_Four-based prediction (for reference)
    sigma_four = compute_sigma_four(sigma_noise, n_sensors)
    n_star_sigma_four = compute_n_star_predicted(sigma_four, rho_min)
    N_max_sigma_four = compute_N_max_for_test_case(n_star_sigma_four, test_case)
    
    # Determine N values to test
    if N_values is None:
        if use_dynamic_N:
            # Dynamic N values centered around prediction with ±6 range
            N_values = compute_N_values_around_prediction(N_max_predicted, delta=6, step=2)
        else:
            # Default fixed range
            N_values = [10, 12, 14, 16, 18, 20, 22, 24]
    
    print(f"  Test case: {test_case}")
    print(f"  rho_min = {rho_min:.3f}")
    print(f"  n*_actual = {n_star_max}")
    print(f"  N_max_predicted ({test_case}) = {N_max_predicted:.2f}")
    print(f"  Testing N values: {N_values}")
    
    # =========================================================================
    # Test each N value
    # =========================================================================
    rmse_position_results = {}
    rmse_intensity_results = {}
    
    for N in N_values:
        # Generate sources for this test case
        source_seed = seed + 10000 + N
        sources, _ = generate_sources_for_test_case(
            test_case, N, source_seed,
            rho=rho, r_min=r_min, r_max=r_max, theta_0=theta_0
        )
        
        # Forward solve
        u_true = forward_solver.solve(sources)
        
        # Add noise
        u_measured = u_true + noise
        
        # Inverse solve
        recovered, residual = solve_inverse_disk(
            u_measured, theta_boundary, n_sources=N,
            n_restarts=n_restarts, seed=seed + 20000 + N
        )
        
        # Convert to tuples if needed
        if hasattr(recovered[0], 'to_tuple'):
            recovered = [s.to_tuple() for s in recovered]
        
        # Compute RMSE
        rmse_pos = compute_position_rmse(sources, recovered)
        rmse_int = compute_intensity_rmse(sources, recovered)
        
        rmse_position_results[N] = float(rmse_pos)
        rmse_intensity_results[N] = float(rmse_int)
        
        print(f"    N={N}: RMSE_pos = {rmse_pos:.6f}, RMSE_int = {rmse_int:.6f}")
    
    # Detect transitions
    N_transition_pos_ratio = detect_N_transition_ratio(rmse_position_results)
    N_transition_pos_deriv = detect_N_transition_derivative(rmse_position_results)
    N_transition_int_ratio = detect_N_transition_ratio(rmse_intensity_results)
    N_transition_int_deriv = detect_N_transition_derivative(rmse_intensity_results)
    
    # Primary transition (position, ratio method)
    N_transition = N_transition_pos_ratio
    
    elapsed = time.time() - start_time
    
    # Build result dict
    result = {
        'seed': int(seed),
        'test_case': test_case,
        'rho_min': float(rho_min),
        
        # Config
        'rho': float(rho),
        'r_min': float(r_min),
        'r_max': float(r_max),
        'theta_0': float(theta_0),
        
        # RMSE results
        'rmse_position': rmse_position_results,
        'rmse_intensity': rmse_intensity_results,
        'N_values_tested': [int(n) for n in N_values],
        
        # Noise analysis
        'noise_fourier_coeffs': [float(c) for c in noise_fourier_coeffs],
        'n_star_max': int(n_star_max),
        'K': int(K),
        'usable_modes': [int(m) for m in usable_modes],
        
        # N_max predictions (using different formulas)
        'N_max_predicted': float(N_max_predicted),  # Using correct formula for test_case
        'N_max_same_radius': float(N_max_same_radius),  # n*
        'N_max_same_angle': float(N_max_same_angle),    # (1/2)n*
        'N_max_general': float(N_max_general),          # (2/3)n*
        
        # σ_Four-based (for reference)
        'n_star_sigma_four': float(n_star_sigma_four),
        'N_max_sigma_four': float(N_max_sigma_four),
        
        # Transition detection
        'N_transition': int(N_transition),
        'N_transition_pos_ratio': int(N_transition_pos_ratio),
        'N_transition_pos_deriv': int(N_transition_pos_deriv),
        'N_transition_int_ratio': int(N_transition_int_ratio),
        'N_transition_int_deriv': int(N_transition_int_deriv),
        
        # Backward compatibility
        'n_star_actual': int(n_star_max),
        'N_max_actual': float(N_max_predicted),
        'n_star_predicted': float(n_star_sigma_four),
        
        'time_seconds': float(elapsed)
    }
    
    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Statistical validation for one seed")
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--test-case', type=str, default='same_radius',
                       choices=['same_radius', 'same_angle', 'general'],
                       help='Test case type')
    parser.add_argument('--rho', type=float, default=0.7, help='Common radius (same_radius case)')
    parser.add_argument('--r-min', type=float, default=0.5, help='Min radius (same_angle, general)')
    parser.add_argument('--r-max', type=float, default=0.9, help='Max radius (same_angle, general)')
    parser.add_argument('--theta-0', type=float, default=0.0, help='Common angle (same_angle case)')
    parser.add_argument('--sigma-noise', type=float, default=0.001, help='Noise std dev')
    parser.add_argument('--sensors', type=int, default=100, help='Number of sensors')
    parser.add_argument('--restarts', type=int, default=15, help='Optimizer restarts')
    parser.add_argument('--no-dynamic-N', action='store_true', help='Use fixed N values')
    
    args = parser.parse_args()
    
    print(f"Running statistical validation")
    print(f"  Seed: {args.seed}")
    print(f"  Test case: {args.test_case}")
    print(f"  sigma_noise: {args.sigma_noise}, sensors: {args.sensors}")
    
    result = run_single_seed_validation(
        seed=args.seed,
        test_case=args.test_case,
        rho=args.rho,
        r_min=args.r_min,
        r_max=args.r_max,
        theta_0=args.theta_0,
        sigma_noise=args.sigma_noise,
        n_sensors=args.sensors,
        n_restarts=args.restarts,
        use_dynamic_N=not args.no_dynamic_N
    )
    
    print(f"\n{'='*60}")
    print(f"RESULTS: Seed {args.seed}, Test Case: {args.test_case}")
    print(f"{'='*60}")
    print(f"  n*_max = {result['n_star_max']}")
    print(f"  K = {result['K']}")
    print(f"  N_max_predicted ({args.test_case}) = {result['N_max_predicted']:.2f}")
    print(f"  N_transition = {result['N_transition']}")
    print(f"  Error = {result['N_transition'] - result['N_max_predicted']:.2f}")
    print(f"  Time: {result['time_seconds']:.1f}s")
    print(f"\n  RMSE_position: {result['rmse_position']}")
    print(f"  RMSE_intensity: {result['rmse_intensity']}")
