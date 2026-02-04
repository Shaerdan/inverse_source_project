#!/usr/bin/env python3
"""
Statistical Validation of the Source Number Bound
==================================================

Tests whether N_max_actual (computed from actual noise Fourier coefficients)
accurately predicts where RMSE transition occurs.

This script handles ONE seed. Run via run_statistical_experiment.py wrapper.

Key difference from test_bound_theory.py:
- Same noise realization used for ALL N values within a seed
- Computes actual noise Fourier coefficients |η̂_n|
- Tests wider range of N to find transition point
- Detects transition from data (no arbitrary threshold)

Usage:
    # Typically called via wrapper, but can run directly:
    python test_statistical_validation.py --seed 0
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


def compute_intensity_rmse(true_sources, recovered_sources) -> float:
    """
    Compute RMSE for intensities after matching sources by position.
    
    Uses Hungarian algorithm to match sources, then computes intensity error.
    """
    from scipy.optimize import linear_sum_assignment
    
    n_true = len(true_sources)
    n_rec = len(recovered_sources)
    
    if n_rec == 0:
        return float('inf')
    
    # Build cost matrix based on position distances
    cost = np.zeros((n_true, n_rec))
    for i, ((x1, y1), _) in enumerate(true_sources):
        for j, ((x2, y2), _) in enumerate(recovered_sources):
            cost[i, j] = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    
    # Match sources
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Compute intensity errors for matched pairs
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
    n* from theory: r_min^{n*} = σ_Four
    Solving: n* = log(σ_Four) / log(r_min)
    """
    return np.log(sigma_four) / np.log(rho_min)


def compute_N_max(n_star: float) -> float:
    """N_max = (2/3) * n*"""
    return (2.0 / 3.0) * n_star


# =============================================================================
# NOISE FOURIER ANALYSIS
# =============================================================================

def compute_noise_fourier_coeffs(noise: np.ndarray, n_modes: int = 50) -> np.ndarray:
    """
    Compute |η̂_n| for n = 1, ..., n_modes using correct normalization.
    
    Uses the standard real Fourier convention:
        a_n = (2/M) * sum_j η_j * cos(n θ_j)
        b_n = (2/M) * sum_j η_j * sin(n θ_j)
        |η̂_n| = sqrt(a_n² + b_n²)
    
    Via FFT: |η̂_n| = (2/M) * |X_n| where X_n is the unnormalized DFT.
    
    Parameters
    ----------
    noise : array of shape (M,)
        Spatial noise: η_j for j = 0, ..., M-1
    n_modes : int
        Number of Fourier modes to compute
    
    Returns
    -------
    eta_abs : array of shape (n_modes,)
        |η̂_n| for n = 1, ..., n_modes
    """
    M = len(noise)
    fft_coeffs = np.fft.fft(noise)  # Unnormalized DFT
    
    # |η̂_n| = (2/M) * |X_n| for n = 1, ..., n_modes
    eta_abs = (2.0 / M) * np.abs(fft_coeffs[1:n_modes + 1])
    
    return eta_abs


def compute_n_star_both_methods(noise_fourier_coeffs: np.ndarray, rho_min: float, 
                                 n_cutoff: int = 50) -> Tuple[int, int, List[int]]:
    """
    Compute n* using both methods with CORRECT SNR condition.
    
    The signal Fourier coefficient from a source is ~ r^n / (π*n).
    The noise Fourier coefficient is |η̂_n|.
    
    SNR > 1 condition: r^n / (π*n) > |η̂_n|
    Rearranged: r^n > π * n * |η̂_n|
    
    Definition A (Max Index): n*_max = max{n : r^n > π*n*|η̂_n|}
    Definition B (Count): K = #{n : r^n > π*n*|η̂_n|}
    
    Parameters
    ----------
    noise_fourier_coeffs : array
        |η̂_n| for n = 1, 2, ..., computed using standard convention
    rho_min : float
        Minimum conformal radius
    n_cutoff : int
        Maximum mode to check (default 50)
    
    Returns
    -------
    n_star_max : int
        Max index where condition holds (Definition A)
    K : int
        Count of modes where condition holds (Definition B)
    usable_modes : list
        List of mode indices that pass the condition
    """
    usable_modes = []
    
    n_max_check = min(n_cutoff, len(noise_fourier_coeffs))
    
    for n in range(1, n_max_check + 1):
        rho_power = rho_min ** n
        eta_n = noise_fourier_coeffs[n - 1]  # 0-indexed array
        
        # CORRECT condition: r_min^n > π * n * |η̂_n|
        threshold = np.pi * n * eta_n
        
        if rho_power > threshold:
            usable_modes.append(n)
    
    n_star_max = max(usable_modes) if usable_modes else 0
    K = len(usable_modes)
    
    return n_star_max, K, usable_modes


# =============================================================================
# SOURCE GENERATION (same pattern as test_bound_theory.py)
# =============================================================================

def generate_zero_sum_intensities(n_sources: int) -> np.ndarray:
    """Generate intensities summing to zero: alternating +1/-1."""
    if n_sources % 2 != 0:
        raise ValueError(f"n_sources must be even, got {n_sources}")
    return np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n_sources)])


def generate_sources_at_rho(n_sources: int, rho: float, seed: int) -> List[Tuple[Tuple[float, float], float]]:
    """
    Generate sources in disk at radius rho with well-separated angles.
    """
    np.random.seed(seed)
    
    angles = np.linspace(0, 2 * np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.1, 0.1, n_sources)  # small perturbation
    
    intensities = generate_zero_sum_intensities(n_sources)
    
    sources = []
    for i, theta in enumerate(angles):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        sources.append(((x, y), intensities[i]))
    
    return sources


# =============================================================================
# TRANSITION DETECTION
# =============================================================================

def detect_N_transition_ratio(rmse_dict: Dict[int, float], ratio_threshold: float = 2.0) -> int:
    """
    Detect N_transition using ratio method.
    
    Find first N where RMSE(N+step) / RMSE(N) > ratio_threshold
    
    Returns the N where the jump occurs (the higher N).
    Returns max N if no jump detected.
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
    
    # No clear jump found
    return N_values[-1]


def detect_N_transition_derivative(rmse_dict: Dict[int, float]) -> int:
    """
    Detect N_transition using derivative method.
    
    Find N where d(log RMSE)/dN is maximized.
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
            # d(log RMSE) / dN
            deriv = (np.log(rmse_next) - np.log(rmse_curr)) / (N_next - N_curr)
            
            if deriv > max_deriv:
                max_deriv = deriv
                N_transition = N_next
    
    return N_transition


# =============================================================================
# MAIN VALIDATION FOR ONE SEED
# =============================================================================

def run_single_seed_validation(
    seed: int,
    rho: float = 0.7,
    sigma_noise: float = 0.001,
    n_sensors: int = 100,
    n_restarts: int = 15,
    N_values: List[int] = None
) -> dict:
    """
    Run statistical validation for ONE seed.
    
    Parameters
    ----------
    seed : int
        Random seed for this experiment (determines noise realization)
    rho : float
        Conformal radius for source placement
    sigma_noise : float
        Absolute noise standard deviation
    n_sensors : int
        Number of boundary sensors
    n_restarts : int
        Number of optimizer restarts
    N_values : list
        N values to test (default: [10, 12, 14, 16, 18, 20, 22, 24])
    
    Returns
    -------
    result : dict
        Contains rmse dict, noise analysis, transition detection
    """
    if N_values is None:
        N_values = [10, 12, 14, 16, 18, 20, 22, 24]
    
    start_time = time.time()
    
    # Computed reference values (σ_Four-based prediction)
    sigma_four = compute_sigma_four(sigma_noise, n_sensors)
    n_star_predicted = compute_n_star_predicted(sigma_four, rho)
    N_max_predicted = compute_N_max(n_star_predicted)
    
    # Setup forward solver (sensor locations)
    forward_solver = AnalyticalForwardSolver(n_boundary_points=n_sensors)
    theta_boundary = np.linspace(0, 2 * np.pi, n_sensors, endpoint=False)
    
    # =========================================================================
    # CRITICAL: Generate noise ONCE for this seed, reuse for all N
    # =========================================================================
    np.random.seed(seed)
    noise = sigma_noise * np.random.randn(n_sensors)
    
    # Compute noise Fourier coefficients (n_modes=50 for thorough analysis)
    noise_fourier_coeffs = compute_noise_fourier_coeffs(noise, n_modes=50)
    
    # Compute n* using BOTH methods with CORRECT condition: r^n > π*n*|η̂_n|
    n_star_max, K, usable_modes = compute_n_star_both_methods(noise_fourier_coeffs, rho, n_cutoff=50)
    
    # Compute N_max from both definitions
    N_max_A = (2.0 / 3.0) * n_star_max  # From max index
    N_max_B = (2.0 / 3.0) * K           # From count
    
    # =========================================================================
    # Test each N value with THE SAME noise
    # =========================================================================
    rmse_position_results = {}
    rmse_intensity_results = {}
    
    for N in N_values:
        # Generate sources (use different seed offset to decouple from noise)
        source_seed = seed + 10000 + N  # different for each N to avoid correlation
        sources = generate_sources_at_rho(N, rho, seed=source_seed)
        
        # Forward solve
        u_true = forward_solver.solve(sources)
        
        # Add THE SAME noise (not regenerated!)
        u_measured = u_true + noise
        
        # Inverse solve using exact same function as test_bound_theory.py
        # Note: seed here is for optimizer restarts, not noise
        recovered, residual = solve_inverse_disk(
            u_measured, theta_boundary, n_sources=N,
            n_restarts=n_restarts, seed=seed + 20000 + N
        )
        
        # Convert Source objects to tuples if needed
        if hasattr(recovered[0], 'to_tuple'):
            recovered = [s.to_tuple() for s in recovered]
        
        # Compute all RMSE metrics
        rmse_pos = compute_position_rmse(sources, recovered)
        rmse_int = compute_intensity_rmse(sources, recovered)
        
        rmse_position_results[N] = float(rmse_pos)
        rmse_intensity_results[N] = float(rmse_int)
        
        print(f"  Seed {seed}, N={N}: RMSE_pos = {rmse_pos:.6f}, RMSE_int = {rmse_int:.6f}")
    
    # Detect transition point (based on position RMSE)
    N_transition_ratio = detect_N_transition_ratio(rmse_position_results)
    N_transition_deriv = detect_N_transition_derivative(rmse_position_results)
    
    # Use ratio method as primary
    N_transition = N_transition_ratio
    
    elapsed = time.time() - start_time
    
    # Build result dict with all metrics
    result = {
        'seed': seed,
        'rho_min': float(rho),
        
        # RMSE results
        'rmse_position': rmse_position_results,
        'rmse_intensity': rmse_intensity_results,
        'rmse': rmse_position_results,  # backward compatibility
        
        # Noise Fourier analysis
        'noise_fourier_coeffs': [float(c) for c in noise_fourier_coeffs],
        
        # n* from actual noise (Definition A: max index)
        'n_star_max': int(n_star_max),
        'N_max_A': float(N_max_A),
        
        # n* from actual noise (Definition B: count)
        'K': int(K),
        'N_max_B': float(N_max_B),
        
        # Usable modes list (for visualization)
        'usable_modes': [int(m) for m in usable_modes],
        
        # σ_Four-based prediction (for comparison)
        'n_star_predicted': float(n_star_predicted),
        'N_max_predicted': float(N_max_predicted),
        
        # Backward compatibility aliases
        'n_star_actual': int(n_star_max),  # alias for n_star_max
        'N_max_actual': float(N_max_A),    # alias for N_max_A
        
        # Transition detection
        'N_transition': int(N_transition),
        'N_transition_ratio': int(N_transition_ratio),
        'N_transition_deriv': int(N_transition_deriv),
        
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
    parser.add_argument('--rho', type=float, default=0.7, help='Conformal radius')
    parser.add_argument('--sigma-noise', type=float, default=0.001, help='Noise std dev')
    parser.add_argument('--sensors', type=int, default=100, help='Number of sensors')
    parser.add_argument('--restarts', type=int, default=15, help='Optimizer restarts')
    
    args = parser.parse_args()
    
    print(f"Running statistical validation for seed {args.seed}")
    print(f"  rho={args.rho}, sigma_noise={args.sigma_noise}, sensors={args.sensors}")
    
    result = run_single_seed_validation(
        seed=args.seed,
        rho=args.rho,
        sigma_noise=args.sigma_noise,
        n_sensors=args.sensors,
        n_restarts=args.restarts
    )
    
    print(f"\nResults:")
    print(f"  n*_max (Definition A) = {result['n_star_max']}")
    print(f"  K (Definition B) = {result['K']}")
    print(f"  n*_predicted (σ_Four) = {result['n_star_predicted']:.2f}")
    print(f"  N_max_A = {result['N_max_A']:.2f}")
    print(f"  N_max_B = {result['N_max_B']:.2f}")
    print(f"  N_max_predicted = {result['N_max_predicted']:.2f}")
    print(f"  N_transition = {result['N_transition']}")
    print(f"  Usable modes: {result['usable_modes'][:10]}... ({len(result['usable_modes'])} total)")
    print(f"  Time: {result['time_seconds']:.1f}s")
    print(f"\n  RMSE_position by N: {result['rmse_position']}")
    print(f"  RMSE_intensity by N: {result['rmse_intensity']}")
