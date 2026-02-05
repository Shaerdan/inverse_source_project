#!/usr/bin/env python3
"""
Perturbation Test: Direct Validation of IFT Local Uniqueness (Test A)
======================================================================

Directly tests the theorem's prediction: for N > N_max = (2/3)*n*, the
truncated forward map h: R^{3N} -> R^{2n*} has a Jacobian null space,
meaning there exist perturbation directions that produce negligible change
in the first n* Fourier coefficients.

CRITICAL: Uses the TRUNCATED forward map (first n* Fourier modes only),
NOT the full boundary potential u(theta_j). The null space exists only
in the truncated 2n* system that the theorem is proved on.

Two complementary analyses:
  1. Random perturbation test: perturb in random directions, check if
     the forward map output changes.
  2. Jacobian SVD: compute the Jacobian numerically via finite differences,
     count singular values near zero to determine null space dimension.

Usage:
    python test_perturbation.py --seed 0 --test-case general_random_intensity
"""

import numpy as np
import sys
import time
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')

# Import source generation and n* computation from existing module
from test_statistical_validation import (
    compute_noise_fourier_coeffs,
    compute_n_star_actual,
    compute_N_max_for_test_case,
    compute_sigma_four,
    compute_n_star_predicted,
    generate_sources_for_test_case,
)


# =============================================================================
# TRUNCATED FORWARD MAP
# =============================================================================

def forward_truncated(r, theta, I, n_star):
    """
    Compute first n* Fourier coefficients of the boundary potential.

    Returns vector of length 2*n_star: [a_1, b_1, a_2, b_2, ..., a_n*, b_n*]

    where:
        a_n = (1/(pi*n)) * sum_k I_k * r_k^n * cos(n*theta_k)
        b_n = (1/(pi*n)) * sum_k I_k * r_k^n * sin(n*theta_k)

    Parameters
    ----------
    r : array (N,)
        Source radii
    theta : array (N,)
        Source angles
    I : array (N,)
        Source intensities
    n_star : int
        Number of Fourier modes

    Returns
    -------
    coeffs : array (2*n_star,)
        Fourier coefficients [a_1, b_1, a_2, b_2, ...]
    """
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)
    I = np.asarray(I, dtype=float)

    coeffs = np.zeros(2 * n_star)
    for n in range(1, n_star + 1):
        r_n = r ** n
        a_n = (1.0 / (np.pi * n)) * np.sum(I * r_n * np.cos(n * theta))
        b_n = (1.0 / (np.pi * n)) * np.sum(I * r_n * np.sin(n * theta))
        coeffs[2 * (n - 1)] = a_n
        coeffs[2 * (n - 1) + 1] = b_n
    return coeffs


# =============================================================================
# NOISE FOURIER COMPONENTS (separate a_n, b_n for truncated data)
# =============================================================================

def compute_noise_fourier_components(noise, n_star, n_sensors):
    """
    Compute the Fourier cosine and sine components of the noise,
    matching the truncated forward map ordering.

    Returns vector of length 2*n_star: [eta_a_1, eta_b_1, ..., eta_a_n*, eta_b_n*]

    Uses the same normalisation convention as the forward map:
        eta_a_n = (2/M) * sum_j eta_j * cos(n * theta_j)
        eta_b_n = (2/M) * sum_j eta_j * sin(n * theta_j)

    Parameters
    ----------
    noise : array (M,)
        Noise at sensor locations
    n_star : int
        Number of Fourier modes
    n_sensors : int
        Number of sensors M

    Returns
    -------
    noise_trunc : array (2*n_star,)
        Truncated noise Fourier components
    """
    M = n_sensors
    theta_sensors = np.linspace(0, 2 * np.pi, M, endpoint=False)

    noise_trunc = np.zeros(2 * n_star)
    for n in range(1, n_star + 1):
        eta_a = (2.0 / M) * np.sum(noise * np.cos(n * theta_sensors))
        eta_b = (2.0 / M) * np.sum(noise * np.sin(n * theta_sensors))
        noise_trunc[2 * (n - 1)] = eta_a
        noise_trunc[2 * (n - 1) + 1] = eta_b
    return noise_trunc


# =============================================================================
# PARAMETER PACKING / UNPACKING (polar: [r1, θ1, I1, r2, θ2, I2, ...])
# =============================================================================

def pack_params(r, theta, I):
    """Pack polar source parameters into a single vector.

    Convention: s = [r_1, theta_1, I_1, r_2, theta_2, I_2, ...]
    Length = 3N.
    """
    N = len(r)
    s = np.zeros(3 * N)
    for k in range(N):
        s[3 * k] = r[k]
        s[3 * k + 1] = theta[k]
        s[3 * k + 2] = I[k]
    return s


def unpack_params(s):
    """Unpack parameter vector into (r, theta, I) arrays."""
    N = len(s) // 3
    r = np.array([s[3 * k] for k in range(N)])
    theta = np.array([s[3 * k + 1] for k in range(N)])
    I = np.array([s[3 * k + 2] for k in range(N)])
    return r, theta, I


# =============================================================================
# CARTESIAN-TO-POLAR CONVERSION
# =============================================================================

def sources_to_polar(sources):
    """Convert source list [((x,y), intensity), ...] to (r, theta, I) arrays."""
    r = np.array([np.sqrt(pos[0]**2 + pos[1]**2) for pos, _ in sources])
    theta = np.array([np.arctan2(pos[1], pos[0]) for pos, _ in sources])
    I = np.array([intensity for _, intensity in sources])
    return r, theta, I


# =============================================================================
# PERTURBATION TEST
# =============================================================================

def perturbation_test(s_true, data_trunc, n_star, epsilon, n_directions=200,
                      seed=None):
    """
    Test the truncated forward map for null-space behaviour using
    SVD-informed perturbation analysis.

    Three types of perturbation are tested:
      1. DIRECTED null-space: perturb along each SVD null vector.
         These should give ||Δh||/ε ≈ O(ε) (second-order).
      2. DIRECTED range-space: perturb along each SVD range vector.
         These should give ||Δh||/ε ≈ O(1).
      3. RANDOM: perturb in random directions; decompose the resulting
         forward change into null/range contributions.

    Parameters
    ----------
    s_true : array (3N,)
        True parameter vector in polar form
    data_trunc : array (2*n_star,)
        Truncated Fourier data (signal + noise)
    n_star : int
        Number of Fourier modes
    epsilon : float
        Perturbation magnitude
    n_directions : int
        Number of random directions to test
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    result : dict
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    dim_params = len(s_true)
    dim_output = 2 * n_star

    r_true, theta_true, I_true = unpack_params(s_true)
    h_true = forward_truncated(r_true, theta_true, I_true, n_star)
    residual_true = np.linalg.norm(h_true - data_trunc)

    # ------------------------------------------------------------------
    # Compute Jacobian via finite differences (small step for accuracy)
    # ------------------------------------------------------------------
    fd_eps = 1e-7
    J = np.zeros((dim_output, dim_params))
    for j in range(dim_params):
        s_plus = s_true.copy(); s_plus[j] += fd_eps
        s_minus = s_true.copy(); s_minus[j] -= fd_eps
        r_p, t_p, I_p = unpack_params(s_plus)
        r_m, t_m, I_m = unpack_params(s_minus)
        J[:, j] = (forward_truncated(r_p, t_p, I_p, n_star)
                    - forward_truncated(r_m, t_m, I_m, n_star)) / (2 * fd_eps)

    U, sigma, Vt = np.linalg.svd(J, full_matrices=True)
    # Vt has shape (3N, 3N); rows are right singular vectors
    # Null space = rows of Vt with index >= rank
    sv_threshold = 1e-10 * sigma[0] if len(sigma) > 0 and sigma[0] > 0 else 1e-10
    rank = int(np.sum(sigma > sv_threshold))
    null_dim = dim_params - rank

    # Null-space basis (rows of Vt from rank onwards)
    V_null = Vt[rank:, :]   # shape (null_dim, 3N)
    V_range = Vt[:rank, :]  # shape (rank, 3N)

    # ------------------------------------------------------------------
    # 1. Directed null-space perturbations
    # ------------------------------------------------------------------
    null_fwd_changes = []
    for i in range(null_dim):
        v = V_null[i, :]
        s_pert = s_true + epsilon * v
        r_p, t_p, I_p = unpack_params(s_pert)
        h_pert = forward_truncated(r_p, t_p, I_p, n_star)
        null_fwd_changes.append(np.linalg.norm(h_pert - h_true) / epsilon)
    null_fwd_changes = np.array(null_fwd_changes) if null_fwd_changes else np.array([])

    # ------------------------------------------------------------------
    # 2. Directed range-space perturbations
    # ------------------------------------------------------------------
    range_fwd_changes = []
    for i in range(rank):
        v = V_range[i, :]
        s_pert = s_true + epsilon * v
        r_p, t_p, I_p = unpack_params(s_pert)
        h_pert = forward_truncated(r_p, t_p, I_p, n_star)
        range_fwd_changes.append(np.linalg.norm(h_pert - h_true) / epsilon)
    range_fwd_changes = np.array(range_fwd_changes)

    # ------------------------------------------------------------------
    # 3. Random perturbations with null/range decomposition
    # ------------------------------------------------------------------
    random_fwd_changes = np.zeros(n_directions)
    random_null_fractions = np.zeros(n_directions)
    residual_changes = np.zeros(n_directions)

    for i in range(n_directions):
        direction = rng.randn(dim_params)
        direction = direction / np.linalg.norm(direction)

        s_pert = s_true + epsilon * direction
        r_p, t_p, I_p = unpack_params(s_pert)
        h_pert = forward_truncated(r_p, t_p, I_p, n_star)

        random_fwd_changes[i] = np.linalg.norm(h_pert - h_true) / epsilon

        residual_pert = np.linalg.norm(h_pert - data_trunc)
        residual_changes[i] = residual_pert - residual_true

        # Null-space fraction of this direction
        if null_dim > 0:
            proj_null = V_null @ direction  # shape (null_dim,)
            random_null_fractions[i] = np.sum(proj_null ** 2)  # ||v||=1, so this is fraction
        else:
            random_null_fractions[i] = 0.0

    return {
        # Directed analysis
        'null_fwd_changes': null_fwd_changes,
        'range_fwd_changes': range_fwd_changes,
        # Random analysis
        'random_fwd_changes': random_fwd_changes,
        'random_null_fractions': random_null_fractions,
        'residual_changes': residual_changes,
        'residual_true': residual_true,
        # SVD info (repeated for convenience)
        'rank': rank,
        'null_dim': null_dim,
    }


# =============================================================================
# JACOBIAN SVD ANALYSIS
# =============================================================================

def compute_jacobian_svd(s_true, n_star, fd_eps=1e-7):
    """
    Compute the Jacobian of the truncated forward map via central finite
    differences, then return its SVD.

    J has shape (2*n_star, 3N).  When 3N > 2*n_star, the null space has
    dimension >= 3N - 2*n_star.

    Parameters
    ----------
    s_true : array (3N,)
        Parameter vector
    n_star : int
        Number of Fourier modes
    fd_eps : float
        Finite difference step size

    Returns
    -------
    result : dict with keys:
        singular_values : array
            Singular values of J (sorted descending)
        rank : int
            Numerical rank (singular values > 1e-10 * max)
        null_dim : int
            Null space dimension = 3N - rank
        null_dim_predicted : int
            Predicted null space dim = max(0, 3N - 2*n_star)
        condition_number : float
            Condition number (ratio of largest to smallest nonzero SV)
    """
    dim_params = len(s_true)
    dim_output = 2 * n_star

    J = np.zeros((dim_output, dim_params))

    for j in range(dim_params):
        s_plus = s_true.copy()
        s_minus = s_true.copy()
        s_plus[j] += fd_eps
        s_minus[j] -= fd_eps

        r_p, theta_p, I_p = unpack_params(s_plus)
        r_m, theta_m, I_m = unpack_params(s_minus)

        h_plus = forward_truncated(r_p, theta_p, I_p, n_star)
        h_minus = forward_truncated(r_m, theta_m, I_m, n_star)

        J[:, j] = (h_plus - h_minus) / (2 * fd_eps)

    # SVD
    U, sigma, Vt = np.linalg.svd(J, full_matrices=False)

    # Numerical rank
    sv_threshold = 1e-10 * sigma[0] if len(sigma) > 0 and sigma[0] > 0 else 1e-10
    rank = int(np.sum(sigma > sv_threshold))
    null_dim = dim_params - rank
    null_dim_predicted = max(0, dim_params - dim_output)

    # Condition number (of the rank-deficient part)
    nonzero_svs = sigma[sigma > sv_threshold]
    if len(nonzero_svs) > 1:
        condition_number = float(nonzero_svs[0] / nonzero_svs[-1])
    else:
        condition_number = float('inf')

    return {
        'singular_values': sigma,
        'rank': rank,
        'null_dim': null_dim,
        'null_dim_predicted': null_dim_predicted,
        'condition_number': condition_number,
        'jacobian_shape': (dim_output, dim_params),
    }


# =============================================================================
# MAIN VALIDATION FOR ONE SEED
# =============================================================================

def run_single_seed_perturbation(
    seed: int,
    test_case: str = 'general_random_intensity',
    rho: float = 0.7,
    r_min: float = 0.5,
    r_max: float = 0.9,
    theta_0: float = 0.0,
    sigma_noise: float = 0.001,
    n_sensors: int = 100,
    N_values: List[int] = None,
    epsilons: List[float] = None,
    n_directions: int = 200,
    random_rho_min: bool = False,
    rho_min_low: float = 0.5,
    rho_min_high: float = 0.7,
    intensity_low: float = 0.5,
    intensity_high: float = 2.0,
) -> dict:
    """
    Run perturbation test (Test A) for ONE seed.

    For each N value, generates sources, computes the truncated forward map
    and truncated noisy data, then runs:
      1. Perturbation test at each epsilon
      2. Jacobian SVD analysis

    Parameters
    ----------
    seed : int
        Random seed
    test_case : str
        Source configuration type
    rho, r_min, r_max, theta_0 : float
        Source geometry parameters
    sigma_noise : float
        Noise standard deviation
    n_sensors : int
        Number of boundary sensors (M)
    N_values : list of int
        Number of sources to test (default: [2, 4, 6, 8, 10, 12, 14, 16])
    epsilons : list of float
        Perturbation magnitudes (default: [1e-7, 0.01, 0.05])
    n_directions : int
        Number of random perturbation directions
    random_rho_min : bool
        If True, randomly sample rho_min per seed
    rho_min_low, rho_min_high : float
        Range for random rho_min sampling
    intensity_low, intensity_high : float
        Intensity magnitude range for general_random_intensity

    Returns
    -------
    result : dict
        Complete results for this seed
    """
    start_time = time.time()

    if N_values is None:
        N_values = [2, 4, 6, 8, 10, 12, 14, 16]

    if epsilons is None:
        epsilons = [1e-7, 0.01, 0.05]

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
    # Generate noise ONCE for this seed (same as RMSE pipeline)
    # -----------------------------------------------------------------
    np.random.seed(seed)
    noise = sigma_noise * np.random.randn(n_sensors)

    # Compute noise Fourier magnitudes for n* (reuse existing function)
    noise_fourier_magnitudes = compute_noise_fourier_coeffs(noise, n_modes=50)

    # Compute n*
    n_star_max, K, usable_modes = compute_n_star_actual(
        noise_fourier_magnitudes, rho_min, n_cutoff=50
    )

    if n_star_max == 0:
        print(f"  WARNING: n_star = 0 for seed {seed}, rho_min = {rho_min:.4f}")
        elapsed = time.time() - start_time
        return {
            'seed': int(seed),
            'test_case': test_case,
            'rho_min': float(rho_min),
            'n_star': 0,
            'N_max': 0.0,
            'results_by_N': [],
            'time_seconds': float(elapsed),
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

    print(f"  Seed {seed}: rho_min={rho_min:.4f}, n*={n_star_max}, "
          f"N_max({test_case})={N_max:.2f}")
    print(f"    Epsilons: {epsilons}, directions: {n_directions}")

    # -----------------------------------------------------------------
    # Test each N value
    # -----------------------------------------------------------------
    results_by_N = []

    for N in N_values:
        # Generate sources (same seeding as RMSE pipeline)
        source_seed = seed + 10000 + N
        sources, _ = generate_sources_for_test_case(
            test_case, N, source_seed,
            rho=rho, r_min=r_min, r_max=r_max, theta_0=theta_0,
            intensity_low=intensity_low, intensity_high=intensity_high,
        )

        # Convert to polar
        r_true, theta_true, I_true = sources_to_polar(sources)
        s_true = pack_params(r_true, theta_true, I_true)

        # Truncated signal at truth
        signal_trunc = forward_truncated(r_true, theta_true, I_true, n_star_max)

        # Truncated data = signal + noise (both truncated to same n*)
        data_trunc = signal_trunc + noise_trunc

        # --- Jacobian SVD analysis ---
        svd_result = compute_jacobian_svd(s_true, n_star_max)

        # --- Perturbation test at each epsilon ---
        perturbation_results = {}
        for eps in epsilons:
            pert = perturbation_test(
                s_true, data_trunc, n_star_max, eps, n_directions,
                seed=seed + 30000 + N + int(eps * 1e10)
            )

            # Key metrics
            perturbation_results[f'eps_{eps}'] = {
                'epsilon': float(eps),
                # Directed null-space perturbations (should be ~O(eps))
                'null_fwd_max': float(np.max(pert['null_fwd_changes'])) if len(pert['null_fwd_changes']) > 0 else None,
                'null_fwd_median': float(np.median(pert['null_fwd_changes'])) if len(pert['null_fwd_changes']) > 0 else None,
                'null_fwd_all': [float(v) for v in pert['null_fwd_changes']] if len(pert['null_fwd_changes']) <= 50 else [float(v) for v in pert['null_fwd_changes'][:50]],
                # Directed range-space perturbations (should be ~O(1))
                'range_fwd_min': float(np.min(pert['range_fwd_changes'])) if len(pert['range_fwd_changes']) > 0 else None,
                'range_fwd_median': float(np.median(pert['range_fwd_changes'])) if len(pert['range_fwd_changes']) > 0 else None,
                # Separation ratio: median(range) / max(null)
                'separation_ratio': (float(np.median(pert['range_fwd_changes']) /
                                           np.max(pert['null_fwd_changes']))
                                     if len(pert['null_fwd_changes']) > 0 and np.max(pert['null_fwd_changes']) > 0
                                     else None),
                # Random direction analysis
                'random_fwd_min': float(np.min(pert['random_fwd_changes'])),
                'random_fwd_median': float(np.median(pert['random_fwd_changes'])),
                'residual_true': float(pert['residual_true']),
            }

        # Build per-N result
        N_result = {
            'N': int(N),
            'n_star': int(n_star_max),
            'N_max': float(N_max),
            'N_minus_Nmax': float(N - N_max),
            'dim_params': int(3 * N),
            'dim_equations': int(2 * n_star_max),

            # Jacobian SVD
            'svd_rank': int(svd_result['rank']),
            'svd_null_dim': int(svd_result['null_dim']),
            'null_dim_predicted': int(svd_result['null_dim_predicted']),
            'svd_singular_values': [float(v) for v in svd_result['singular_values']],
            'svd_condition_number': float(svd_result['condition_number']),

            # Perturbation results per epsilon
            'perturbation': perturbation_results,
        }

        # Summary line using the smallest epsilon (most accurate linearised analysis)
        eps_key = f'eps_{epsilons[0]}'
        null_max = perturbation_results[eps_key]['null_fwd_max']
        range_med = perturbation_results[eps_key]['range_fwd_median']
        sep = perturbation_results[eps_key]['separation_ratio']
        null_str = f"null_max={null_max:.2e}" if null_max is not None else "null_max=N/A"
        range_str = f"range_med={range_med:.2e}" if range_med is not None else "range_med=N/A"
        sep_str = f"sep={sep:.0f}x" if sep is not None else "sep=N/A"
        print(f"    N={N:2d}: 3N={3*N:2d} vs 2n*={2*n_star_max:2d} | "
              f"SVD rank={svd_result['rank']:2d}, null={svd_result['null_dim']:2d} "
              f"(pred={svd_result['null_dim_predicted']:2d}) | "
              f"{null_str}, {range_str}, {sep_str}")

        results_by_N.append(N_result)

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
        'epsilons': [float(e) for e in epsilons],
        'n_directions': int(n_directions),

        # Reference values
        'sigma_noise': float(sigma_noise),
        'n_sensors': int(n_sensors),
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
        description="Perturbation test (Test A) for one seed"
    )
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--test-case', type=str, default='general_random_intensity',
                        choices=['same_radius', 'same_angle', 'general',
                                 'general_random_intensity'],
                        help='Test case type')
    parser.add_argument('--rho', type=float, default=0.7,
                        help='Common radius (same_radius case)')
    parser.add_argument('--r-min', type=float, default=0.5,
                        help='Min radius (same_angle, general)')
    parser.add_argument('--r-max', type=float, default=0.9,
                        help='Max radius (same_angle, general)')
    parser.add_argument('--theta-0', type=float, default=0.0,
                        help='Common angle (same_angle case)')
    parser.add_argument('--sigma-noise', type=float, default=0.001,
                        help='Noise std dev')
    parser.add_argument('--sensors', type=int, default=100,
                        help='Number of sensors')
    parser.add_argument('--n-directions', type=int, default=200,
                        help='Number of random perturbation directions')
    parser.add_argument('--random-rho-min', action='store_true',
                        help='Randomly sample rho_min per seed')
    parser.add_argument('--rho-min-low', type=float, default=0.5,
                        help='Lower bound for random rho_min')
    parser.add_argument('--rho-min-high', type=float, default=0.7,
                        help='Upper bound for random rho_min')
    parser.add_argument('--intensity-low', type=float, default=0.5,
                        help='Min intensity magnitude')
    parser.add_argument('--intensity-high', type=float, default=2.0,
                        help='Max intensity magnitude')

    args = parser.parse_args()

    print(f"Running perturbation test (Test A)")
    print(f"  Seed: {args.seed}")
    print(f"  Test case: {args.test_case}")
    print(f"  sigma_noise: {args.sigma_noise}, sensors: {args.sensors}")
    if args.random_rho_min:
        print(f"  Random rho_min: [{args.rho_min_low}, {args.rho_min_high}]")

    result = run_single_seed_perturbation(
        seed=args.seed,
        test_case=args.test_case,
        rho=args.rho,
        r_min=args.r_min,
        r_max=args.r_max,
        theta_0=args.theta_0,
        sigma_noise=args.sigma_noise,
        n_sensors=args.sensors,
        n_directions=args.n_directions,
        random_rho_min=args.random_rho_min,
        rho_min_low=args.rho_min_low,
        rho_min_high=args.rho_min_high,
        intensity_low=args.intensity_low,
        intensity_high=args.intensity_high,
    )

    print(f"\n{'='*60}")
    print(f"RESULTS: Seed {args.seed}, Test Case: {args.test_case}")
    print(f"{'='*60}")
    print(f"  rho_min = {result['rho_min']:.4f}")
    print(f"  n* = {result['n_star']}")
    print(f"  N_max = {result['N_max']:.2f}")
    print(f"  Time: {result['time_seconds']:.1f}s")
