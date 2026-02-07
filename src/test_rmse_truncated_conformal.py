#!/usr/bin/env python3
"""
Truncated RMSE Test for Conformal Domains (Ellipse, Brain)
============================================================

Extends test_rmse_truncated.py to ellipse and brain domains using
conformal mapping. The key insight: all Fourier computations happen
in the disk domain after mapping via f: Ω → D.

Data Flow:
- Physical domain: source generation, sensor placement, final RMSE
- Disk domain: Fourier truncation, n* computation, optimization

Sensor Placement:
- Sensors at evenly-spaced DISK angles θ_j = 2πj/M
- Physical locations: z_j = f^{-1}(e^{iθ_j})
- This ensures valid DFT for Fourier coefficient computation

Usage:
    python test_rmse_truncated_conformal.py --seed 0 --domain ellipse
    python test_rmse_truncated_conformal.py --seed 0 --domain brain
"""

import numpy as np
import sys
import time
from typing import List, Tuple, Callable
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')

# Import from existing modules
from conformal_solver import MFSConformalMap, ConformalMap
from mesh import get_brain_boundary

from test_statistical_validation import (
    compute_noise_fourier_coeffs,
    compute_n_star_actual,
    compute_N_max_for_test_case,
    compute_sigma_four,
    compute_n_star_predicted,
    compute_N_values_around_prediction,
)

from test_perturbation import forward_truncated

from test_rmse_truncated import compute_all_rmse


# =============================================================================
# CONFORMAL MAP CREATION
# =============================================================================

def create_ellipse_map(a: float = 1.5, b: float = 0.8, 
                       n_boundary: int = 256, n_charge: int = 200) -> MFSConformalMap:
    """Create MFS conformal map for ellipse with semi-axes a, b."""
    def ellipse_boundary(t):
        return a * np.cos(t) + 1j * b * np.sin(t)
    return MFSConformalMap(ellipse_boundary, n_boundary=n_boundary, n_charge=n_charge)


def create_brain_map(n_boundary: int = 200, n_charge: int = 150) -> MFSConformalMap:
    """Create MFS conformal map for brain domain."""
    boundary_pts = get_brain_boundary(n_boundary)
    z_boundary = boundary_pts[:, 0] + 1j * boundary_pts[:, 1]
    
    t_vals = np.linspace(0, 2 * np.pi, len(z_boundary), endpoint=False)
    real_interp = interp1d(t_vals, z_boundary.real, kind='cubic', fill_value='extrapolate')
    imag_interp = interp1d(t_vals, z_boundary.imag, kind='cubic', fill_value='extrapolate')
    
    def brain_boundary(t):
        t = np.asarray(t) % (2 * np.pi)
        return real_interp(t) + 1j * imag_interp(t)
    
    return MFSConformalMap(brain_boundary, n_boundary=n_boundary, n_charge=n_charge)


def create_conformal_map(domain: str, **kwargs) -> MFSConformalMap:
    """Factory function for conformal maps."""
    if domain == 'ellipse':
        return create_ellipse_map(**kwargs)
    elif domain == 'brain':
        return create_brain_map(**kwargs)
    else:
        raise ValueError(f"Unknown domain: {domain}. Use 'ellipse' or 'brain'.")


# =============================================================================
# SENSOR PLACEMENT (Evenly-spaced in disk, mapped to physical)
# =============================================================================

def get_sensors_from_disk_angles(cmap: ConformalMap, n_sensors: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Place sensors at evenly-spaced DISK angles, map to physical domain.
    
    Returns
    -------
    z_physical : complex array (n_sensors,)
        Sensor locations in physical domain
    w_disk : complex array (n_sensors,)
        Sensor locations in disk (on unit circle)
    """
    # Evenly spaced angles in disk
    theta_disk = np.linspace(0, 2 * np.pi, n_sensors, endpoint=False)
    
    # Points on unit circle in disk
    w_disk = np.exp(1j * theta_disk)
    
    # Map back to physical domain
    z_physical = cmap.from_disk(w_disk)
    
    return z_physical, w_disk


# =============================================================================
# SOURCE GENERATION IN PHYSICAL DOMAIN
# =============================================================================

def generate_sources_ellipse(n_sources: int, r_min: float, r_max: float, 
                              seed: int, cmap: MFSConformalMap,
                              intensity_low: float = 0.5, 
                              intensity_high: float = 2.0) -> Tuple[List, float]:
    """
    Generate sources in ellipse at specified conformal radii range.
    
    Sources are generated in disk coordinates at radii [r_min, r_max],
    then mapped to physical domain.
    
    Returns
    -------
    sources : list of ((x, y), intensity) in PHYSICAL coordinates
    rho_min : minimum conformal radius
    """
    rng = np.random.RandomState(seed)
    
    # Generate in disk coordinates
    radii = rng.uniform(r_min, r_max, n_sources)
    angles = np.linspace(0, 2 * np.pi, n_sources, endpoint=False)
    angles += rng.uniform(-0.2, 0.2, n_sources)  # Small perturbation
    
    # Disk positions
    w_sources = radii * np.exp(1j * angles)
    
    # Map to physical domain
    z_sources = cmap.from_disk(w_sources)
    
    # Generate intensities (random magnitudes, alternating signs, sum to zero)
    magnitudes = rng.uniform(intensity_low, intensity_high, n_sources)
    signs = np.array([(-1) ** i for i in range(n_sources)], dtype=float)
    intensities = magnitudes * signs
    intensities = intensities - np.mean(intensities)  # Ensure sum = 0
    
    # Build source list in physical coordinates
    sources = [((z_sources[k].real, z_sources[k].imag), intensities[k]) 
               for k in range(n_sources)]
    
    rho_min = float(np.min(radii))
    return sources, rho_min


def generate_sources_brain(n_sources: int, r_min: float, r_max: float,
                            seed: int, cmap: MFSConformalMap,
                            intensity_low: float = 0.5,
                            intensity_high: float = 2.0) -> Tuple[List, float]:
    """
    Generate sources in brain domain at specified conformal radii range.
    Same approach as ellipse.
    """
    # Same implementation - conformal map handles the geometry
    return generate_sources_ellipse(n_sources, r_min, r_max, seed, cmap,
                                     intensity_low, intensity_high)


def generate_sources_conformal(domain: str, n_sources: int, r_min: float, r_max: float,
                                seed: int, cmap: MFSConformalMap,
                                intensity_low: float = 0.5,
                                intensity_high: float = 2.0) -> Tuple[List, float]:
    """Factory function for source generation."""
    if domain == 'ellipse':
        return generate_sources_ellipse(n_sources, r_min, r_max, seed, cmap,
                                         intensity_low, intensity_high)
    elif domain == 'brain':
        return generate_sources_brain(n_sources, r_min, r_max, seed, cmap,
                                       intensity_low, intensity_high)
    else:
        raise ValueError(f"Unknown domain: {domain}")


# =============================================================================
# PHYSICAL TO DISK CONVERSION
# =============================================================================

def sources_to_disk_polar(sources: List, cmap: ConformalMap) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert physical sources to disk polar coordinates.
    
    Parameters
    ----------
    sources : list of ((x, y), intensity) in physical coordinates
    cmap : conformal map f: Ω → D
    
    Returns
    -------
    r : array of conformal radii |f(ξ_k)|
    theta : array of disk angles arg(f(ξ_k))
    I : array of intensities (unchanged)
    """
    z_phys = np.array([s[0][0] + 1j * s[0][1] for s in sources])
    I = np.array([s[1] for s in sources])
    
    w_disk = cmap.to_disk(z_phys)
    
    r = np.abs(w_disk)
    theta = np.angle(w_disk)
    
    return r, theta, I


# =============================================================================
# NOISE FOURIER COMPONENTS (computed in disk domain)
# =============================================================================

def compute_noise_fourier_components_conformal(noise: np.ndarray, n_star: int) -> np.ndarray:
    """
    Compute truncated Fourier noise components.
    
    Since sensors are at evenly-spaced disk angles, standard DFT is valid.
    
    Returns
    -------
    eta_trunc : array of shape (2 * n_star,)
        [Re(η̂_1), Im(η̂_1), ..., Re(η̂_n*), Im(η̂_n*)]
    """
    M = len(noise)
    
    # DFT
    fft_coeffs = np.fft.fft(noise) / M
    
    # Extract modes 1 to n*
    eta_trunc = np.zeros(2 * n_star)
    for n in range(1, n_star + 1):
        eta_trunc[2 * (n - 1)] = fft_coeffs[n].real
        eta_trunc[2 * (n - 1) + 1] = fft_coeffs[n].imag
    
    return eta_trunc


# =============================================================================
# TRUNCATED INVERSE SOLVER (operates in disk domain)
# =============================================================================

class TruncatedConformalInverseSolver:
    """
    Nonlinear inverse solver operating on the truncated Fourier system
    for conformal domains.
    
    Optimization is performed in DISK coordinates (r, θ) represented as
    Cartesian (x, y) with constraint x² + y² < 1.
    
    Results are converted back to physical coordinates via f^{-1}.
    """
    
    def __init__(self, n_sources: int, n_star: int, cmap: ConformalMap):
        self.n_sources = n_sources
        self.n_star = n_star
        self.cmap = cmap
        self.d_trunc = None
    
    def set_data(self, d_trunc: np.ndarray):
        """Set truncated Fourier data to fit."""
        assert len(d_trunc) == 2 * self.n_star
        self.d_trunc = d_trunc.copy()
    
    def _params_to_disk_polar(self, params: np.ndarray):
        """Convert Cartesian disk params to polar for forward_truncated."""
        n = self.n_sources
        x = np.array([params[2 * i] for i in range(n)])
        y = np.array([params[2 * i + 1] for i in range(n)])
        q = np.array([params[2 * n + i] for i in range(n)])
        
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        return r, theta, q
    
    def _objective(self, params: np.ndarray) -> float:
        """||h(s) - d_trunc||²"""
        r, theta, q = self._params_to_disk_polar(params)
        h = forward_truncated(r, theta, q, self.n_star)
        return float(np.sum((h - self.d_trunc) ** 2))
    
    def _get_initial_guess(self, strategy: str, restart_idx: int) -> np.ndarray:
        """Generate initial guess in disk coordinates."""
        n = self.n_sources
        rng = np.random.RandomState(restart_idx * 1000 + 7)
        
        if strategy == 'spread':
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            r0 = 0.5
            x0 = r0 * np.cos(angles)
            y0 = r0 * np.sin(angles)
        elif strategy == 'circle':
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n
            r0 = 0.3
            x0 = r0 * np.cos(angles)
            y0 = r0 * np.sin(angles)
        else:  # random
            r0 = rng.uniform(0.1, 0.85, n)
            angles = rng.uniform(0, 2 * np.pi, n)
            x0 = r0 * np.cos(angles)
            y0 = r0 * np.sin(angles)
        
        # Alternating intensities, centered
        q0 = np.array([(-1) ** i for i in range(n)], dtype=float)
        q0 = q0 - np.mean(q0)
        q0 += 0.1 * rng.randn(n)
        q0 = q0 - np.mean(q0)
        
        params = np.zeros(3 * n)
        for i in range(n):
            params[2 * i] = x0[i]
            params[2 * i + 1] = y0[i]
        params[2 * n:] = q0
        return params
    
    def solve(self, n_restarts: int = 15, maxiter: int = 10000,
              seed: int = 42) -> Tuple[List, float]:
        """
        Solve via SLSQP with multistart.
        
        Returns sources in PHYSICAL coordinates.
        """
        if self.d_trunc is None:
            raise ValueError("Call set_data() first")
        
        np.random.seed(seed)
        n = self.n_sources
        
        # Bounds in disk coordinates
        box_bounds = [(-0.95, 0.95)] * (2 * n) + [(-5.0, 5.0)] * n
        
        # Constraints
        def intensity_sum(params):
            return sum(params[2 * n + i] for i in range(n))
        
        def disk_ineq(params):
            # 1 - r² > 0 for each source
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
            # Return worst case in physical coords
            return [((0.0, 0.0), 0.0)] * n, np.inf
        
        # Extract disk coordinates
        params = best_result.x
        w_disk = np.array([params[2 * i] + 1j * params[2 * i + 1] for i in range(n)])
        q = np.array([params[2 * n + i] for i in range(n)])
        
        # Map to physical domain
        z_phys = self.cmap.from_disk(w_disk)
        
        # Build source list in physical coordinates
        sources = [((z_phys[k].real, z_phys[k].imag), q[k]) for k in range(n)]
        
        return sources, best_fun


# =============================================================================
# MAIN VALIDATION FOR ONE SEED
# =============================================================================

def run_single_seed_conformal(
    seed: int,
    domain: str = 'ellipse',
    test_case: str = 'general_random_intensity',
    r_min: float = 0.5,
    r_max: float = 0.7,
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
    ellipse_a: float = 1.5,
    ellipse_b: float = 0.8,
) -> dict:
    """
    Run truncated RMSE test for conformal domain (ellipse or brain).
    
    Returns
    -------
    result : dict
        Complete results including all three RMSE metrics
    """
    start_time = time.time()
    
    # -----------------------------------------------------------------
    # Create conformal map
    # -----------------------------------------------------------------
    if domain == 'ellipse':
        cmap = create_ellipse_map(a=ellipse_a, b=ellipse_b)
    elif domain == 'brain':
        cmap = create_brain_map()
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    # -----------------------------------------------------------------
    # Determine rho_min range
    # -----------------------------------------------------------------
    if random_rho_min:
        rng_rho = np.random.RandomState(seed + 99999)
        rho_min_actual = rng_rho.uniform(rho_min_low, rho_min_high)
        r_min = rho_min_actual
        r_max = min(r_min + 0.2, 0.9)
    else:
        rho_min_actual = r_min
    
    # -----------------------------------------------------------------
    # Place sensors at evenly-spaced disk angles
    # -----------------------------------------------------------------
    z_sensors, w_sensors = get_sensors_from_disk_angles(cmap, n_sensors)
    theta_disk = np.angle(w_sensors)  # For DFT interpretation
    
    # -----------------------------------------------------------------
    # Generate noise ONCE for this seed
    # -----------------------------------------------------------------
    np.random.seed(seed)
    noise = sigma_noise * np.random.randn(n_sensors)
    
    # Compute noise Fourier magnitudes
    noise_fourier_mags = compute_noise_fourier_coeffs(noise, n_modes=50)
    
    # Compute n*
    n_star_max, K, usable_modes = compute_n_star_actual(
        noise_fourier_mags, rho_min_actual, n_cutoff=50
    )
    
    if n_star_max == 0:
        print(f"  WARNING: n_star = 0 for seed {seed}")
        return {
            'seed': int(seed), 'domain': domain, 'test_case': test_case,
            'rho_min': float(rho_min_actual), 'n_star': 0, 'N_max': 0.0,
            'results_by_N': [], 'time_seconds': float(time.time() - start_time),
            'error': 'n_star is zero',
        }
    
    # N_max for this test case
    N_max = compute_N_max_for_test_case(float(n_star_max), test_case)
    
    # Truncated noise components
    noise_trunc = compute_noise_fourier_components_conformal(noise, n_star_max)
    
    # Reference values
    sigma_four = compute_sigma_four(sigma_noise, n_sensors)
    n_star_sigma_four = compute_n_star_predicted(sigma_four, rho_min_actual)
    N_max_sigma_four = compute_N_max_for_test_case(n_star_sigma_four, test_case)
    
    # Determine N values to test
    if N_values is None:
        if use_dynamic_N:
            N_values = compute_N_values_around_prediction(N_max, delta=6, step=2)
        else:
            N_values = [2, 4, 6, 8, 10, 12, 14, 16]
    
    print(f"  Seed {seed} [{domain}]: rho_min={rho_min_actual:.4f}, n*={n_star_max}, "
          f"N_max({test_case})={N_max:.2f}")
    print(f"    N values: {N_values}")
    
    # -----------------------------------------------------------------
    # Test each N value
    # -----------------------------------------------------------------
    results_by_N = []
    
    for N in N_values:
        t0 = time.time()
        
        # Generate sources in physical domain
        source_seed = seed + 10000 + N
        sources, _ = generate_sources_conformal(
            domain, N, r_min, r_max, source_seed, cmap,
            intensity_low=intensity_low, intensity_high=intensity_high
        )
        
        # Convert to disk polar for truncated forward map
        r_true, theta_true, I_true = sources_to_disk_polar(sources, cmap)
        
        # Truncated signal at truth
        signal_trunc = forward_truncated(r_true, theta_true, I_true, n_star_max)
        
        # Truncated data = signal + noise
        data_trunc = signal_trunc + noise_trunc
        
        # --- Solve truncated inverse problem ---
        solver = TruncatedConformalInverseSolver(
            n_sources=N, n_star=n_star_max, cmap=cmap
        )
        solver.set_data(data_trunc)
        
        recovered, residual = solver.solve(
            n_restarts=n_restarts,
            seed=seed + 20000 + N,
        )
        
        # Compute all three RMSE metrics (in physical coordinates)
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
        'domain': domain,
        'test_case': test_case,
        'rho_min': float(rho_min_actual),
        'r_min': float(r_min),
        'r_max': float(r_max),
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
        
        # Domain-specific
        'ellipse_a': float(ellipse_a) if domain == 'ellipse' else None,
        'ellipse_b': float(ellipse_b) if domain == 'ellipse' else None,
        
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
        description="Truncated RMSE test for conformal domains (ellipse, brain)"
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--domain', type=str, default='ellipse',
                        choices=['ellipse', 'brain'])
    parser.add_argument('--test-case', type=str, default='general_random_intensity',
                        choices=['same_radius', 'same_angle', 'general',
                                 'general_random_intensity'])
    parser.add_argument('--r-min', type=float, default=0.5)
    parser.add_argument('--r-max', type=float, default=0.7)
    parser.add_argument('--sigma-noise', type=float, default=0.001)
    parser.add_argument('--sensors', type=int, default=100)
    parser.add_argument('--n-restarts', type=int, default=15)
    parser.add_argument('--random-rho-min', action='store_true')
    parser.add_argument('--rho-min-low', type=float, default=0.5)
    parser.add_argument('--rho-min-high', type=float, default=0.7)
    parser.add_argument('--intensity-low', type=float, default=0.5)
    parser.add_argument('--intensity-high', type=float, default=2.0)
    parser.add_argument('--ellipse-a', type=float, default=1.5)
    parser.add_argument('--ellipse-b', type=float, default=0.8)
    
    args = parser.parse_args()
    
    print(f"Running truncated RMSE test (conformal)")
    print(f"  Seed: {args.seed}, Domain: {args.domain}, Test case: {args.test_case}")
    
    result = run_single_seed_conformal(
        seed=args.seed,
        domain=args.domain,
        test_case=args.test_case,
        r_min=args.r_min,
        r_max=args.r_max,
        sigma_noise=args.sigma_noise,
        n_sensors=args.sensors,
        n_restarts=args.n_restarts,
        random_rho_min=args.random_rho_min,
        rho_min_low=args.rho_min_low,
        rho_min_high=args.rho_min_high,
        intensity_low=args.intensity_low,
        intensity_high=args.intensity_high,
        ellipse_a=args.ellipse_a,
        ellipse_b=args.ellipse_b,
    )
    
    print(f"\n{'='*60}")
    print(f"RESULTS: Seed {args.seed}, Domain: {args.domain}")
    print(f"  n* = {result['n_star']}, N_max = {result['N_max']:.2f}")
    print(f"  Time: {result['time_seconds']:.1f}s")
