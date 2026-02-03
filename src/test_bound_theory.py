#!/usr/bin/env python3
"""
Test Script: Validating the Theoretical Source Number Bound
============================================================

This script tests the main theoretical result:

    N ≤ (2/3) * log(σ_Four) / log(ρ_min)

where:
- N = number of sources
- σ_Four = noise level in Fourier domain
- ρ_min = minimum conformal radius (= r for disk, = |f(ξ)| for general domains)

The test:
1. Fix σ_Four (via sensor count M and relative noise σ_rel)
2. Fix ρ_min (source depth in conformal coordinates)
3. Compute N_max from the bound
4. Test N = floor(N_max) - 1 → should succeed (RMSE < threshold)
5. Test N = floor(N_max) + 1 → should fail (RMSE > threshold)

Usage:
    python test_bound_theory.py [--domain disk|ellipse|brain] [--rho 0.7]
"""

import numpy as np
import sys
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, '.')

from analytical_solver import (
    AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver,
    greens_function_disk_neumann
)
from conformal_solver import (
    MFSConformalMap, ConformalForwardSolver, ConformalNonlinearInverseSolver,
    create_conformal_map
)
from domains import get_domain, DiskDomain, EllipseDomain, BrainDomain


# =============================================================================
# THEORETICAL BOUND COMPUTATION
# =============================================================================

def compute_sigma_four(sigma_noise: float, n_sensors: int) -> float:
    """
    Compute the noise level in the Fourier domain.
    
    Parameters
    ----------
    sigma_noise : float
        Absolute noise standard deviation (not relative!)
    n_sensors : int
        Number of sensor points on boundary
    
    Returns
    -------
    sigma_four : float
        Standard deviation of noise in Fourier coefficients
        
    Notes
    -----
    If noise is added as: u_meas = u_true + σ_noise * randn(M)
    Then by DFT/Parseval: σ_Four ≈ σ_noise / sqrt(M)
    
    This is independent of the signal magnitude - exactly as in the theory!
    """
    sigma_four = sigma_noise / np.sqrt(n_sensors)
    return sigma_four


def compute_N_max(sigma_four: float, rho_min: float) -> float:
    """
    Compute the theoretical maximum number of recoverable sources.
    
    N_max = (2/3) * log(σ_Four) / log(ρ_min)
    
    Parameters
    ----------
    sigma_four : float
        Noise level in Fourier domain (must be < 1)
    rho_min : float
        Minimum conformal radius (must be in (0, 1))
    
    Returns
    -------
    N_max : float
        Theoretical bound (may be non-integer)
    """
    if sigma_four >= 1 or sigma_four <= 0:
        raise ValueError(f"sigma_four must be in (0, 1), got {sigma_four}")
    if rho_min >= 1 or rho_min <= 0:
        raise ValueError(f"rho_min must be in (0, 1), got {rho_min}")
    
    # Both log terms are negative, so ratio is positive
    N_max = (2/3) * np.log(sigma_four) / np.log(rho_min)
    return N_max


def compute_rho_min_for_N(N: int, sigma_four: float) -> float:
    """
    Compute the minimum conformal radius needed to recover N sources.
    
    Inverts: N = (2/3) * log(σ_Four) / log(ρ_min)
    To get: ρ_min = exp((2/3) * log(σ_Four) / N) = σ_Four^(2/(3N))
    """
    return sigma_four ** (2 / (3 * N))


# =============================================================================
# CONFORMAL RADIUS COMPUTATION
# =============================================================================

def compute_conformal_radius_disk(sources: List[Tuple[Tuple[float, float], float]]) -> np.ndarray:
    """For the disk, conformal radius = Euclidean radius."""
    radii = []
    for (x, y), _ in sources:
        radii.append(np.sqrt(x**2 + y**2))
    return np.array(radii)


def compute_conformal_radius_ellipse(sources: List[Tuple[Tuple[float, float], float]], 
                                      a: float = 1.5, b: float = 0.8) -> np.ndarray:
    """Compute conformal radii for ellipse using MFS conformal map."""
    def ellipse_boundary(t):
        return a * np.cos(t) + 1j * b * np.sin(t)
    
    cmap = MFSConformalMap(ellipse_boundary, n_boundary=256, n_charge=200)
    
    radii = []
    for (x, y), _ in sources:
        z = x + 1j * y
        w = cmap.to_disk(z)
        radii.append(np.abs(w))
    return np.array(radii)


def compute_conformal_radius_brain(sources: List[Tuple[Tuple[float, float], float]]) -> np.ndarray:
    """Compute conformal radii for brain using MFS conformal map."""
    from mesh import get_brain_boundary
    
    # Get brain boundary and create parametric function
    boundary_pts = get_brain_boundary(200)
    z_boundary = boundary_pts[:, 0] + 1j * boundary_pts[:, 1]
    
    # Create interpolated boundary function
    t_vals = np.linspace(0, 2*np.pi, len(z_boundary), endpoint=False)
    from scipy.interpolate import interp1d
    real_interp = interp1d(t_vals, z_boundary.real, kind='cubic', fill_value='extrapolate')
    imag_interp = interp1d(t_vals, z_boundary.imag, kind='cubic', fill_value='extrapolate')
    
    def brain_boundary(t):
        t = t % (2*np.pi)
        return real_interp(t) + 1j * imag_interp(t)
    
    cmap = MFSConformalMap(brain_boundary, n_boundary=200, n_charge=150)
    
    radii = []
    for (x, y), _ in sources:
        z = x + 1j * y
        w = cmap.to_disk(z)
        radii.append(np.abs(w))
    return np.array(radii)


# =============================================================================
# SOURCE GENERATION AT FIXED CONFORMAL DEPTH
# =============================================================================

def generate_zero_sum_intensities(n_sources: int, seed: int = 42) -> np.ndarray:
    """
    Generate n_sources intensities that sum to zero.
    
    Uses alternating +1, -1 pattern.
    NOTE: This only works correctly for even n_sources!
    """
    if n_sources % 2 != 0:
        raise ValueError(f"n_sources must be even for clean ±1 intensities, got {n_sources}")
    
    intensities = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n_sources)])
    return intensities


def generate_sources_at_conformal_radius_disk(n_sources: int, rho_target: float, 
                                               seed: int = 42) -> List[Tuple[Tuple[float, float], float]]:
    """
    Generate sources in disk with specified conformal radius (= Euclidean radius).
    
    Sources are evenly spread in angle with intensities summing to zero.
    """
    np.random.seed(seed)
    
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.1, 0.1, n_sources)  # Small perturbation
    
    intensities = generate_zero_sum_intensities(n_sources, seed + 1000)
    
    sources = []
    for i, theta in enumerate(angles):
        x = rho_target * np.cos(theta)
        y = rho_target * np.sin(theta)
        sources.append(((x, y), intensities[i]))
    
    return sources


def generate_sources_at_conformal_radius_ellipse(n_sources: int, rho_target: float,
                                                   a: float = 1.5, b: float = 0.8,
                                                   seed: int = 42) -> List[Tuple[Tuple[float, float], float]]:
    """
    Generate sources in ellipse with specified conformal radius.
    
    Uses inverse conformal map to place sources at target conformal depth.
    """
    np.random.seed(seed)
    
    def ellipse_boundary(t):
        return a * np.cos(t) + 1j * b * np.sin(t)
    
    cmap = MFSConformalMap(ellipse_boundary, n_boundary=256, n_charge=200)
    
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.1, 0.1, n_sources)
    
    intensities = generate_zero_sum_intensities(n_sources, seed + 1000)
    
    sources = []
    for i, theta in enumerate(angles):
        # Target point in disk coordinates
        w_target = rho_target * np.exp(1j * theta)
        # Map to ellipse
        z = cmap.from_disk(w_target)
        x, y = z.real, z.imag
        sources.append(((x, y), intensities[i]))
    
    return sources


def generate_sources_at_conformal_radius_brain(n_sources: int, rho_target: float,
                                                 seed: int = 42) -> List[Tuple[Tuple[float, float], float]]:
    """Generate sources in brain with specified conformal radius."""
    np.random.seed(seed)
    
    from mesh import get_brain_boundary
    boundary_pts = get_brain_boundary(200)
    z_boundary = boundary_pts[:, 0] + 1j * boundary_pts[:, 1]
    
    t_vals = np.linspace(0, 2*np.pi, len(z_boundary), endpoint=False)
    from scipy.interpolate import interp1d
    real_interp = interp1d(t_vals, z_boundary.real, kind='cubic', fill_value='extrapolate')
    imag_interp = interp1d(t_vals, z_boundary.imag, kind='cubic', fill_value='extrapolate')
    
    def brain_boundary(t):
        t = t % (2*np.pi)
        return real_interp(t) + 1j * imag_interp(t)
    
    cmap = MFSConformalMap(brain_boundary, n_boundary=200, n_charge=150)
    
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.1, 0.1, n_sources)
    
    intensities = generate_zero_sum_intensities(n_sources, seed + 1000)
    
    sources = []
    for i, theta in enumerate(angles):
        w_target = rho_target * np.exp(1j * theta)
        z = cmap.from_disk(w_target)
        x, y = z.real, z.imag
        sources.append(((x, y), intensities[i]))
    
    return sources


# =============================================================================
# INVERSE SOLVER
# =============================================================================

def solve_inverse_disk(u_measured: np.ndarray, theta_boundary: np.ndarray,
                        n_sources: int, n_restarts: int = 10,
                        seed: int = 42) -> Tuple[List, float]:
    """Solve inverse problem for disk using analytical forward."""
    np.random.seed(seed)
    
    solver = AnalyticalNonlinearInverseSolver(
        n_sources=n_sources,
        n_boundary=len(theta_boundary)
    )
    solver.set_measured_data(u_measured)
    
    result = solver.solve(method='SLSQP', n_restarts=n_restarts)
    
    return result.sources, result.residual


def solve_inverse_conformal(u_measured: np.ndarray, 
                             cmap,
                             sensor_locations: np.ndarray,
                             n_sources: int, 
                             n_restarts: int = 10,
                             seed: int = 42) -> Tuple[List, float]:
    """Solve inverse problem using conformal mapping approach."""
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
# POSITION ERROR COMPUTATION
# =============================================================================

def compute_position_rmse(true_sources: List, recovered_sources: List) -> float:
    """
    Compute RMSE between true and recovered source positions.
    Uses optimal matching (Hungarian algorithm).
    
    Handles both tuple format ((x,y), q) and Source objects.
    """
    from scipy.optimize import linear_sum_assignment
    
    n = len(true_sources)
    if len(recovered_sources) != n:
        return np.inf
    
    # Extract positions from true sources
    true_pos = []
    for s in true_sources:
        if hasattr(s, 'x'):  # Source object
            true_pos.append((s.x, s.y))
        else:  # Tuple format
            true_pos.append(s[0])
    
    # Extract positions from recovered sources
    rec_pos = []
    for s in recovered_sources:
        if hasattr(s, 'x'):  # Source object
            rec_pos.append((s.x, s.y))
        else:  # Tuple format
            rec_pos.append(s[0])
    
    # Build cost matrix
    cost = np.zeros((n, n))
    for i, (x1, y1) in enumerate(true_pos):
        for j, (x2, y2) in enumerate(rec_pos):
            cost[i, j] = (x1 - x2)**2 + (y1 - y2)**2
    
    # Optimal matching
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Compute RMSE
    total_sq_error = sum(cost[i, j] for i, j in zip(row_ind, col_ind))
    rmse = np.sqrt(total_sq_error / n)
    
    return rmse


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_single_test(domain: str, sources: List, recovered: List, 
                     rho_target: float, N_max: float, rmse: float,
                     sensor_locations: np.ndarray = None,
                     boundary_func=None, save_path: str = None):
    """
    Plot true vs recovered sources for a single test.
    
    Parameters
    ----------
    domain : str
        Domain name
    sources : List
        True sources
    recovered : List
        Recovered sources
    rho_target : float
        Target conformal radius
    N_max : float
        Theoretical maximum N
    rmse : float
        Position RMSE
    sensor_locations : np.ndarray, optional
        Sensor positions for plotting boundary
    boundary_func : callable, optional
        Function to generate boundary points
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Plot boundary
    if domain == 'disk':
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, label='Boundary')
        # Plot conformal radius circle
        ax.plot(rho_target * np.cos(theta), rho_target * np.sin(theta), 
                'g--', linewidth=1, alpha=0.5, label=f'ρ = {rho_target}')
    elif sensor_locations is not None:
        # Use sensor locations to approximate boundary
        ax.plot(sensor_locations[:, 0], sensor_locations[:, 1], 
                'k-', linewidth=2, label='Boundary')
    
    # Extract positions and intensities
    true_pos = [s[0] if isinstance(s, tuple) else (s.x, s.y) for s in sources]
    true_int = [s[1] if isinstance(s, tuple) else s.intensity for s in sources]
    rec_pos = [s[0] if isinstance(s, tuple) else (s.x, s.y) for s in recovered]
    rec_int = [s[1] if isinstance(s, tuple) else s.intensity for s in recovered]
    
    # Plot true sources
    for i, ((x, y), I) in enumerate(zip(true_pos, true_int)):
        color = 'red' if I > 0 else 'blue'
        ax.scatter(x, y, c=color, s=200, marker='o', edgecolors='black', 
                   linewidths=2, zorder=10, label='True' if i == 0 else '')
    
    # Plot recovered sources
    for i, ((x, y), I) in enumerate(zip(rec_pos, rec_int)):
        color = 'orange' if I > 0 else 'cyan'
        ax.scatter(x, y, c=color, s=150, marker='x', linewidths=3, 
                   zorder=11, label='Recovered' if i == 0 else '')
    
    # Draw lines connecting true to matched recovered
    from scipy.optimize import linear_sum_assignment
    n = len(true_pos)
    cost = np.zeros((n, n))
    for i, (x1, y1) in enumerate(true_pos):
        for j, (x2, y2) in enumerate(rec_pos):
            cost[i, j] = (x1 - x2)**2 + (y1 - y2)**2
    row_ind, col_ind = linear_sum_assignment(cost)
    
    for i, j in zip(row_ind, col_ind):
        x1, y1 = true_pos[i]
        x2, y2 = rec_pos[j]
        ax.plot([x1, x2], [y1, y2], 'g--', alpha=0.5, linewidth=1)
    
    N = len(sources)
    status = "SUCCESS" if rmse < 0.05 else "FAILED"
    relation = ">" if N > N_max else "≤"
    
    ax.set_title(f'{domain.upper()}: N={N} (N_max={N_max:.2f}, N{relation}N_max)\n'
                 f'RMSE={rmse:.2e} - {status}', fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def plot_summary(results: List, domain: str, save_path: str = None):
    """
    Plot summary: RMSE vs N with theoretical bound marked.
    
    Parameters
    ----------
    results : List[TestResult]
        List of test results
    domain : str
        Domain name
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    N_values = [r.N_tested for r in results]
    rmse_values = [r.position_rmse for r in results]
    N_max = results[0].N_max_theory
    
    # Color by success/failure
    colors = ['green' if r.actual_success else 'red' for r in results]
    
    ax.scatter(N_values, rmse_values, c=colors, s=150, zorder=10, edgecolors='black')
    ax.plot(N_values, rmse_values, 'b--', alpha=0.5, zorder=5)
    
    # Mark theoretical bound
    ax.axvline(x=N_max, color='purple', linestyle='--', linewidth=2, 
               label=f'N_max = {N_max:.2f}')
    
    # Mark threshold
    ax.axhline(y=0.05, color='orange', linestyle=':', linewidth=2,
               label='Success threshold (0.05)')
    
    ax.set_xlabel('Number of Sources (N)', fontsize=12)
    ax.set_ylabel('Position RMSE', fontsize=12)
    ax.set_title(f'{domain.upper()}: RMSE vs N\n'
                 f'Theory: N ≤ {N_max:.2f} for recovery', fontsize=14)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    for r in results:
        status = "✓" if r.actual_success else "✗"
        ax.annotate(status, (r.N_tested, r.position_rmse), 
                   textcoords="offset points", xytext=(0, 10),
                   ha='center', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


# =============================================================================
# TEST RUNNER
# =============================================================================

@dataclass
class TestResult:
    """Container for test results."""
    domain: str
    rho_target: float
    rho_actual: np.ndarray
    sigma_four: float
    N_max_theory: float
    N_tested: int
    position_rmse: float
    expected_success: bool
    actual_success: bool
    time_seconds: float
    # Added for visualization
    true_sources: List = None
    recovered_sources: List = None
    sensor_locations: np.ndarray = None


def run_test(domain: str, n_sources: int, rho_target: float,
             sigma_noise: float = 0.001, n_sensors: int = 100,
             n_restarts: int = 10, seed: int = 42,
             rmse_threshold: float = 0.05) -> TestResult:
    """
    Run a single test of the theoretical bound.
    
    Parameters
    ----------
    domain : str
        'disk', 'ellipse', or 'brain'
    n_sources : int
        Number of sources to recover
    rho_target : float
        Target conformal radius for source placement
    sigma_noise : float
        Absolute noise standard deviation (not relative!)
    n_sensors : int
        Number of sensors
    n_restarts : int
        Number of optimization restarts
    seed : int
        Random seed
    rmse_threshold : float
        Success threshold for position RMSE
    
    Returns
    -------
    result : TestResult
    """
    np.random.seed(seed)
    start_time = time.time()
    
    # Generate sources at target conformal depth
    if domain == 'disk':
        sources = generate_sources_at_conformal_radius_disk(n_sources, rho_target, seed)
        rho_actual = compute_conformal_radius_disk(sources)
    elif domain == 'ellipse':
        sources = generate_sources_at_conformal_radius_ellipse(n_sources, rho_target, seed=seed)
        rho_actual = compute_conformal_radius_ellipse(sources)
    elif domain == 'brain':
        sources = generate_sources_at_conformal_radius_brain(n_sources, rho_target, seed=seed)
        rho_actual = compute_conformal_radius_brain(sources)
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    rho_min = np.min(rho_actual)
    
    # Forward solve
    if domain == 'disk':
        theta = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
        solver = AnalyticalForwardSolver(n_boundary_points=n_sensors)
        u_true = solver.solve(sources)
        sensor_locations = np.column_stack([np.cos(theta), np.sin(theta)])
    else:
        # Conformal solver
        if domain == 'ellipse':
            cmap = create_conformal_map('ellipse', a=1.5, b=0.8)
        else:  # brain
            from mesh import get_brain_boundary
            boundary_pts = get_brain_boundary(200)
            z_boundary = boundary_pts[:, 0] + 1j * boundary_pts[:, 1]
            t_vals = np.linspace(0, 2*np.pi, len(z_boundary), endpoint=False)
            from scipy.interpolate import interp1d
            real_interp = interp1d(t_vals, z_boundary.real, kind='cubic', fill_value='extrapolate')
            imag_interp = interp1d(t_vals, z_boundary.imag, kind='cubic', fill_value='extrapolate')
            def brain_boundary(t):
                t = t % (2*np.pi)
                return real_interp(t) + 1j * imag_interp(t)
            cmap = MFSConformalMap(brain_boundary, n_boundary=200, n_charge=150)
        
        fwd_solver = ConformalForwardSolver(cmap, n_sensors)
        u_true = fwd_solver.solve(sources)
        sensor_locations = fwd_solver.boundary_points
    
    # Compute noise level - ABSOLUTE (independent of signal magnitude)
    sigma_four = compute_sigma_four(sigma_noise, n_sensors)
    
    # Compute theoretical bound using ACTUAL rho_min
    N_max = compute_N_max(sigma_four, rho_min)
    
    # Add noise - ABSOLUTE (exactly as theory assumes)
    noise = sigma_noise * np.random.randn(n_sensors)
    u_measured = u_true + noise
    
    # Solve inverse problem
    if domain == 'disk':
        theta = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
        recovered, residual = solve_inverse_disk(u_measured, theta, n_sources, 
                                                  n_restarts=n_restarts, seed=seed+100)
    else:
        recovered, residual = solve_inverse_conformal(u_measured, cmap, sensor_locations,
                                                       n_sources, n_restarts=n_restarts, 
                                                       seed=seed+100)
    
    # Compute position error
    rmse = compute_position_rmse(sources, recovered)
    
    elapsed = time.time() - start_time
    
    # Determine expected success
    expected_success = n_sources <= N_max
    actual_success = rmse < rmse_threshold
    
    return TestResult(
        domain=domain,
        rho_target=rho_target,
        rho_actual=rho_actual,
        sigma_four=sigma_four,
        N_max_theory=N_max,
        N_tested=n_sources,
        position_rmse=rmse,
        expected_success=expected_success,
        actual_success=actual_success,
        time_seconds=elapsed,
        true_sources=sources,
        recovered_sources=recovered,
        sensor_locations=sensor_locations
    )


def run_bound_validation(domain: str, rho_target: float = 0.7,
                          sigma_noise: float = 0.001, n_sensors: int = 100,
                          n_restarts: int = 15, seed: int = 42,
                          rmse_threshold: float = 0.05,
                          save_plots: bool = True,
                          output_dir: str = '.'):
    """
    Run complete bound validation for a domain.
    
    Tests N values around the theoretical bound N_max.
    
    Parameters
    ----------
    sigma_noise : float
        Absolute noise standard deviation (not relative!)
        σ_Four = σ_noise / √M
    """
    import os
    
    print(f"\n{'='*70}")
    print(f"BOUND VALIDATION: {domain.upper()} DOMAIN")
    print(f"{'='*70}")
    
    # First, estimate N_max with pilot sources
    pilot_sources = generate_sources_at_conformal_radius_disk(2, rho_target, seed) if domain == 'disk' \
        else generate_sources_at_conformal_radius_ellipse(2, rho_target, seed=seed) if domain == 'ellipse' \
        else generate_sources_at_conformal_radius_brain(2, rho_target, seed=seed)
    
    if domain == 'disk':
        rho_actual = compute_conformal_radius_disk(pilot_sources)
    elif domain == 'ellipse':
        rho_actual = compute_conformal_radius_ellipse(pilot_sources)
    else:
        rho_actual = compute_conformal_radius_brain(pilot_sources)
    
    rho_min_pilot = np.min(rho_actual)
    
    # Compute sigma_four - EXACTLY as theory (no signal dependence!)
    sigma_four = compute_sigma_four(sigma_noise, n_sensors)
    N_max_est = compute_N_max(sigma_four, rho_min_pilot)
    
    print(f"\nParameters:")
    print(f"  Target conformal radius: ρ_target = {rho_target:.3f}")
    print(f"  Actual conformal radii: ρ_actual = {rho_actual}")
    print(f"  Minimum conformal radius: ρ_min = {rho_min_pilot:.4f}")
    print(f"  Absolute noise: σ_noise = {sigma_noise}")
    print(f"  Number of sensors: M = {n_sensors}")
    print(f"  σ_Four = σ_noise/√M = {sigma_four:.6f}")
    print(f"  N_max = (2/3) × log(σ_Four)/log(ρ_min) = {N_max_est:.2f}")
    
    # Test values around the bound (EVEN N ONLY to ensure clean +1/-1 intensities)
    N_floor = int(np.floor(N_max_est))
    # Make N_floor even
    if N_floor % 2 == 1:
        N_floor_even = N_floor - 1
    else:
        N_floor_even = N_floor
    
    # Test even values: N_floor-2, N_floor, N_floor+2, N_floor+4
    test_N_values = [N for N in [N_floor_even - 2, N_floor_even, N_floor_even + 2, N_floor_even + 4] 
                     if N >= 2]
    
    print(f"\nTesting N ∈ {test_N_values} (even only for clean ±1 intensities)")
    print(f"Theory: N ≤ {N_max_est:.2f} should succeed, N > {N_max_est:.2f} should fail")
    print(f"\n{'-'*70}")
    
    results = []
    for N in test_N_values:
        print(f"\nTesting N = {N}...")
        result = run_test(domain, N, rho_target, sigma_noise, n_sensors,
                          n_restarts=n_restarts, seed=seed, rmse_threshold=rmse_threshold)
        results.append(result)
        
        status = "✓ SUCCESS" if result.actual_success else "✗ FAILED"
        match = "as expected" if result.expected_success == result.actual_success else "UNEXPECTED!"
        
        print(f"  N = {N}: RMSE = {result.position_rmse:.6f}, σ_Four = {result.sigma_four:.6f}")
        print(f"          N_max(theory) = {result.N_max_theory:.2f}")
        print(f"          {status} ({match})")
        print(f"          Time: {result.time_seconds:.1f}s")
        
        # Save individual plot
        if save_plots:
            plot_path = os.path.join(output_dir, f"bound_test_{domain}_N{N}.png")
            plot_single_test(
                domain=domain,
                sources=result.true_sources,
                recovered=result.recovered_sources,
                rho_target=rho_target,
                N_max=result.N_max_theory,
                rmse=result.position_rmse,
                sensor_locations=result.sensor_locations,
                save_path=plot_path
            )
    
    # Save summary plot
    if save_plots:
        summary_path = os.path.join(output_dir, f"bound_test_{domain}_summary.png")
        plot_summary(results, domain, save_path=summary_path)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'N':>4} | {'ρ_min':>8} | {'σ_Four':>10} | {'N_max':>6} | {'RMSE':>10} | {'Result':>12} | {'Match'}")
    print("-" * 70)
    
    all_match = True
    for r in results:
        status = "SUCCESS" if r.actual_success else "FAILED"
        match = "✓" if r.expected_success == r.actual_success else "✗ UNEXPECTED"
        if r.expected_success != r.actual_success:
            all_match = False
        print(f"{r.N_tested:>4} | {np.min(r.rho_actual):>8.4f} | {r.sigma_four:>10.6f} | "
              f"{r.N_max_theory:>6.2f} | {r.position_rmse:>10.6f} | {status:>12} | {match}")
    
    print(f"\n{'='*70}")
    if all_match:
        print("✓ THEORY VALIDATED: All results match theoretical predictions!")
    else:
        print("⚠ DISCREPANCY: Some results don't match theoretical predictions.")
        print("  This could indicate:")
        print("  - The bound is not sharp (theory gives necessary but not sufficient condition)")
        print("  - Numerical issues (optimizer stuck in local minimum)")
        print("  - Sources too close together (violates well-separated assumption)")
    print(f"{'='*70}\n")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test theoretical source number bound")
    parser.add_argument('--domain', type=str, default='disk',
                       choices=['disk', 'ellipse', 'brain', 'all'],
                       help='Domain to test')
    parser.add_argument('--rho', type=float, default=0.7,
                       help='Target conformal radius (default: 0.7)')
    parser.add_argument('--sigma-noise', type=float, default=0.001,
                       help='Absolute noise std deviation (default: 0.001)')
    parser.add_argument('--sensors', type=int, default=100,
                       help='Number of sensors (default: 100)')
    parser.add_argument('--restarts', type=int, default=15,
                       help='Number of optimization restarts (default: 15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--threshold', type=float, default=0.05,
                       help='RMSE success threshold (default: 0.05)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plot generation')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Directory to save plots (default: current directory)')
    
    args = parser.parse_args()
    
    # Compute sigma_four for display
    sigma_four_display = args.sigma_noise / np.sqrt(args.sensors)
    
    print("="*70)
    print("THEORETICAL BOUND VALIDATION")
    print("="*70)
    print(f"\nBound: N ≤ (2/3) × log(σ_Four) / log(ρ_min)")
    print(f"where: σ_Four = σ_noise / √M")
    print(f"\nTest parameters:")
    print(f"  Target conformal radius: ρ = {args.rho}")
    print(f"  Absolute noise: σ_noise = {args.sigma_noise}")
    print(f"  Sensors: M = {args.sensors}")
    print(f"  σ_Four = {args.sigma_noise}/√{args.sensors} = {sigma_four_display:.6f}")
    print(f"  Success threshold: RMSE < {args.threshold}")
    
    if args.domain == 'all':
        domains = ['disk', 'ellipse', 'brain']
    else:
        domains = [args.domain]
    
    all_results = {}
    for domain in domains:
        results = run_bound_validation(
            domain=domain,
            rho_target=args.rho,
            sigma_noise=args.sigma_noise,
            n_sensors=args.sensors,
            n_restarts=args.restarts,
            seed=args.seed,
            rmse_threshold=args.threshold,
            save_plots=not args.no_plots,
            output_dir=args.output_dir
        )
        all_results[domain] = results
    
    # Final comparison across domains
    if len(domains) > 1:
        print("\n" + "="*70)
        print("CROSS-DOMAIN COMPARISON")
        print("="*70)
        
        for domain, results in all_results.items():
            print(f"\n{domain.upper()}:")
            for r in results:
                status = "✓" if r.actual_success else "✗"
                print(f"  N={r.N_tested}: ρ_min={np.min(r.rho_actual):.4f}, "
                      f"N_max={r.N_max_theory:.2f}, RMSE={r.position_rmse:.6f} {status}")
