#!/usr/bin/env python3
"""
Test Script: Validating Theoretical Bound via Variance Analysis
================================================================

This script tests the theoretical bound:

    N ≤ (2/3) × log(σ_Four) / log(ρ_min)

by examining the VARIANCE of recovery error across multiple random seeds.

KEY INSIGHT:
- Below N_max: Problem is well-posed → consistent results across seeds
- Above N_max: Problem is ill-posed → high variance across seeds
                (multiple solutions fit the data equally well)

This is a more rigorous test than looking for a "jump" in error, because:
1. The transition is gradual, not discontinuous
2. Mean error naturally grows with N regardless of the bound
3. Variance captures the loss of uniqueness that the theory predicts

Usage:
    python test_bound_variance.py --domain disk --rho 0.7 --n-seeds 5
    python test_bound_variance.py --domain all --rho 0.7 --sigma-noise 0.0001

Author: Shaerdan Shataer
Date: February 2026
"""

import numpy as np
import sys
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
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
    σ_Four = σ_noise / √M
    """
    return sigma_noise / np.sqrt(n_sensors)


def compute_N_max(sigma_four: float, rho_min: float) -> float:
    """
    Compute the theoretical maximum number of recoverable sources.
    N_max = (2/3) × log(σ_Four) / log(ρ_min)
    """
    if sigma_four >= 1 or sigma_four <= 0:
        raise ValueError(f"sigma_four must be in (0, 1), got {sigma_four}")
    if rho_min >= 1 or rho_min <= 0:
        raise ValueError(f"rho_min must be in (0, 1), got {rho_min}")
    
    return (2/3) * np.log(sigma_four) / np.log(rho_min)


# =============================================================================
# CONFORMAL RADIUS COMPUTATION
# =============================================================================

def compute_conformal_radius_disk(sources: List) -> np.ndarray:
    """For the disk, conformal radius = Euclidean radius."""
    radii = [np.sqrt(x**2 + y**2) for (x, y), _ in sources]
    return np.array(radii)


def compute_conformal_radius_ellipse(sources: List, a: float = 1.5, b: float = 0.8) -> np.ndarray:
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


def compute_conformal_radius_brain(sources: List) -> np.ndarray:
    """Compute conformal radii for brain using MFS conformal map."""
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
    
    radii = []
    for (x, y), _ in sources:
        z = x + 1j * y
        w = cmap.to_disk(z)
        radii.append(np.abs(w))
    return np.array(radii)


# =============================================================================
# SOURCE GENERATION
# =============================================================================

def generate_zero_sum_intensities(n_sources: int) -> np.ndarray:
    """Generate n_sources intensities that sum to zero (alternating +1, -1)."""
    if n_sources % 2 != 0:
        raise ValueError(f"n_sources must be even, got {n_sources}")
    return np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n_sources)])


def generate_sources_disk(n_sources: int, rho_target: float, seed: int) -> List:
    """Generate sources in disk at specified conformal radius."""
    np.random.seed(seed)
    
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.1, 0.1, n_sources)
    
    intensities = generate_zero_sum_intensities(n_sources)
    
    sources = []
    for i, theta in enumerate(angles):
        x = rho_target * np.cos(theta)
        y = rho_target * np.sin(theta)
        sources.append(((x, y), intensities[i]))
    
    return sources


def generate_sources_ellipse(n_sources: int, rho_target: float, seed: int,
                              a: float = 1.5, b: float = 0.8) -> List:
    """Generate sources in ellipse at specified conformal radius."""
    np.random.seed(seed)
    
    def ellipse_boundary(t):
        return a * np.cos(t) + 1j * b * np.sin(t)
    
    cmap = MFSConformalMap(ellipse_boundary, n_boundary=256, n_charge=200)
    
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.1, 0.1, n_sources)
    
    intensities = generate_zero_sum_intensities(n_sources)
    
    sources = []
    for i, theta in enumerate(angles):
        w_target = rho_target * np.exp(1j * theta)
        z = cmap.from_disk(w_target)
        x, y = z.real, z.imag
        sources.append(((x, y), intensities[i]))
    
    return sources


def generate_sources_brain(n_sources: int, rho_target: float, seed: int) -> List:
    """Generate sources in brain at specified conformal radius."""
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
    
    intensities = generate_zero_sum_intensities(n_sources)
    
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

def solve_inverse_disk(u_measured: np.ndarray, n_sensors: int,
                        n_sources: int, n_restarts: int, seed: int) -> Tuple[List, float]:
    """Solve inverse problem for disk."""
    solver = AnalyticalNonlinearInverseSolver(n_sources=n_sources, n_boundary=n_sensors)
    solver.set_measured_data(u_measured)
    result = solver.solve(method='SLSQP', n_restarts=n_restarts)
    return result.sources, result.residual


def solve_inverse_conformal(u_measured: np.ndarray, cmap,
                             sensor_locations: np.ndarray,
                             n_sources: int, n_restarts: int, seed: int) -> Tuple[List, float]:
    """Solve inverse problem using conformal mapping."""
    solver = ConformalNonlinearInverseSolver(
        cmap, n_sources=n_sources, n_boundary=len(sensor_locations),
        sensor_locations=sensor_locations
    )
    sources, residual = solver.solve(u_measured, method='SLSQP',
                                      n_restarts=n_restarts, seed=seed)
    return sources, residual


# =============================================================================
# POSITION ERROR
# =============================================================================

def compute_position_rmse(true_sources: List, recovered_sources: List) -> float:
    """Compute RMSE using optimal matching (Hungarian algorithm)."""
    from scipy.optimize import linear_sum_assignment
    
    n = len(true_sources)
    if len(recovered_sources) != n:
        return np.inf
    
    true_pos = [s[0] if isinstance(s, tuple) else (s.x, s.y) for s in true_sources]
    rec_pos = [s[0] if isinstance(s, tuple) else (s.x, s.y) for s in recovered_sources]
    
    cost = np.zeros((n, n))
    for i, (x1, y1) in enumerate(true_pos):
        for j, (x2, y2) in enumerate(rec_pos):
            cost[i, j] = (x1 - x2)**2 + (y1 - y2)**2
    
    row_ind, col_ind = linear_sum_assignment(cost)
    total_sq_error = sum(cost[i, j] for i, j in zip(row_ind, col_ind))
    
    return np.sqrt(total_sq_error / n)


# =============================================================================
# SINGLE RUN
# =============================================================================

def run_single(args_tuple) -> Tuple[float, float]:
    """
    Run a single test and return (rmse, residual).
    
    Takes a tuple for compatibility with multiprocessing.map()
    
    Parameters (via tuple)
    ----------------------
    domain, n_sources, rho_target, sigma_noise, n_sensors, n_restarts, seed
    
    Returns
    -------
    rmse : float
        Position RMSE
    residual : float
        Data fit residual
    """
    domain, n_sources, rho_target, sigma_noise, n_sensors, n_restarts, seed = args_tuple
    
    # Generate sources
    if domain == 'disk':
        sources = generate_sources_disk(n_sources, rho_target, seed)
    elif domain == 'ellipse':
        sources = generate_sources_ellipse(n_sources, rho_target, seed)
    elif domain == 'brain':
        sources = generate_sources_brain(n_sources, rho_target, seed)
    
    # Forward solve
    if domain == 'disk':
        theta = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
        fwd = AnalyticalForwardSolver(n_boundary_points=n_sensors)
        u_true = fwd.solve(sources)
        sensor_locations = np.column_stack([np.cos(theta), np.sin(theta)])
        cmap = None
    else:
        if domain == 'ellipse':
            cmap = create_conformal_map('ellipse', a=1.5, b=0.8)
        else:
            cmap = create_conformal_map('brain')
        fwd = ConformalForwardSolver(cmap, n_sensors)
        u_true = fwd.solve(sources)
        sensor_locations = fwd.boundary_points
    
    # Add noise
    np.random.seed(seed + 10000)  # Different seed for noise
    noise = sigma_noise * np.random.randn(n_sensors)
    u_measured = u_true + noise
    
    # Solve inverse
    if domain == 'disk':
        recovered, residual = solve_inverse_disk(u_measured, n_sensors, n_sources,
                                                   n_restarts, seed + 20000)
    else:
        recovered, residual = solve_inverse_conformal(u_measured, cmap, sensor_locations,
                                                        n_sources, n_restarts, seed + 20000)
    
    rmse = compute_position_rmse(sources, recovered)
    return rmse, residual


# =============================================================================
# VARIANCE ANALYSIS
# =============================================================================

@dataclass
class VarianceResult:
    """Results for one N value across multiple seeds."""
    n_sources: int
    n_seeds: int
    rmse_values: np.ndarray
    residual_values: np.ndarray
    
    @property
    def rmse_mean(self) -> float:
        return np.mean(self.rmse_values)
    
    @property
    def rmse_std(self) -> float:
        return np.std(self.rmse_values)
    
    @property
    def rmse_min(self) -> float:
        return np.min(self.rmse_values)
    
    @property
    def rmse_max(self) -> float:
        return np.max(self.rmse_values)
    
    @property
    def residual_mean(self) -> float:
        return np.mean(self.residual_values)
    
    @property
    def success_rate(self) -> float:
        """Fraction of runs with RMSE < 0.05"""
        return np.mean(self.rmse_values < 0.05)


def run_variance_analysis(domain: str, n_sources: int, rho_target: float,
                           sigma_noise: float, n_sensors: int, n_restarts: int,
                           n_seeds: int, base_seed: int = 42,
                           n_jobs: int = -1) -> VarianceResult:
    """
    Run multiple seeds for one N value and collect statistics.
    
    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs. -1 means use all CPUs.
    """
    from multiprocessing import Pool, cpu_count
    
    if n_jobs == -1:
        n_jobs = cpu_count()
    
    # Prepare arguments for each seed
    seeds = [base_seed + i * 1000 for i in range(n_seeds)]
    args_list = [(domain, n_sources, rho_target, sigma_noise, n_sensors, n_restarts, seed) 
                 for seed in seeds]
    
    # Run in parallel
    if n_jobs > 1 and n_seeds > 1:
        with Pool(processes=min(n_jobs, n_seeds)) as pool:
            results = pool.map(run_single, args_list)
        rmse_values = [r[0] for r in results]
        residual_values = [r[1] for r in results]
    else:
        # Sequential fallback
        rmse_values = []
        residual_values = []
        for args in args_list:
            rmse, residual = run_single(args)
            rmse_values.append(rmse)
            residual_values.append(residual)
    
    return VarianceResult(
        n_sources=n_sources,
        n_seeds=n_seeds,
        rmse_values=np.array(rmse_values),
        residual_values=np.array(residual_values)
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_variance_analysis(results: List[VarianceResult], N_max: float,
                            domain: str, save_path: str = None):
    """
    Plot variance analysis results:
    - Mean RMSE with error bars (std)
    - Individual points for each seed
    - Vertical line at N_max
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    N_values = [r.n_sources for r in results]
    
    # Plot 1: Mean RMSE with std error bars
    ax1 = axes[0, 0]
    means = [r.rmse_mean for r in results]
    stds = [r.rmse_std for r in results]
    ax1.errorbar(N_values, means, yerr=stds, fmt='o-', capsize=5, 
                  markersize=8, linewidth=2, label='Mean ± Std')
    ax1.axvline(x=N_max, color='red', linestyle='--', linewidth=2, label=f'N_max = {N_max:.2f}')
    ax1.axhline(y=0.05, color='green', linestyle=':', linewidth=2, label='Success threshold')
    ax1.set_xlabel('Number of Sources (N)', fontsize=12)
    ax1.set_ylabel('Position RMSE', fontsize=12)
    ax1.set_title('Mean RMSE with Standard Deviation', fontsize=14)
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Std of RMSE (key indicator!)
    ax2 = axes[0, 1]
    ax2.plot(N_values, stds, 'o-', markersize=8, linewidth=2, color='purple')
    ax2.axvline(x=N_max, color='red', linestyle='--', linewidth=2, label=f'N_max = {N_max:.2f}')
    ax2.set_xlabel('Number of Sources (N)', fontsize=12)
    ax2.set_ylabel('Std of RMSE across seeds', fontsize=12)
    ax2.set_title('RMSE Variance (Key Indicator of Ill-Posedness)', fontsize=14)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: All individual RMSE values
    ax3 = axes[1, 0]
    for r in results:
        # Jitter x slightly for visibility
        x_jitter = r.n_sources + np.random.uniform(-0.15, 0.15, len(r.rmse_values))
        ax3.scatter(x_jitter, r.rmse_values, alpha=0.5, s=30)
    ax3.axvline(x=N_max, color='red', linestyle='--', linewidth=2, label=f'N_max = {N_max:.2f}')
    ax3.axhline(y=0.05, color='green', linestyle=':', linewidth=2, label='Success threshold')
    ax3.set_xlabel('Number of Sources (N)', fontsize=12)
    ax3.set_ylabel('Position RMSE', fontsize=12)
    ax3.set_title('Individual RMSE Values (All Seeds)', fontsize=14)
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Success rate
    ax4 = axes[1, 1]
    success_rates = [r.success_rate * 100 for r in results]
    colors = ['green' if r.n_sources <= N_max else 'red' for r in results]
    ax4.bar(N_values, success_rates, color=colors, alpha=0.7, edgecolor='black')
    ax4.axvline(x=N_max, color='red', linestyle='--', linewidth=2, label=f'N_max = {N_max:.2f}')
    ax4.set_xlabel('Number of Sources (N)', fontsize=12)
    ax4.set_ylabel('Success Rate (%)', fontsize=12)
    ax4.set_title('Success Rate (RMSE < 0.05)', fontsize=14)
    ax4.set_ylim(0, 105)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{domain.upper()} Domain: Variance Analysis of Theoretical Bound', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_full_variance_analysis(domain: str, rho_target: float = 0.7,
                                 sigma_noise: float = 0.001, n_sensors: int = 100,
                                 n_restarts: int = 15, n_seeds: int = 5,
                                 base_seed: int = 42, n_jobs: int = -1,
                                 save_plots: bool = True, output_dir: str = '.'):
    """
    Run full variance analysis for a domain.
    
    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs. -1 means use all CPUs.
    """
    import os
    from multiprocessing import cpu_count
    
    if n_jobs == -1:
        n_jobs = cpu_count()
    
    print(f"\n{'='*70}")
    print(f"VARIANCE ANALYSIS: {domain.upper()} DOMAIN")
    print(f"{'='*70}")
    
    # Compute theoretical bound
    sigma_four = compute_sigma_four(sigma_noise, n_sensors)
    
    # Get rho_min from pilot sources
    pilot_sources = generate_sources_disk(2, rho_target, base_seed) if domain == 'disk' \
        else generate_sources_ellipse(2, rho_target, base_seed) if domain == 'ellipse' \
        else generate_sources_brain(2, rho_target, base_seed)
    
    if domain == 'disk':
        rho_actual = compute_conformal_radius_disk(pilot_sources)
    elif domain == 'ellipse':
        rho_actual = compute_conformal_radius_ellipse(pilot_sources)
    else:
        rho_actual = compute_conformal_radius_brain(pilot_sources)
    
    rho_min = np.min(rho_actual)
    N_max = compute_N_max(sigma_four, rho_min)
    
    print(f"\nParameters:")
    print(f"  Target conformal radius: ρ_target = {rho_target:.3f}")
    print(f"  Minimum conformal radius: ρ_min = {rho_min:.4f}")
    print(f"  Absolute noise: σ_noise = {sigma_noise}")
    print(f"  Number of sensors: M = {n_sensors}")
    print(f"  σ_Four = {sigma_four:.6f}")
    print(f"  N_max = {N_max:.2f}")
    print(f"  Seeds per N: {n_seeds}")
    print(f"  Parallel jobs: {n_jobs}")
    
    # Test N values around the bound
    N_floor = int(np.floor(N_max))
    if N_floor % 2 == 1:
        N_floor_even = N_floor - 1
    else:
        N_floor_even = N_floor
    
    test_N_values = [N for N in [N_floor_even - 4, N_floor_even - 2, N_floor_even, 
                                  N_floor_even + 2, N_floor_even + 4, N_floor_even + 6] 
                     if N >= 2]
    
    print(f"\nTesting N ∈ {test_N_values} with {n_seeds} seeds each")
    print(f"Theory: N ≤ {N_max:.2f} should have LOW variance")
    print(f"        N > {N_max:.2f} should have HIGH variance")
    print(f"\n{'-'*70}")
    
    results = []
    for N in test_N_values:
        print(f"\nTesting N = {N} ({n_seeds} seeds, {n_jobs} parallel)...", end=" ", flush=True)
        t_start = time.time()
        
        result = run_variance_analysis(domain, N, rho_target, sigma_noise,
                                        n_sensors, n_restarts, n_seeds, base_seed,
                                        n_jobs=n_jobs)
        results.append(result)
        
        elapsed = time.time() - t_start
        relation = ">" if N > N_max else "≤"
        print(f"Mean={result.rmse_mean:.2e}, Std={result.rmse_std:.2e}, "
              f"Success={result.success_rate*100:.0f}%, N{relation}N_max ({elapsed:.1f}s)")
    
    # Save plot
    if save_plots:
        plot_path = os.path.join(output_dir, f"variance_analysis_{domain}.png")
        plot_variance_analysis(results, N_max, domain, save_path=plot_path)
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'N':>4} | {'Mean':>10} | {'Std':>10} | {'Min':>10} | {'Max':>10} | {'Success':>8} | {'N vs N_max'}")
    print("-" * 80)
    
    for r in results:
        relation = "N > N_max" if r.n_sources > N_max else "N ≤ N_max"
        print(f"{r.n_sources:>4} | {r.rmse_mean:>10.2e} | {r.rmse_std:>10.2e} | "
              f"{r.rmse_min:>10.2e} | {r.rmse_max:>10.2e} | {r.success_rate*100:>7.0f}% | {relation}")
    
    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print("  - LOW std across seeds → well-posed (unique solution)")
    print("  - HIGH std across seeds → ill-posed (multiple solutions fit data)")
    print(f"  - Theory predicts transition at N_max = {N_max:.2f}")
    print(f"{'='*70}\n")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test theoretical bound via variance analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (uses all CPUs by default)
    python test_bound_variance.py --domain disk --rho 0.7 --n-seeds 5
    
    # Sequential execution (for debugging)
    python test_bound_variance.py --domain disk --rho 0.7 --n-jobs 1
    
    # Use 4 parallel workers
    python test_bound_variance.py --domain disk --rho 0.7 --n-jobs 4
    
    # All domains
    python test_bound_variance.py --domain all --rho 0.7 --sigma-noise 0.0001
        """
    )
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
    parser.add_argument('--n-seeds', type=int, default=5,
                        help='Number of random seeds per N (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 = all CPUs, 1 = sequential)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plot generation')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save plots')
    
    args = parser.parse_args()
    
    from multiprocessing import cpu_count
    n_jobs_actual = cpu_count() if args.n_jobs == -1 else args.n_jobs
    
    sigma_four = args.sigma_noise / np.sqrt(args.sensors)
    
    print("="*70)
    print("THEORETICAL BOUND VALIDATION VIA VARIANCE ANALYSIS")
    print("="*70)
    print(f"\nBound: N ≤ (2/3) × log(σ_Four) / log(ρ_min)")
    print(f"\nKey insight:")
    print(f"  Below N_max: Problem is well-posed → LOW variance across seeds")
    print(f"  Above N_max: Problem is ill-posed → HIGH variance across seeds")
    print(f"\nTest parameters:")
    print(f"  Target conformal radius: ρ = {args.rho}")
    print(f"  Absolute noise: σ_noise = {args.sigma_noise}")
    print(f"  Sensors: M = {args.sensors}")
    print(f"  σ_Four = {sigma_four:.6f}")
    print(f"  Seeds per N: {args.n_seeds}")
    print(f"  Parallel jobs: {n_jobs_actual}")
    
    if args.domain == 'all':
        domains = ['disk', 'ellipse', 'brain']
    else:
        domains = [args.domain]
    
    all_results = {}
    for domain in domains:
        results = run_full_variance_analysis(
            domain=domain,
            rho_target=args.rho,
            sigma_noise=args.sigma_noise,
            n_sensors=args.sensors,
            n_restarts=args.restarts,
            n_seeds=args.n_seeds,
            base_seed=args.seed,
            n_jobs=args.n_jobs,
            save_plots=not args.no_plots,
            output_dir=args.output_dir
        )
        all_results[domain] = results
