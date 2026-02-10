#!/usr/bin/env python3
"""
Multi-seed validation of depth-weighted linear methods.

Runs multiple seeds and produces:
1. Figure D: Box plots comparing original vs weighted across seeds
2. Summary table (CSV) with statistics

Usage:
    python run_multiseed_validation.py --domain disk --n-seeds 20 --output-dir ./figs_boundary_bias
    
For cluster (SLURM), see generate_slurm_jobs.py
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
import sys
from scipy import stats

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analytical_solver import greens_function_disk_neumann
try:
    from conformal_solver import MFSConformalMap
    HAS_CONFORMAL = True
except ImportError:
    HAS_CONFORMAL = False

from depth_weighted_solvers import (
    compute_depth_weights, solve_l2_weighted, solve_l1_weighted_admm,
    solve_tv_weighted, compute_l_curve_weighted
)


def get_domain_boundary(domain_type):
    """Get boundary function for domain type."""
    if domain_type == 'disk':
        return lambda t: np.exp(1j * t)
    elif domain_type == 'ellipse':
        a, b = 1.5, 0.8
        return lambda t: a * np.cos(t) + 1j * b * np.sin(t)
    elif domain_type == 'brain':
        def brain_boundary(t):
            r = 0.8 + 0.15 * np.cos(2*t) + 0.1 * np.cos(3*t) - 0.05 * np.cos(5*t)
            return r * np.exp(1j * t)
        return brain_boundary
    else:
        raise ValueError(f"Unknown domain: {domain_type}")


def run_single_seed(domain_type, seed, n_sources=4, rho_min=0.5, rho_max=0.7,
                    n_sensors=100, sigma_noise=0.001, n_radial=15, n_angular=24):
    """Run all methods for a single seed, return summary stats."""
    
    rng = np.random.RandomState(seed)
    
    # Setup conformal map
    cmap = None
    if domain_type != 'disk' and HAS_CONFORMAL:
        boundary_func = get_domain_boundary(domain_type)
        cmap = MFSConformalMap(boundary_func, n_boundary=256, n_charge=200)
    
    # Generate grid
    grid_points = []
    conformal_radii = []
    radii = np.linspace(0.1, 0.95, n_radial)
    angles = np.linspace(0, 2*np.pi, n_angular, endpoint=False)
    
    for r in radii:
        for theta in angles:
            if domain_type == 'disk':
                z = r * np.exp(1j * theta)
                grid_points.append(z)
                conformal_radii.append(r)
            elif cmap is not None:
                w = r * np.exp(1j * theta)
                z = cmap.from_disk(w)
                grid_points.append(z)
                conformal_radii.append(r)
    
    grid_points = np.array(grid_points)
    conformal_radii = np.array(conformal_radii)
    Mg = len(grid_points)
    
    # Generate sensors
    theta_sens = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
    if domain_type == 'disk':
        sensors = np.exp(1j * theta_sens)
    elif cmap is not None:
        w_boundary = np.exp(1j * theta_sens)
        sensors = np.array([cmap.from_disk(w) for w in w_boundary])
    else:
        sensors = np.exp(1j * theta_sens)  # Fallback
    
    # Generate true sources
    true_rho = rng.uniform(rho_min, rho_max, n_sources)
    true_phi = np.linspace(0, 2*np.pi, n_sources, endpoint=False) + rng.uniform(0, 0.3, n_sources)
    true_intensities = rng.uniform(0.5, 2.0, n_sources) * np.array([(-1)**k for k in range(n_sources)])
    true_intensities = true_intensities - np.mean(true_intensities)
    
    if domain_type == 'disk':
        true_positions = true_rho * np.exp(1j * true_phi)
    elif cmap is not None:
        w_true = true_rho * np.exp(1j * true_phi)
        true_positions = np.array([cmap.from_disk(w) for w in w_true])
    else:
        true_positions = true_rho * np.exp(1j * true_phi)
    
    # Build Green's matrix
    G = np.zeros((n_sensors, Mg))
    for j, xi in enumerate(grid_points):
        for i, x_sens in enumerate(sensors):
            if domain_type == 'disk':
                G[i, j] = greens_function_disk_neumann(
                    np.array([x_sens.real, x_sens.imag]),
                    np.array([xi.real, xi.imag])
                )
            elif cmap is not None:
                w_source = cmap.to_disk(xi)
                w_sens = cmap.to_disk(x_sens)
                G[i, j] = greens_function_disk_neumann(
                    np.array([w_sens.real, w_sens.imag]),
                    np.array([w_source.real, w_source.imag])
                )
    
    # Compute forward data
    u = np.zeros(n_sensors)
    for pos, I in zip(true_positions, true_intensities):
        for i, x_sens in enumerate(sensors):
            if domain_type == 'disk':
                u[i] += I * greens_function_disk_neumann(
                    np.array([x_sens.real, x_sens.imag]),
                    np.array([pos.real, pos.imag])
                )
            elif cmap is not None:
                w_source = cmap.to_disk(pos)
                w_sens = cmap.to_disk(x_sens)
                u[i] += I * greens_function_disk_neumann(
                    np.array([w_sens.real, w_sens.imag]),
                    np.array([w_source.real, w_source.imag])
                )
    
    u = u - np.mean(u)
    noise = rng.randn(n_sensors) * sigma_noise
    u_data = u + noise
    
    # Compatibility constraint
    A_eq = np.ones((1, Mg))
    b_eq = np.array([0.0])
    
    # Target zone mask
    target_mask = (conformal_radii >= rho_min - 0.05) & (conformal_radii <= rho_max + 0.05)
    
    # Run methods
    results = {}
    
    for method_name, reg_type in [('L2', 'l2'), ('L1', 'l1'), ('TV', 'tv')]:
        for variant in ['original', 'weighted']:
            key = f'{method_name}_{variant}'
            
            try:
                if variant == 'original':
                    weights = None
                    noise_level = None  # L-curve
                else:
                    weights = compute_depth_weights(conformal_radii, beta=1.0)
                    noise_level = sigma_noise  # Discrepancy principle
                
                alpha, _, _ = compute_l_curve_weighted(
                    G, u_data, reg_type=reg_type,
                    weights=weights, noise_level=noise_level,
                    n_alphas=50
                )
                
                if reg_type == 'l2':
                    q = solve_l2_weighted(G, u_data, alpha, A_eq, b_eq, weights=weights)
                elif reg_type == 'l1':
                    q = solve_l1_weighted_admm(G, u_data, alpha, A_eq, b_eq, weights=weights)
                else:
                    q = solve_tv_weighted(G, u_data, alpha, A_eq, b_eq, weights=weights)
                
                # Compute target percentage
                total = np.sum(np.abs(q))
                target = np.sum(np.abs(q[target_mask]))
                target_pct = 100 * target / total if total > 0 else 0
                
                results[key] = {
                    'alpha': alpha,
                    'target_pct': target_pct,
                    'total_intensity': total
                }
            except Exception as e:
                results[key] = {
                    'alpha': 0,
                    'target_pct': 0,
                    'total_intensity': 0,
                    'error': str(e)
                }
    
    return results


def figure_D_boxplots(all_results, output_path):
    """Create box plots comparing original vs weighted across seeds."""
    
    methods = ['L2', 'L1', 'TV']
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        
        orig_vals = [r[f'{method}_original']['target_pct'] for r in all_results]
        weight_vals = [r[f'{method}_weighted']['target_pct'] for r in all_results]
        
        positions = [0, 1]
        bp = ax.boxplot([orig_vals, weight_vals], positions=positions, widths=0.6,
                        patch_artist=True)
        
        # Colors
        bp['boxes'][0].set_facecolor('coral')
        bp['boxes'][1].set_facecolor('steelblue')
        
        # Statistics
        orig_med = np.median(orig_vals)
        weight_med = np.median(weight_vals)
        
        # Mann-Whitney test
        stat, p_val = stats.mannwhitneyu(weight_vals, orig_vals, alternative='greater')
        
        ax.set_xticks(positions)
        ax.set_xticklabels(['Original', 'Weighted'])
        ax.set_ylabel('Target Zone Intensity (%)')
        ax.set_title(f'{method}\nMedian: {orig_med:.1f}% → {weight_med:.1f}%\np={p_val:.2e}')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Horizontal line at true source zone
        ax.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='Random baseline')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Figure D saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Multi-seed validation')
    parser.add_argument('--domain', type=str, default='disk', 
                        choices=['disk', 'ellipse', 'brain'])
    parser.add_argument('--n-seeds', type=int, default=20)
    parser.add_argument('--start-seed', type=int, default=0)
    parser.add_argument('--n-sources', type=int, default=4)
    parser.add_argument('--rho-min', type=float, default=0.5)
    parser.add_argument('--rho-max', type=float, default=0.7)
    parser.add_argument('--n-sensors', type=int, default=100)
    parser.add_argument('--sigma-noise', type=float, default=0.001)
    parser.add_argument('--output-dir', type=str, default='./figs_boundary_bias')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"=" * 60)
    print(f"Multi-Seed Validation: {args.n_seeds} seeds")
    print(f"Domain: {args.domain}")
    print(f"=" * 60)
    
    all_results = []
    
    for i, seed in enumerate(range(args.start_seed, args.start_seed + args.n_seeds)):
        print(f"\nSeed {seed} ({i+1}/{args.n_seeds})...")
        
        result = run_single_seed(
            args.domain, seed, args.n_sources, 
            args.rho_min, args.rho_max,
            args.n_sensors, args.sigma_noise
        )
        all_results.append(result)
        
        # Print progress
        l1_orig = result['L1_original']['target_pct']
        l1_weight = result['L1_weighted']['target_pct']
        print(f"  L1: {l1_orig:.1f}% → {l1_weight:.1f}%")
    
    # Generate Figure D
    print("\n--- Figure D: Box Plots ---")
    fig_d_path = os.path.join(args.output_dir, f'fig_D_boxplots_{args.domain}.png')
    figure_D_boxplots(all_results, fig_d_path)
    
    # Summary statistics
    print("\n--- Summary Statistics ---")
    print("\nMethod              | Original (med ± IQR) | Weighted (med ± IQR) | Improvement")
    print("-" * 80)
    
    summary_data = []
    for method in ['L2', 'L1', 'TV']:
        orig_vals = [r[f'{method}_original']['target_pct'] for r in all_results]
        weight_vals = [r[f'{method}_weighted']['target_pct'] for r in all_results]
        
        orig_med = np.median(orig_vals)
        orig_iqr = np.percentile(orig_vals, 75) - np.percentile(orig_vals, 25)
        weight_med = np.median(weight_vals)
        weight_iqr = np.percentile(weight_vals, 75) - np.percentile(weight_vals, 25)
        
        improvement = weight_med - orig_med
        
        print(f"{method:20s} | {orig_med:5.1f}% ± {orig_iqr:4.1f}%      | {weight_med:5.1f}% ± {weight_iqr:4.1f}%      | +{improvement:.1f}%")
        
        summary_data.append({
            'method': method,
            'original_median': orig_med,
            'original_iqr': orig_iqr,
            'weighted_median': weight_med,
            'weighted_iqr': weight_iqr,
            'improvement': improvement
        })
    
    # Save summary
    summary_path = os.path.join(args.output_dir, f'summary_multiseed_{args.domain}.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'domain': args.domain,
            'n_seeds': args.n_seeds,
            'n_sources': args.n_sources,
            'rho_range': [args.rho_min, args.rho_max],
            'summary': summary_data,
            'all_results': all_results
        }, f, indent=2)
    print(f"\nSummary saved: {summary_path}")
    
    print(f"\n{'=' * 60}")
    print("DONE!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
