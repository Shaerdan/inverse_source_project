#!/usr/bin/env python3
"""
Generate all publication figures for the linear methods boundary bias section.

This script produces:
1. Figure A: Column norm sensitivity curve (||G_j|| vs rho_j)
2. Figure B: 3x2 heatmap grid (L2/L1/TV × Original/Weighted)
3. Figure C: Intensity distribution bar chart by radial zone
4. Figure D: Summary table data (saved as CSV)

Usage:
    python generate_boundary_bias_figures.py --domain disk --seed 42 --output-dir ./figs_boundary_bias

Requirements:
    - numpy, scipy, matplotlib
    - The project's analytical_solver.py and conformal_solver.py modules
    - The depth_weighted_solvers.py module
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import argparse
import json
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analytical_solver import AnalyticalForwardSolver, greens_function_disk_neumann
from conformal_solver import MFSConformalMap, ConformalForwardSolver
from depth_weighted_solvers import (
    compute_conformal_radii_disk, compute_conformal_radii_general,
    compute_depth_weights, solve_l2_weighted, solve_l1_weighted_admm,
    solve_tv_weighted, compute_l_curve_weighted, compute_intensity_distribution
)

# Try importing mesh module for domain boundaries
try:
    from mesh import get_boundary_function
    HAS_MESH = True
except ImportError:
    HAS_MESH = False


def get_domain_boundary(domain_type):
    """Get boundary function for domain type."""
    if domain_type == 'disk':
        return lambda t: np.exp(1j * t)
    elif domain_type == 'ellipse':
        a, b = 1.5, 0.8
        return lambda t: a * np.cos(t) + 1j * b * np.sin(t)
    elif domain_type == 'brain':
        if HAS_MESH:
            return get_boundary_function('brain')
        else:
            # Fallback: use a simple brain-like shape
            def brain_boundary(t):
                r = 0.8 + 0.15 * np.cos(2*t) + 0.1 * np.cos(3*t) - 0.05 * np.cos(5*t)
                return r * np.exp(1j * t)
            return brain_boundary
    else:
        raise ValueError(f"Unknown domain: {domain_type}")


def build_green_matrix(domain_type, grid_points, sensor_locations, cmap=None):
    """Build the Green's function matrix G."""
    M = len(sensor_locations)
    Mg = len(grid_points)
    G = np.zeros((M, Mg))
    
    if domain_type == 'disk':
        for j, xi in enumerate(grid_points):
            for i, x_sens in enumerate(sensor_locations):
                # Use analytical Green's function for disk
                G[i, j] = greens_function_disk_neumann(
                    np.array([x_sens.real, x_sens.imag]),
                    np.array([xi.real, xi.imag])
                )
    else:
        # For general domains, use conformal mapping
        if cmap is None:
            raise ValueError("Need conformal map for non-disk domains")
        for j, xi in enumerate(grid_points):
            w_source = cmap.to_disk(xi)
            for i, x_sens in enumerate(sensor_locations):
                w_sens = cmap.to_disk(x_sens)
                # Green's function in disk coordinates
                G[i, j] = greens_function_disk_neumann(
                    np.array([w_sens.real, w_sens.imag]),
                    np.array([w_source.real, w_source.imag])
                )
    
    return G


def generate_grid_points(domain_type, n_radial=15, n_angular=24, cmap=None):
    """Generate interior grid points for linear methods."""
    grid_points = []
    conformal_radii = []
    
    radii = np.linspace(0.1, 0.95, n_radial)
    angles = np.linspace(0, 2*np.pi, n_angular, endpoint=False)
    
    if domain_type == 'disk':
        for r in radii:
            for theta in angles:
                z = r * np.exp(1j * theta)
                grid_points.append(z)
                conformal_radii.append(r)
    else:
        # For general domains, generate in disk then map to physical
        if cmap is None:
            raise ValueError("Need conformal map for non-disk domains")
        for r in radii:
            for theta in angles:
                w = r * np.exp(1j * theta)  # disk coordinates
                z = cmap.from_disk(w)        # physical coordinates
                grid_points.append(z)
                conformal_radii.append(r)
    
    return np.array(grid_points), np.array(conformal_radii)


def generate_sensors(domain_type, n_sensors=100, cmap=None):
    """Generate sensor locations on boundary."""
    theta = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
    
    if domain_type == 'disk':
        return np.exp(1j * theta)
    else:
        # Map from disk boundary to physical boundary
        w_boundary = np.exp(1j * theta)
        return np.array([cmap.from_disk(w) for w in w_boundary])


def generate_true_sources(domain_type, n_sources=4, rho_min=0.5, rho_max=0.7, 
                          seed=42, cmap=None):
    """Generate true point sources in target zone."""
    rng = np.random.RandomState(seed)
    
    # Generate in conformal coordinates
    rho = rng.uniform(rho_min, rho_max, n_sources)
    phi = np.linspace(0, 2*np.pi, n_sources, endpoint=False) + rng.uniform(0, 0.3, n_sources)
    
    # Intensities: alternating signs, random magnitudes
    intensities = rng.uniform(0.5, 2.0, n_sources) * np.array([(-1)**k for k in range(n_sources)])
    intensities = intensities - np.mean(intensities)  # zero-sum
    
    # Convert to physical coordinates
    if domain_type == 'disk':
        positions = rho * np.exp(1j * phi)
    else:
        w_disk = rho * np.exp(1j * phi)
        positions = np.array([cmap.from_disk(w) for w in w_disk])
    
    return positions, intensities, rho


def compute_forward_data(positions, intensities, sensor_locations, domain_type, 
                         sigma_noise=0.001, seed=42, cmap=None):
    """Compute noisy boundary measurements."""
    rng = np.random.RandomState(seed + 1000)
    
    M = len(sensor_locations)
    u = np.zeros(M)
    
    if domain_type == 'disk':
        for pos, I in zip(positions, intensities):
            for i, x_sens in enumerate(sensor_locations):
                u[i] += I * greens_function_disk_neumann(
                    np.array([x_sens.real, x_sens.imag]),
                    np.array([pos.real, pos.imag])
                )
    else:
        for pos, I in zip(positions, intensities):
            w_source = cmap.to_disk(pos)
            for i, x_sens in enumerate(sensor_locations):
                w_sens = cmap.to_disk(x_sens)
                u[i] += I * greens_function_disk_neumann(
                    np.array([w_sens.real, w_sens.imag]),
                    np.array([w_source.real, w_source.imag])
                )
    
    # Zero-mean and add noise
    u = u - np.mean(u)
    noise = rng.randn(M) * sigma_noise
    u_noisy = u + noise
    
    return u_noisy, u


# =============================================================================
# FIGURE A: Column Norm Sensitivity
# =============================================================================

def figure_A_column_norm_sensitivity(G, conformal_radii, output_path):
    """
    Plot ||G_j|| vs conformal radius rho_j.
    Shows the ~20x sensitivity variation with depth.
    """
    column_norms = np.linalg.norm(G, axis=0)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.scatter(conformal_radii, column_norms, alpha=0.5, s=20, c='steelblue')
    
    # Fit and plot trend
    rho_sorted = np.sort(np.unique(conformal_radii))
    norm_means = [np.mean(column_norms[conformal_radii == r]) for r in rho_sorted]
    ax.plot(rho_sorted, norm_means, 'r-', linewidth=2, label='Mean trend')
    
    # Compute sensitivity ratio
    inner_norm = np.mean(column_norms[conformal_radii < 0.3])
    outer_norm = np.mean(column_norms[conformal_radii > 0.85])
    ratio = outer_norm / inner_norm
    
    ax.set_xlabel(r'Conformal radius $\rho$', fontsize=12)
    ax.set_ylabel(r'Column norm $\|G_j\|$', fontsize=12)
    ax.set_title(f'Green\'s Matrix Column Sensitivity\n(boundary/center ratio: {ratio:.1f}×)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add annotation
    ax.annotate(f'{ratio:.1f}× sensitivity\nvariation', 
                xy=(0.7, outer_norm * 0.8), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Figure A saved: {output_path}")
    print(f"  Sensitivity ratio (boundary/center): {ratio:.1f}×")
    
    return ratio


# =============================================================================
# FIGURE B: Heatmap Grid (3×2)
# =============================================================================

def figure_B_heatmap_grid(results_dict, grid_points, true_positions, 
                          domain_type, output_path, cmap_domain=None):
    """
    Create 3×2 heatmap grid: rows = L2/L1/TV, columns = Original/Weighted.
    """
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    
    methods = ['L2', 'L1', 'TV']
    variants = ['original', 'weighted']
    
    # Get grid coordinates for plotting
    grid_x = np.array([z.real for z in grid_points])
    grid_y = np.array([z.imag for z in grid_points])
    
    # True source positions
    true_x = np.array([z.real for z in true_positions])
    true_y = np.array([z.imag for z in true_positions])
    
    for i, method in enumerate(methods):
        for j, variant in enumerate(variants):
            ax = axes[i, j]
            
            key = f'{method}_{variant}'
            q = results_dict.get(key, np.zeros(len(grid_points)))
            
            # Create triangulation for smooth plotting
            from matplotlib.tri import Triangulation
            triang = Triangulation(grid_x, grid_y)
            
            # Color normalization: symmetric around zero
            vmax = max(np.abs(q).max(), 0.1)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            
            # Plot heatmap
            tcf = ax.tricontourf(triang, q, levels=50, cmap='RdBu_r', norm=norm)
            
            # Plot domain boundary
            t_boundary = np.linspace(0, 2*np.pi, 200)
            if domain_type == 'disk':
                boundary = np.exp(1j * t_boundary)
            elif domain_type == 'ellipse':
                a, b = 1.5, 0.8
                boundary = a * np.cos(t_boundary) + 1j * b * np.sin(t_boundary)
            else:
                boundary_func = get_domain_boundary(domain_type)
                boundary = boundary_func(t_boundary)
            ax.plot(boundary.real, boundary.imag, 'k-', linewidth=1.5)
            
            # Plot true sources
            ax.scatter(true_x, true_y, c='lime', s=100, marker='x', 
                      linewidths=3, zorder=10, label='True sources')
            
            # Title
            title = f'{method} ({variant.capitalize()})'
            ax.set_title(title, fontsize=11)
            ax.set_aspect('equal')
            ax.set_xlim(boundary.real.min() - 0.1, boundary.real.max() + 0.1)
            ax.set_ylim(boundary.imag.min() - 0.1, boundary.imag.max() + 0.1)
            
            # Colorbar
            plt.colorbar(tcf, ax=ax, shrink=0.6, label='Intensity')
    
    # Column headers
    axes[0, 0].annotate('Original (L-curve)', xy=(0.5, 1.15), 
                        xycoords='axes fraction', ha='center', fontsize=12, fontweight='bold')
    axes[0, 1].annotate('Depth-Weighted', xy=(0.5, 1.15), 
                        xycoords='axes fraction', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Figure B saved: {output_path}")


# =============================================================================
# FIGURE C: Intensity Distribution Bar Chart
# =============================================================================

def figure_C_intensity_distribution(results_dict, conformal_radii, true_rho,
                                     output_path):
    """
    Bar chart showing intensity distribution by radial zone.
    Compares original vs weighted for each method.
    """
    zones = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    zone_labels = ['[0, 0.5)', '[0.5, 0.7)', '[0.7, 0.9)', '[0.9, 1.0)']
    
    # Identify target zone (where true sources are)
    target_zone_idx = 1  # [0.5, 0.7)
    
    methods = ['L2', 'L1', 'TV']
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    x = np.arange(len(zones))
    width = 0.35
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        
        q_orig = results_dict.get(f'{method}_original', np.zeros(len(conformal_radii)))
        q_weight = results_dict.get(f'{method}_weighted', np.zeros(len(conformal_radii)))
        
        # Compute intensity fractions per zone
        orig_fracs = []
        weight_fracs = []
        total_orig = np.sum(np.abs(q_orig))
        total_weight = np.sum(np.abs(q_weight))
        
        for z_lo, z_hi in zones:
            mask = (conformal_radii >= z_lo) & (conformal_radii < z_hi)
            orig_fracs.append(np.sum(np.abs(q_orig[mask])) / total_orig * 100 if total_orig > 0 else 0)
            weight_fracs.append(np.sum(np.abs(q_weight[mask])) / total_weight * 100 if total_weight > 0 else 0)
        
        # Plot bars
        bars1 = ax.bar(x - width/2, orig_fracs, width, label='Original', color='coral', alpha=0.8)
        bars2 = ax.bar(x + width/2, weight_fracs, width, label='Weighted', color='steelblue', alpha=0.8)
        
        # Highlight target zone
        ax.axvspan(target_zone_idx - 0.5, target_zone_idx + 0.5, alpha=0.2, color='green')
        
        ax.set_xlabel('Radial Zone (conformal)', fontsize=11)
        ax.set_ylabel('Intensity Fraction (%)', fontsize=11)
        ax.set_title(f'{method} Regularization', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(zone_labels, fontsize=9)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add target zone annotation
        if idx == 1:  # Middle plot
            ax.annotate('Target\nzone', xy=(target_zone_idx, 80), fontsize=9,
                       ha='center', color='darkgreen')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Figure C saved: {output_path}")


# =============================================================================
# Run all linear methods (original and weighted)
# =============================================================================

def run_all_methods(G, u_data, conformal_radii, sigma_noise, n_sensors):
    """Run L2, L1, TV with both original and weighted approaches."""
    results = {}
    stats = {}
    
    Mg = G.shape[1]
    
    # Compatibility constraint matrix
    A_eq = np.ones((1, Mg))
    b_eq = np.array([0.0])
    
    methods_config = [
        ('L2', 'l2'),
        ('L1', 'l1'),
        ('TV', 'tv'),
    ]
    
    for method_name, reg_type in methods_config:
        print(f"\n  Running {method_name}...")
        
        # --- ORIGINAL (no weighting, L-curve) ---
        try:
            alpha_orig, _, _ = compute_l_curve_weighted(
                G, u_data, reg_type=reg_type, 
                weights=None, noise_level=None,  # L-curve selection
                n_alphas=50
            )
            
            if reg_type == 'l2':
                q_orig = solve_l2_weighted(G, u_data, alpha_orig, A_eq, b_eq, weights=None)
            elif reg_type == 'l1':
                q_orig = solve_l1_weighted_admm(G, u_data, alpha_orig, A_eq, b_eq, weights=None)
            else:  # tv
                q_orig = solve_tv_weighted(G, u_data, alpha_orig, A_eq, b_eq, weights=None)
            
            results[f'{method_name}_original'] = q_orig
            stats[f'{method_name}_original_alpha'] = alpha_orig
        except Exception as e:
            print(f"    Original failed: {e}")
            results[f'{method_name}_original'] = np.zeros(Mg)
            stats[f'{method_name}_original_alpha'] = 0.0
        
        # --- WEIGHTED (beta=1.0, discrepancy principle) ---
        try:
            beta = 1.0
            weights = compute_depth_weights(conformal_radii, beta=beta)
            
            alpha_weight, _, _ = compute_l_curve_weighted(
                G, u_data, reg_type=reg_type,
                weights=weights, noise_level=sigma_noise,  # Discrepancy principle
                n_alphas=50
            )
            
            if reg_type == 'l2':
                q_weight = solve_l2_weighted(G, u_data, alpha_weight, A_eq, b_eq, weights=weights)
            elif reg_type == 'l1':
                q_weight = solve_l1_weighted_admm(G, u_data, alpha_weight, A_eq, b_eq, weights=weights)
            else:  # tv
                q_weight = solve_tv_weighted(G, u_data, alpha_weight, A_eq, b_eq, weights=weights)
            
            results[f'{method_name}_weighted'] = q_weight
            stats[f'{method_name}_weighted_alpha'] = alpha_weight
        except Exception as e:
            print(f"    Weighted failed: {e}")
            results[f'{method_name}_weighted'] = np.zeros(Mg)
            stats[f'{method_name}_weighted_alpha'] = 0.0
    
    return results, stats


# =============================================================================
# Compute summary statistics
# =============================================================================

def compute_summary_stats(results, conformal_radii, true_rho):
    """Compute target zone percentage and other stats."""
    summary = {}
    
    # Target zone: where true sources are
    rho_min, rho_max = true_rho.min(), true_rho.max()
    target_mask = (conformal_radii >= rho_min - 0.05) & (conformal_radii <= rho_max + 0.05)
    
    for key, q in results.items():
        total_intensity = np.sum(np.abs(q))
        if total_intensity > 0:
            target_intensity = np.sum(np.abs(q[target_mask]))
            target_pct = 100 * target_intensity / total_intensity
        else:
            target_pct = 0.0
        
        summary[key] = {
            'target_pct': target_pct,
            'total_intensity': total_intensity,
            'n_active': np.sum(np.abs(q) > 0.01 * np.abs(q).max())
        }
    
    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate boundary bias publication figures')
    parser.add_argument('--domain', type=str, default='disk', 
                        choices=['disk', 'ellipse', 'brain'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-sources', type=int, default=4)
    parser.add_argument('--rho-min', type=float, default=0.5)
    parser.add_argument('--rho-max', type=float, default=0.7)
    parser.add_argument('--n-sensors', type=int, default=100)
    parser.add_argument('--sigma-noise', type=float, default=0.001)
    parser.add_argument('--n-radial', type=int, default=15)
    parser.add_argument('--n-angular', type=int, default=24)
    parser.add_argument('--output-dir', type=str, default='./figs_boundary_bias')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"=" * 60)
    print(f"Generating Boundary Bias Figures")
    print(f"Domain: {args.domain}, Seed: {args.seed}")
    print(f"Sources: N={args.n_sources}, rho in [{args.rho_min}, {args.rho_max}]")
    print(f"=" * 60)
    
    # --- Setup domain ---
    cmap = None
    if args.domain != 'disk':
        print("\nComputing conformal map...")
        boundary_func = get_domain_boundary(args.domain)
        cmap = MFSConformalMap(boundary_func, n_boundary=256, n_charge=200)
    
    # --- Generate grid ---
    print("\nGenerating interior grid...")
    grid_points, conformal_radii = generate_grid_points(
        args.domain, args.n_radial, args.n_angular, cmap
    )
    print(f"  Grid size: {len(grid_points)} points")
    
    # --- Generate sensors ---
    print("Generating sensors...")
    sensors = generate_sensors(args.domain, args.n_sensors, cmap)
    
    # --- Generate true sources ---
    print("Generating true sources...")
    true_positions, true_intensities, true_rho = generate_true_sources(
        args.domain, args.n_sources, args.rho_min, args.rho_max, args.seed, cmap
    )
    print(f"  True source radii: {true_rho}")
    
    # --- Build Green's matrix ---
    print("Building Green's matrix...")
    G = build_green_matrix(args.domain, grid_points, sensors, cmap)
    print(f"  Matrix size: {G.shape}")
    
    # --- Compute forward data ---
    print("Computing forward data...")
    u_data, u_exact = compute_forward_data(
        true_positions, true_intensities, sensors, 
        args.domain, args.sigma_noise, args.seed, cmap
    )
    
    # === FIGURE A: Column Norm Sensitivity ===
    print("\n--- Figure A: Column Norm Sensitivity ---")
    fig_a_path = os.path.join(args.output_dir, f'fig_A_column_norm_{args.domain}.png')
    sensitivity_ratio = figure_A_column_norm_sensitivity(G, conformal_radii, fig_a_path)
    
    # === Run all methods ===
    print("\n--- Running Linear Methods ---")
    results, method_stats = run_all_methods(G, u_data, conformal_radii, 
                                            args.sigma_noise, args.n_sensors)
    
    # === FIGURE B: Heatmap Grid ===
    print("\n--- Figure B: Heatmap Grid ---")
    fig_b_path = os.path.join(args.output_dir, f'fig_B_heatmaps_{args.domain}.png')
    figure_B_heatmap_grid(results, grid_points, true_positions, 
                          args.domain, fig_b_path, cmap)
    
    # === FIGURE C: Intensity Distribution ===
    print("\n--- Figure C: Intensity Distribution ---")
    fig_c_path = os.path.join(args.output_dir, f'fig_C_intensity_dist_{args.domain}.png')
    figure_C_intensity_distribution(results, conformal_radii, true_rho, fig_c_path)
    
    # === Summary Statistics ===
    print("\n--- Summary Statistics ---")
    summary = compute_summary_stats(results, conformal_radii, true_rho)
    
    # Print table
    print("\n  Method              | Alpha      | Target %")
    print("  " + "-" * 45)
    for method in ['L2', 'L1', 'TV']:
        for variant in ['original', 'weighted']:
            key = f'{method}_{variant}'
            alpha = method_stats.get(f'{key}_alpha', 0)
            target = summary[key]['target_pct']
            print(f"  {key:20s} | {alpha:.2e} | {target:5.1f}%")
    
    # Save summary to JSON
    summary_path = os.path.join(args.output_dir, f'summary_{args.domain}_seed{args.seed}.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'domain': args.domain,
            'seed': args.seed,
            'n_sources': args.n_sources,
            'rho_range': [args.rho_min, args.rho_max],
            'true_rho': true_rho.tolist(),
            'sensitivity_ratio': sensitivity_ratio,
            'method_stats': method_stats,
            'summary': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in summary.items()}
        }, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")
    
    print(f"\n{'=' * 60}")
    print("DONE! Figures saved to:", args.output_dir)
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
