#!/usr/bin/env python3
"""
Multi-seed statistical validation for boundary bias analysis.

USAGE: Place this file in src/ directory alongside mesh.py, conformal_solver.py, etc.
       Run from src/ directory:
       
       cd src
       python run_multiseed_validation_v2.py --domain disk --n-seeds 50 --output-dir ../results
       
Generates:
    Figure D: Box plots comparing target zone recovery across seeds
    Statistical summary with Mann-Whitney tests
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse
import os
import json
from datetime import datetime

# Import from local modules (assumes script is in src/ directory)
from mesh import get_brain_boundary
from conformal_solver import MFSConformalMap
from depth_weighted_solvers import (
    DepthWeightedL2Solver,
    DepthWeightedL1Solver, 
    DepthWeightedTVSolver,
)


# =============================================================================
# Domain setup (matching run_comparison_job.py)
# =============================================================================

ELLIPSE_A = 1.5
ELLIPSE_B = 0.8


def get_domain_setup(domain_type):
    """Get boundary function and conformal map for domain."""
    if domain_type == 'disk':
        def boundary_func(t):
            return np.exp(1j * t)
        return boundary_func, None
    
    elif domain_type == 'ellipse':
        def boundary_func(t):
            return complex(ELLIPSE_A * np.cos(t), ELLIPSE_B * np.sin(t))
        cmap = MFSConformalMap(boundary_func, n_boundary=256, n_charge=200)
        return boundary_func, cmap
    
    elif domain_type == 'brain':
        boundary_pts = get_brain_boundary(n_points=200)
        
        def boundary_func(t):
            n = len(boundary_pts)
            t = t % (2 * np.pi)
            idx = int(t / (2 * np.pi) * n) % n
            next_idx = (idx + 1) % n
            frac = (t / (2 * np.pi) * n) - idx
            x = boundary_pts[idx, 0] * (1 - frac) + boundary_pts[next_idx, 0] * frac
            y = boundary_pts[idx, 1] * (1 - frac) + boundary_pts[next_idx, 1] * frac
            return complex(x, y)
        
        cmap = MFSConformalMap(boundary_func, n_boundary=256, n_charge=200)
        return boundary_func, cmap
    
    else:
        raise ValueError(f"Unknown domain: {domain_type}")


def generate_grid(cmap, n_radial=20, n_angular=40):
    """Generate interior grid in disk coordinates, map to physical if needed."""
    radii = np.linspace(0.05, 0.95, n_radial)
    angles = np.linspace(0, 2*np.pi, n_angular, endpoint=False)
    
    disk_points = []
    for r in radii:
        for theta in angles:
            disk_points.append([r * np.cos(theta), r * np.sin(theta)])
    disk_points = np.array(disk_points)
    
    conformal_radii = np.sqrt(disk_points[:, 0]**2 + disk_points[:, 1]**2)
    
    if cmap is None:
        physical_points = disk_points.copy()
    else:
        physical_points = np.zeros_like(disk_points)
        for i, (x, y) in enumerate(disk_points):
            z = cmap.from_disk(complex(x, y))
            physical_points[i] = [z.real, z.imag]
    
    return physical_points, disk_points, conformal_radii


def generate_sources(n_sources, rho_range=(0.5, 0.7), seed=42):
    """Generate source configuration in disk coordinates."""
    np.random.seed(seed)
    
    sources = []
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.2, 0.2, n_sources)
    
    for i, theta in enumerate(angles):
        r = np.random.uniform(rho_range[0], rho_range[1])
        x, y = r * np.cos(theta), r * np.sin(theta)
        intensity = (-1)**i * np.random.uniform(0.8, 1.2)
        sources.append(((x, y), intensity))
    
    total = sum(s[1] for s in sources)
    sources = [((s[0][0], s[0][1]), s[1] - total/n_sources) for s in sources]
    
    return sources


def map_sources_to_physical(disk_sources, cmap):
    """Map sources from disk to physical coordinates."""
    if cmap is None:
        return disk_sources
    
    physical_sources = []
    for (x, y), intensity in disk_sources:
        z = cmap.from_disk(complex(x, y))
        physical_sources.append(((z.real, z.imag), intensity))
    return physical_sources


def forward_solve(sources, n_sensors, cmap=None):
    """
    Compute boundary potential from sources using Neumann Green's function.
    
    Matches run_comparison_job.py forward_solve_disk/forward_solve_conformal.
    """
    theta_sensors = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
    
    if cmap is None:
        # Disk domain - analytical Neumann Green's function formula
        u = np.zeros(n_sensors)
        for (x, y), q in sources:
            r = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            for j, theta in enumerate(theta_sensors):
                angle_diff = theta - phi
                u[j] += q * (-1.0 / (2 * np.pi)) * np.log(1 + r**2 - 2*r*np.cos(angle_diff))
        sensor_positions = [(np.cos(t), np.sin(t)) for t in theta_sensors]
    else:
        # Conformal domain - map sources to disk, use disk formula
        u = np.zeros(n_sensors)
        sensor_positions = []
        
        # Precompute sensor positions in physical domain
        for theta in theta_sensors:
            z_sensor = cmap.from_disk(np.exp(1j * theta))
            sensor_positions.append((z_sensor.real, z_sensor.imag))
        
        for (x, y), q in sources:
            # Map source from physical to disk coordinates
            z = complex(x, y)
            w = cmap.to_disk(z)
            r = abs(w)
            phi = np.angle(w)
            
            # Apply disk formula with disk coordinates
            for j, theta in enumerate(theta_sensors):
                angle_diff = theta - phi
                u[j] += q * (-1.0 / (2 * np.pi)) * np.log(1 + r**2 - 2*r*np.cos(angle_diff))
    
    # Remove mean (gauge freedom in Neumann problem)
    u -= np.mean(u)
    
    theta_sensors = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
    return u, theta_sensors, sensor_positions


def build_greens_matrix(grid_physical, sensor_positions, theta_sensors, cmap=None):
    """
    Build Green's matrix using Neumann boundary formula.
    
    Matches run_comparison_job.py build_greens_matrix_disk/build_greens_matrix_conformal.
    """
    n_sensors = len(sensor_positions)
    n_grid = len(grid_physical)
    G = np.zeros((n_sensors, n_grid))
    
    for j in range(n_grid):
        if cmap is None:
            # Disk domain - grid is already in disk coords
            r_j = np.sqrt(grid_physical[j, 0]**2 + grid_physical[j, 1]**2)
            phi_j = np.arctan2(grid_physical[j, 1], grid_physical[j, 0])
        else:
            # Conformal domain - map grid point to disk
            z_j = complex(grid_physical[j, 0], grid_physical[j, 1])
            w_j = cmap.to_disk(z_j)
            r_j = abs(w_j)
            phi_j = np.angle(w_j)
        
        # Keep away from center and boundary
        r_j = max(r_j, 1e-10)
        r_j = min(r_j, 0.999)
        
        for i in range(n_sensors):
            angle_diff = theta_sensors[i] - phi_j
            G[i, j] = -1.0 / (2 * np.pi) * np.log(1 + r_j**2 - 2*r_j*np.cos(angle_diff))
    
    # Center columns
    G -= np.mean(G, axis=0, keepdims=True)
    
    return G


def compute_target_percentage(q, conformal_radii, target_range=(0.5, 0.7)):
    """Compute percentage of intensity in target zone."""
    mask = (conformal_radii >= target_range[0]) & (conformal_radii < target_range[1])
    total = np.sum(np.abs(q))
    if total < 1e-12:
        return 0.0
    return 100.0 * np.sum(np.abs(q[mask])) / total


def run_single_seed(seed, domain_type, cmap, n_sources=4, n_sensors=64, 
                   sigma_noise=0.01, beta=1.0):
    """Run analysis for a single seed, return target percentages."""
    
    # Generate grid (same for all seeds)
    grid_physical, grid_disk, conformal_radii = generate_grid(cmap)
    
    # Generate sources
    disk_sources = generate_sources(n_sources, rho_range=(0.5, 0.7), seed=seed)
    physical_sources = map_sources_to_physical(disk_sources, cmap)
    
    # Forward solve
    u_true, theta_sensors, sensor_positions = forward_solve(physical_sources, n_sensors, cmap)
    
    # Add noise
    np.random.seed(seed + 1000)
    noise = np.random.randn(n_sensors) * sigma_noise
    u_noisy = u_true + noise
    
    # Build Green's matrix
    G = build_greens_matrix(grid_physical, sensor_positions, theta_sensors, cmap)
    
    # Compute weights
    weights = 1.0 / (1.0 - conformal_radii + 0.05) ** beta
    
    # Target residual
    target_residual = 1.3 * sigma_noise * np.sqrt(n_sensors)
    
    results = {}
    
    for method in ['L2', 'L1', 'TV']:
        if method == 'L2':
            solver_orig = DepthWeightedL2Solver(G)
            solver_weight = DepthWeightedL2Solver(G, depth_weights=weights)
        elif method == 'L1':
            solver_orig = DepthWeightedL1Solver(G)
            solver_weight = DepthWeightedL1Solver(G, depth_weights=weights)
        else:
            solver_orig = DepthWeightedTVSolver(G, grid_physical)
            solver_weight = DepthWeightedTVSolver(G, grid_physical, depth_weights=weights)
        
        q_orig = solver_orig.solve(u_noisy, alpha_selection='discrepancy',
                                   target_residual=target_residual)['solution']
        q_weight = solver_weight.solve(u_noisy, alpha_selection='discrepancy',
                                       target_residual=target_residual)['solution']
        
        results[f'{method}_original'] = compute_target_percentage(q_orig, conformal_radii)
        results[f'{method}_weighted'] = compute_target_percentage(q_weight, conformal_radii)
    
    return results


def run_multiseed(domain_type, n_seeds, n_sources=4, n_sensors=64, 
                  sigma_noise=0.01, beta=1.0, start_seed=0):
    """Run analysis across multiple seeds."""
    
    print(f"\nSetting up {domain_type} domain...")
    _, cmap = get_domain_setup(domain_type)
    
    all_results = {
        'L2_original': [], 'L2_weighted': [],
        'L1_original': [], 'L1_weighted': [],
        'TV_original': [], 'TV_weighted': [],
    }
    
    print(f"Running {n_seeds} seeds...")
    for i, seed in enumerate(range(start_seed, start_seed + n_seeds)):
        if (i + 1) % 10 == 0:
            print(f"  Seed {i+1}/{n_seeds}...")
        
        try:
            results = run_single_seed(seed, domain_type, cmap, n_sources, 
                                     n_sensors, sigma_noise, beta)
            for key in all_results:
                all_results[key].append(results[key])
        except Exception as e:
            print(f"  Warning: Seed {seed} failed: {e}")
    
    # Convert to arrays
    for key in all_results:
        all_results[key] = np.array(all_results[key])
    
    return all_results


def generate_figure_D(all_results, domain_type, output_dir):
    """Generate box plot comparison figure."""
    
    methods = ['L2', 'L1', 'TV']
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    for ax, method in zip(axes, methods):
        data_orig = all_results[f'{method}_original']
        data_weight = all_results[f'{method}_weighted']
        
        # Box plot
        bp = ax.boxplot([data_orig, data_weight], labels=['Original', 'Weighted'],
                       patch_artist=True)
        
        bp['boxes'][0].set_facecolor('salmon')
        bp['boxes'][1].set_facecolor('steelblue')
        
        # Statistics
        median_orig = np.median(data_orig)
        median_weight = np.median(data_weight)
        
        # Mann-Whitney test
        stat, pval = stats.mannwhitneyu(data_weight, data_orig, alternative='greater')
        
        ax.set_ylabel('Target zone intensity (%)', fontsize=12)
        ax.set_title(f'{method}\n'
                    f'Median: {median_orig:.1f}% → {median_weight:.1f}%\n'
                    f'p = {pval:.2e}', fontsize=12)
        ax.set_ylim(0, 100)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% reference')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Target Zone Recovery ({domain_type.capitalize()} Domain)\n'
                f'n = {len(all_results["L2_original"])} seeds', fontsize=14)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'fig_D_multiseed_{domain_type}.pdf')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.savefig(filepath.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filepath}")


def generate_statistics_summary(all_results, domain_type, output_dir):
    """Generate statistical summary JSON and text files."""
    
    summary = {
        'domain': domain_type,
        'n_seeds': len(all_results['L2_original']),
        'timestamp': datetime.now().isoformat(),
        'methods': {}
    }
    
    methods = ['L2', 'L1', 'TV']
    
    for method in methods:
        data_orig = all_results[f'{method}_original']
        data_weight = all_results[f'{method}_weighted']
        
        stat, pval = stats.mannwhitneyu(data_weight, data_orig, alternative='greater')
        
        summary['methods'][method] = {
            'original': {
                'median': float(np.median(data_orig)),
                'mean': float(np.mean(data_orig)),
                'std': float(np.std(data_orig)),
                'q25': float(np.percentile(data_orig, 25)),
                'q75': float(np.percentile(data_orig, 75)),
            },
            'weighted': {
                'median': float(np.median(data_weight)),
                'mean': float(np.mean(data_weight)),
                'std': float(np.std(data_weight)),
                'q25': float(np.percentile(data_weight, 25)),
                'q75': float(np.percentile(data_weight, 75)),
            },
            'improvement': float(np.median(data_weight) - np.median(data_orig)),
            'mann_whitney_U': float(stat),
            'mann_whitney_p': float(pval),
        }
    
    # Save JSON
    json_path = os.path.join(output_dir, f'statistics_{domain_type}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save text summary
    txt_path = os.path.join(output_dir, f'statistics_{domain_type}.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Statistical Summary: {domain_type.capitalize()} Domain\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Number of seeds: {summary['n_seeds']}\n\n")
        
        for method in methods:
            m = summary['methods'][method]
            f.write(f"{method}:\n")
            f.write(f"  Original:  {m['original']['median']:.1f}% "
                   f"(IQR: {m['original']['q25']:.1f}-{m['original']['q75']:.1f}%)\n")
            f.write(f"  Weighted:  {m['weighted']['median']:.1f}% "
                   f"(IQR: {m['weighted']['q25']:.1f}-{m['weighted']['q75']:.1f}%)\n")
            f.write(f"  Improvement: +{m['improvement']:.1f}%\n")
            f.write(f"  Mann-Whitney p = {m['mann_whitney_p']:.2e}\n\n")
    
    print(f"  Saved: {json_path}")
    print(f"  Saved: {txt_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Multi-seed boundary bias validation')
    parser.add_argument('--domain', type=str, default='disk',
                       choices=['disk', 'ellipse', 'brain'])
    parser.add_argument('--n-seeds', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='../results')
    parser.add_argument('--n-sources', type=int, default=4)
    parser.add_argument('--n-sensors', type=int, default=64)
    parser.add_argument('--sigma-noise', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--start-seed', type=int, default=0)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Multi-seed validation for {args.domain} domain")
    print(f"{'='*60}")
    
    # Run all seeds
    all_results = run_multiseed(
        args.domain, args.n_seeds, args.n_sources, args.n_sensors,
        args.sigma_noise, args.beta, args.start_seed
    )
    
    # Generate outputs
    print("\nGenerating outputs...")
    generate_figure_D(all_results, args.domain, args.output_dir)
    generate_statistics_summary(all_results, args.domain, args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
