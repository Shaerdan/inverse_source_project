#!/usr/bin/env python3
"""
Generate publication figures for boundary bias analysis.

USAGE: Place this file in src/ directory alongside mesh.py, conformal_solver.py, etc.
       Run from src/ directory:
       
       cd src
       python generate_boundary_bias_figures_v2.py --domain disk --output-dir ../figures
       python generate_boundary_bias_figures_v2.py --domain ellipse --output-dir ../figures
       python generate_boundary_bias_figures_v2.py --domain brain --output-dir ../figures

Generates:
    Figure A: Column norm ||G_j|| vs conformal radius (sensitivity analysis)
    Figure B: 3x2 heatmap grid comparing original vs weighted L2/L1/TV
    Figure C: Intensity distribution by radial zone
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from matplotlib.colors import Normalize
import argparse
import os
import sys

# Import from local modules (assumes script is in src/ directory)
from mesh import get_brain_boundary
from conformal_solver import MFSConformalMap
from depth_weighted_solvers import (
    DepthWeightedL2Solver,
    DepthWeightedL1Solver, 
    DepthWeightedTVSolver,
    compute_conformal_radii_disk
)


# =============================================================================
# Domain boundary definitions (matching run_comparison_job.py)
# =============================================================================

ELLIPSE_A = 1.5
ELLIPSE_B = 0.8


def get_domain_setup(domain_type):
    """
    Get boundary function and conformal map for domain.
    
    Returns
    -------
    boundary_func : callable
        t -> complex, boundary parameterization
    cmap : MFSConformalMap or None
        Conformal map (None for disk)
    plot_boundary : callable
        Function to add boundary to matplotlib axis
    """
    if domain_type == 'disk':
        def boundary_func(t):
            return np.exp(1j * t)
        
        def plot_boundary(ax):
            theta = np.linspace(0, 2*np.pi, 200)
            ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
        
        return boundary_func, None, plot_boundary
    
    elif domain_type == 'ellipse':
        def boundary_func(t):
            return complex(ELLIPSE_A * np.cos(t), ELLIPSE_B * np.sin(t))
        
        cmap = MFSConformalMap(boundary_func, n_boundary=256, n_charge=200)
        
        def plot_boundary(ax):
            ellipse = Ellipse((0, 0), 2*ELLIPSE_A, 2*ELLIPSE_B, 
                            fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(ellipse)
        
        return boundary_func, cmap, plot_boundary
    
    elif domain_type == 'brain':
        # Use actual brain boundary from mesh.py
        boundary_pts = get_brain_boundary(n_points=200)
        
        def boundary_func(t):
            """Interpolate brain boundary points."""
            n = len(boundary_pts)
            # Normalize t to [0, 2pi)
            t = t % (2 * np.pi)
            idx = int(t / (2 * np.pi) * n) % n
            next_idx = (idx + 1) % n
            frac = (t / (2 * np.pi) * n) - idx
            x = boundary_pts[idx, 0] * (1 - frac) + boundary_pts[next_idx, 0] * frac
            y = boundary_pts[idx, 1] * (1 - frac) + boundary_pts[next_idx, 1] * frac
            return complex(x, y)
        
        cmap = MFSConformalMap(boundary_func, n_boundary=256, n_charge=200)
        
        def plot_boundary(ax):
            # Close the polygon
            pts = np.vstack([boundary_pts, boundary_pts[0]])
            ax.plot(pts[:, 0], pts[:, 1], 'k-', linewidth=2)
        
        return boundary_func, cmap, plot_boundary
    
    else:
        raise ValueError(f"Unknown domain: {domain_type}")


def generate_grid(domain_type, cmap, n_radial=20, n_angular=40):
    """
    Generate interior grid points with conformal radii.
    
    Returns
    -------
    grid_physical : ndarray (N, 2)
        Grid points in physical coordinates
    grid_disk : ndarray (N, 2)  
        Grid points in disk coordinates
    conformal_radii : ndarray (N,)
        Conformal radius of each point
    """
    # Generate grid in disk coordinates
    radii = np.linspace(0.05, 0.95, n_radial)
    angles = np.linspace(0, 2*np.pi, n_angular, endpoint=False)
    
    disk_points = []
    for r in radii:
        for theta in angles:
            disk_points.append([r * np.cos(theta), r * np.sin(theta)])
    disk_points = np.array(disk_points)
    
    conformal_radii = np.sqrt(disk_points[:, 0]**2 + disk_points[:, 1]**2)
    
    if cmap is None:
        # Disk domain - physical = disk
        physical_points = disk_points.copy()
    else:
        # Map to physical domain
        physical_points = np.zeros_like(disk_points)
        for i, (x, y) in enumerate(disk_points):
            w = complex(x, y)
            z = cmap.from_disk(w)
            physical_points[i] = [z.real, z.imag]
    
    return physical_points, disk_points, conformal_radii


def generate_sources(n_sources, rho_range=(0.5, 0.7), seed=42):
    """
    Generate source configuration in disk coordinates.
    
    Returns list of ((x, y), intensity) tuples in disk coordinates.
    """
    np.random.seed(seed)
    
    sources = []
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.2, 0.2, n_sources)  # Small perturbation
    
    for i, theta in enumerate(angles):
        r = np.random.uniform(rho_range[0], rho_range[1])
        x, y = r * np.cos(theta), r * np.sin(theta)
        intensity = (-1)**i * np.random.uniform(0.8, 1.2)
        sources.append(((x, y), intensity))
    
    # Center intensities
    total = sum(s[1] for s in sources)
    sources = [((s[0][0], s[0][1]), s[1] - total/n_sources) for s in sources]
    
    return sources


def map_sources_to_physical(disk_sources, cmap):
    """Map sources from disk to physical coordinates."""
    if cmap is None:
        return disk_sources
    
    physical_sources = []
    for (x, y), intensity in disk_sources:
        w = complex(x, y)
        z = cmap.from_disk(w)
        physical_sources.append(((z.real, z.imag), intensity))
    return physical_sources


def forward_solve(sources, n_sensors, cmap=None):
    """
    Compute boundary potential from sources using Neumann Green's function.
    
    For disk: uses analytical formula (Poisson kernel form)
    For conformal domains: maps sources to disk, uses disk formula
    
    This matches run_comparison_job.py forward_solve_disk/forward_solve_conformal.
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
        # This exploits conformal invariance of harmonic functions
        u = np.zeros(n_sensors)
        sensor_positions = []
        
        # Precompute sensor positions in physical domain
        for theta in theta_sensors:
            w_sensor = np.exp(1j * theta)
            z_sensor = cmap.from_disk(w_sensor)
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
    
    return u, theta_sensors, sensor_positions


def build_greens_matrix(grid_physical, sensor_positions, theta_sensors, cmap=None):
    """
    Build Green's matrix using Neumann boundary formula.
    
    Matches run_comparison_job.py build_greens_matrix_disk/build_greens_matrix_conformal.
    
    For each grid point:
    1. Map to disk coordinates (if conformal domain)
    2. Use disk boundary formula: -1/(2π) * log(1 + r² - 2r*cos(θ - φ))
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
    
    # Center columns (removes constant offset, standard for Neumann)
    G -= np.mean(G, axis=0, keepdims=True)
    
    return G


# =============================================================================
# Figure Generation Functions
# =============================================================================

def generate_figure_A(G, conformal_radii, domain_type, output_dir):
    """
    Figure A: Column norm vs conformal radius.
    
    Shows the ~20x sensitivity variation that causes boundary bias.
    """
    column_norms = np.linalg.norm(G, axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(conformal_radii, column_norms, alpha=0.5, s=20)
    
    # Fit power law: ||G_j|| ~ C(1-rho)^gamma
    log_x = np.log(1 - conformal_radii + 1e-6)
    log_y = np.log(column_norms)
    valid = np.isfinite(log_x) & np.isfinite(log_y)
    
    if np.sum(valid) > 10:
        coeffs = np.polyfit(log_x[valid], log_y[valid], 1)
        gamma = coeffs[0]
        C = np.exp(coeffs[1])
        
        rho_fit = np.linspace(0.05, 0.95, 100)
        norm_fit = C * (1 - rho_fit) ** gamma
        ax.plot(rho_fit, norm_fit, 'r-', linewidth=2, 
               label=f'Fit: $\\|G_j\\| \\sim {C:.1f}(1-\\rho)^{{{gamma:.2f}}}$')
        ax.legend(fontsize=12)
    
    # Mark target zone
    ax.axvspan(0.5, 0.7, alpha=0.2, color='green', label='Target zone')
    
    ax.set_xlabel('Conformal radius $\\rho_j$', fontsize=14)
    ax.set_ylabel('Column norm $\\|G_j\\|$', fontsize=14)
    ax.set_title(f'Green\'s Function Sensitivity ({domain_type.capitalize()} Domain)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add ratio annotation
    norm_center = np.median(column_norms[conformal_radii < 0.3])
    norm_edge = np.median(column_norms[conformal_radii > 0.9])
    ratio = norm_edge / norm_center
    ax.text(0.95, 0.95, f'Edge/Center ratio: {ratio:.1f}×', 
           transform=ax.transAxes, ha='right', va='top', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, f'fig_A_column_norms_{domain_type}.pdf')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.savefig(filepath.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filepath}")
    return gamma if 'gamma' in dir() else None


def generate_figure_B(domain_type, cmap, plot_boundary, grid_physical, grid_disk,
                     conformal_radii, true_sources_physical, true_sources_disk,
                     u_noisy, sensor_positions, theta_sensors, sigma_noise, n_sensors, output_dir):
    """
    Figure B: 3x2 heatmap grid comparing original vs weighted methods.
    """
    G = build_greens_matrix(grid_physical, sensor_positions, theta_sensors, cmap)
    
    # Compute weights
    weights = 1.0 / (1.0 - conformal_radii + 0.05) ** 1.0  # beta=1.0
    
    # Target residual for discrepancy principle
    target_residual = 1.3 * sigma_noise * np.sqrt(n_sensors)
    
    # Solve with each method
    methods = ['L2', 'L1', 'TV']
    results_original = {}
    results_weighted = {}
    
    for method in methods:
        if method == 'L2':
            solver_orig = DepthWeightedL2Solver(G)
            solver_weight = DepthWeightedL2Solver(G, depth_weights=weights)
        elif method == 'L1':
            solver_orig = DepthWeightedL1Solver(G)
            solver_weight = DepthWeightedL1Solver(G, depth_weights=weights)
        else:  # TV
            solver_orig = DepthWeightedTVSolver(G, grid_physical)
            solver_weight = DepthWeightedTVSolver(G, grid_physical, depth_weights=weights)
        
        results_original[method] = solver_orig.solve(u_noisy, alpha_selection='discrepancy',
                                                     target_residual=target_residual)
        results_weighted[method] = solver_weight.solve(u_noisy, alpha_selection='discrepancy',
                                                       target_residual=target_residual)
    
    # Create 3x2 figure
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    
    # Get domain bounds
    x_phys = grid_physical[:, 0]
    y_phys = grid_physical[:, 1]
    x_margin = 0.1 * (x_phys.max() - x_phys.min())
    y_margin = 0.1 * (y_phys.max() - y_phys.min())
    xlim = (x_phys.min() - x_margin, x_phys.max() + x_margin)
    ylim = (y_phys.min() - y_margin, y_phys.max() + y_margin)
    
    for row, method in enumerate(methods):
        for col, (result_dict, title_suffix) in enumerate([
            (results_original, 'Original'),
            (results_weighted, 'Weighted')
        ]):
            ax = axes[row, col]
            q = result_dict[method]['solution']
            
            # Scatter plot with intensity as color
            q_abs = np.abs(q)
            vmax = np.percentile(q_abs, 99)
            
            scatter = ax.scatter(grid_physical[:, 0], grid_physical[:, 1],
                               c=q_abs, cmap='hot', s=15, alpha=0.8,
                               norm=Normalize(vmin=0, vmax=vmax))
            
            # Add domain boundary
            plot_boundary(ax)
            
            # Mark true sources
            for (x, y), intensity in true_sources_physical:
                ax.scatter([x], [y], marker='x', s=200, c='cyan', linewidths=3, zorder=10)
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal')
            ax.set_title(f'{method} {title_suffix}', fontsize=14)
            
            if col == 0:
                ax.set_ylabel('y', fontsize=12)
            if row == 2:
                ax.set_xlabel('x', fontsize=12)
            
            plt.colorbar(scatter, ax=ax, label='$|q_j|$')
    
    plt.suptitle(f'Reconstruction Comparison ({domain_type.capitalize()} Domain)\n'
                f'True sources marked with ×', fontsize=14, y=1.02)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'fig_B_heatmaps_{domain_type}.pdf')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.savefig(filepath.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filepath}")
    
    return results_original, results_weighted


def generate_figure_C(conformal_radii, results_original, results_weighted, 
                     domain_type, output_dir):
    """
    Figure C: Intensity distribution by radial zone.
    """
    zones = [(0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    zone_labels = ['[0, 0.5)', '[0.5, 0.7)\n(target)', '[0.7, 0.9)', '[0.9, 1.0)']
    
    methods = ['L2', 'L1', 'TV']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(len(zones))
    width = 0.35
    
    for ax, method in zip(axes, methods):
        q_orig = results_original[method]['solution']
        q_weight = results_weighted[method]['solution']
        
        # Compute zone percentages
        total_orig = np.sum(np.abs(q_orig))
        total_weight = np.sum(np.abs(q_weight))
        
        pct_orig = []
        pct_weight = []
        
        for z_min, z_max in zones:
            mask = (conformal_radii >= z_min) & (conformal_radii < z_max)
            pct_orig.append(100 * np.sum(np.abs(q_orig[mask])) / total_orig)
            pct_weight.append(100 * np.sum(np.abs(q_weight[mask])) / total_weight)
        
        bars1 = ax.bar(x - width/2, pct_orig, width, label='Original', color='salmon')
        bars2 = ax.bar(x + width/2, pct_weight, width, label='Weighted', color='steelblue')
        
        ax.set_ylabel('% of total $\\sum|q_j|$', fontsize=12)
        ax.set_xlabel('Conformal radius zone', fontsize=12)
        ax.set_title(f'{method}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(zone_labels)
        ax.legend()
        ax.set_ylim(0, 100)
        
        # Highlight target zone
        ax.axvspan(0.5, 1.5, alpha=0.2, color='green')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    plt.suptitle(f'Intensity Distribution by Radial Zone ({domain_type.capitalize()} Domain)', 
                fontsize=14)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'fig_C_zones_{domain_type}.pdf')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.savefig(filepath.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Generate boundary bias figures')
    parser.add_argument('--domain', type=str, default='disk',
                       choices=['disk', 'ellipse', 'brain'],
                       help='Domain type')
    parser.add_argument('--output-dir', type=str, default='../figures',
                       help='Output directory for figures')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--n-sources', type=int, default=4,
                       help='Number of true sources')
    parser.add_argument('--n-sensors', type=int, default=64,
                       help='Number of boundary sensors')
    parser.add_argument('--sigma-noise', type=float, default=0.01,
                       help='Noise standard deviation')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating figures for {args.domain} domain")
    print(f"{'='*60}")
    
    # Setup domain
    print("\n1. Setting up domain...")
    boundary_func, cmap, plot_boundary = get_domain_setup(args.domain)
    
    # Generate grid
    print("2. Generating interior grid...")
    grid_physical, grid_disk, conformal_radii = generate_grid(args.domain, cmap)
    print(f"   Grid points: {len(grid_physical)}")
    
    # Generate sources
    print("3. Generating sources...")
    disk_sources = generate_sources(args.n_sources, rho_range=(0.5, 0.7), seed=args.seed)
    physical_sources = map_sources_to_physical(disk_sources, cmap)
    print(f"   Sources: {len(physical_sources)}")
    
    # Forward solve
    print("4. Computing forward solution...")
    u_true, theta_sensors, sensor_positions = forward_solve(physical_sources, args.n_sensors, cmap)
    
    # Add noise
    np.random.seed(args.seed + 1000)
    noise = np.random.randn(args.n_sensors) * args.sigma_noise
    u_noisy = u_true + noise
    
    # Build Green's matrix for Figure A
    print("5. Building Green's matrix...")
    G = build_greens_matrix(grid_physical, sensor_positions, theta_sensors, cmap)
    
    # Generate figures
    print("\n6. Generating figures...")
    
    print("   Figure A: Column norms...")
    generate_figure_A(G, conformal_radii, args.domain, args.output_dir)
    
    print("   Figure B: Heatmaps...")
    results_orig, results_weight = generate_figure_B(
        args.domain, cmap, plot_boundary, grid_physical, grid_disk,
        conformal_radii, physical_sources, disk_sources,
        u_noisy, sensor_positions, theta_sensors, args.sigma_noise, args.n_sensors, args.output_dir
    )
    
    print("   Figure C: Zone distribution...")
    generate_figure_C(conformal_radii, results_orig, results_weight, 
                     args.domain, args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"All figures saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
