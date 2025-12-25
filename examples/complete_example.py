#!/usr/bin/env python
"""
Complete Example: Inverse Source Localization
==============================================

This script demonstrates all major features of the package:
1. BEM forward and inverse solving
2. Conformal BEM for general domains
3. Multiple regularization methods
4. Parameter sweeps and L-curve analysis
5. Method comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent to path for development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from inverse_source import (
    bem_solver,
    conformal_bem,
    regularization,
    parameter_study,
    utils,
    config,
)


def example_bem_disk():
    """Example 1: BEM on unit disk."""
    print("\n" + "="*60)
    print("EXAMPLE 1: BEM on Unit Disk")
    print("="*60)
    
    # Define true sources
    sources_true = [
        ((-0.3, 0.4), 1.0),
        ((0.5, 0.3), 1.0),
        ((-0.4, -0.4), -1.0),
        ((0.3, -0.5), -1.0),
    ]
    
    print("\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources_true):
        print(f"  {i+1}: ({x:+.3f}, {y:+.3f}), q = {q:+.2f}")
    
    # Forward solve
    forward = bem_solver.BEMForwardSolver(n_boundary_points=100)
    u_clean = forward.solve(sources_true)
    
    # Add noise
    noise_level = 0.001
    u_measured = u_clean + noise_level * np.random.randn(len(u_clean))
    print(f"\nAdded noise with σ = {noise_level}")
    
    # Nonlinear inverse solve
    print("\nNonlinear solver (continuous source positions)...")
    inverse = bem_solver.BEMNonlinearInverseSolver(n_sources=4, n_boundary=100)
    inverse.set_measured_data(u_measured)
    sources_rec, result = inverse.solve(method='L-BFGS-B', maxiter=100)
    
    print("\nRecovered sources (nonlinear):")
    for i, ((x, y), q) in enumerate(sources_rec):
        print(f"  {i+1}: ({x:+.3f}, {y:+.3f}), q = {q:+.3f}")
    
    # Compute and display fit
    u_rec = forward.solve(sources_rec)
    residual = np.linalg.norm(u_rec - u_measured)
    print(f"\nResidual: {residual:.4e}")
    
    # Linear inverse solve
    print("\nLinear solver (grid-based with L1 regularization)...")
    linear = bem_solver.BEMLinearInverseSolver(n_boundary=100)
    linear.build_greens_matrix()
    
    # Find optimal alpha via L-curve
    alphas, residuals, reg_norms, alpha_opt = utils.l_curve_analysis(
        linear, u_measured, method='l1'
    )
    print(f"Optimal α = {alpha_opt:.2e}")
    
    q_linear = linear.solve_l1(u_measured - np.mean(u_measured), alpha=alpha_opt)
    
    # Extract significant sources
    threshold = 0.1 * np.max(np.abs(q_linear))
    significant = np.where(np.abs(q_linear) > threshold)[0]
    print(f"\nRecovered {len(significant)} significant sources (linear)")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Source locations
    utils.plot_sources(sources_true, sources_rec, ax=axes[0, 0],
                      title='Nonlinear Recovery: True (○) vs Recovered (+)')
    
    # Boundary fit
    utils.plot_boundary_data(forward.theta, u_measured, u_rec, ax=axes[0, 1],
                            title='Nonlinear: Boundary Fit')
    
    # Linear recovery heatmap
    ax = axes[1, 0]
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'k-', linewidth=2)
    vmax = max(np.max(np.abs(q_linear)), 0.01)
    scatter = ax.scatter(
        linear.interior_points[:, 0], linear.interior_points[:, 1],
        c=q_linear, cmap='RdBu_r', s=30, vmin=-vmax, vmax=vmax
    )
    plt.colorbar(scatter, ax=ax, label='q')
    for (x, y), q in sources_true:
        ax.plot(x, y, 'ko', markersize=12, markerfacecolor='none', markeredgewidth=2)
    ax.set_aspect('equal')
    ax.set_title('Linear Recovery (L1 regularization)')
    
    # L-curve
    utils.plot_l_curve(alphas, residuals, reg_norms, alpha_opt, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('results/example1_bem_disk.png', dpi=150, bbox_inches='tight')
    print("\nSaved: results/example1_bem_disk.png")
    
    return sources_true, sources_rec


def example_conformal_ellipse():
    """Example 2: Conformal BEM on ellipse."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Conformal BEM on Ellipse")
    print("="*60)
    
    # Create ellipse domain
    ellipse = conformal_bem.EllipseMap(a=2.0, b=1.0)
    print(f"Ellipse with semi-axes a=2.0, b=1.0")
    
    # True sources (inside ellipse)
    sources_true = [
        ((-0.5, 0.3), 1.0),
        ((0.8, 0.2), 1.0),
        ((-0.3, -0.3), -1.0),
        ((0.5, -0.4), -1.0),
    ]
    
    print("\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources_true):
        print(f"  {i+1}: ({x:+.3f}, {y:+.3f}), q = {q:+.2f}")
    
    # Forward solve
    solver = conformal_bem.ConformalBEMSolver(ellipse, n_boundary=100)
    u_measured = solver.solve_forward(sources_true)
    u_measured += 0.001 * np.random.randn(len(u_measured))
    
    # Inverse solve
    print("\nSolving nonlinear inverse problem...")
    inverse = conformal_bem.ConformalNonlinearInverse(ellipse, n_sources=4, n_boundary=100)
    inverse.set_measured_data(u_measured)
    sources_rec, result = inverse.solve(method='L-BFGS-B', maxiter=100)
    
    print("\nRecovered sources:")
    for i, ((x, y), q) in enumerate(sources_rec):
        print(f"  {i+1}: ({x:+.3f}, {y:+.3f}), q = {q:+.3f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    boundary = ellipse.boundary_physical(200)
    ax.plot(np.real(boundary), np.imag(boundary), 'k-', linewidth=2)
    
    for (x, y), q in sources_true:
        color = 'red' if q > 0 else 'blue'
        ax.plot(x, y, 'o', color=color, markersize=15, 
                markerfacecolor='none', markeredgewidth=2)
    
    for (x, y), q in sources_rec:
        color = 'red' if q > 0 else 'blue'
        ax.plot(x, y, '+', color=color, markersize=15, markeredgewidth=3)
    
    ax.set_aspect('equal')
    ax.set_title('Ellipse Domain: True (○) vs Recovered (+)')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    u_rec = solver.solve_forward(sources_rec)
    theta = solver.boundary_param
    ax.plot(theta, u_measured, 'b-', linewidth=2, label='Measured')
    ax.plot(theta, u_rec, 'r--', linewidth=2, label='Recovered')
    ax.set_xlabel('Boundary parameter')
    ax.set_ylabel('u')
    ax.set_title(f'Boundary Fit (residual = {np.linalg.norm(u_rec - u_measured):.4e})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/example2_conformal_ellipse.png', dpi=150, bbox_inches='tight')
    print("\nSaved: results/example2_conformal_ellipse.png")


def example_regularization_comparison():
    """Example 3: Compare regularization methods."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Regularization Method Comparison")
    print("="*60)
    
    # Setup problem
    sources_true = utils.create_test_sources(n_sources=4, seed=42)
    
    forward = bem_solver.BEMForwardSolver(n_boundary_points=100)
    u_measured = forward.solve(sources_true)
    u_measured += 0.001 * np.random.randn(len(u_measured))
    
    linear = bem_solver.BEMLinearInverseSolver(n_boundary=100)
    linear.build_greens_matrix()
    
    # L-curve analysis for multiple methods
    print("\nPerforming L-curve analysis...")
    results = parameter_study.l_curve_analysis(
        linear, u_measured, methods=['l1', 'l2'], verbose=True
    )
    
    # Plot comparison
    fig = parameter_study.plot_l_curve_comparison(results)
    plt.savefig('results/example3_l_curve_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved: results/example3_l_curve_comparison.png")
    
    # Solve with optimal parameters
    print("\nSolving with optimal parameters...")
    u_centered = u_measured - np.mean(u_measured)
    
    q_l1 = linear.solve_l1(u_centered, alpha=results['l1'].alpha_optimal)
    q_l2 = linear.solve_l2(u_centered, alpha=results['l2'].alpha_optimal)
    
    # Compare sparsity
    print(f"\nL1: {np.sum(np.abs(q_l1) > 0.1*np.max(np.abs(q_l1)))} significant sources")
    print(f"L2: {np.sum(np.abs(q_l2) > 0.1*np.max(np.abs(q_l2)))} significant sources")
    
    # Plot solutions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    theta_circle = np.linspace(0, 2*np.pi, 100)
    
    for ax, q, method in zip(axes[:2], [q_l1, q_l2], ['L1', 'L2']):
        ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'k-', linewidth=2)
        vmax = max(np.max(np.abs(q)), 0.01)
        scatter = ax.scatter(
            linear.interior_points[:, 0], linear.interior_points[:, 1],
            c=q, cmap='RdBu_r', s=30, vmin=-vmax, vmax=vmax
        )
        plt.colorbar(scatter, ax=ax, label='q')
        for (x, y), q_true in sources_true:
            ax.plot(x, y, 'ko', markersize=12, markerfacecolor='none', markeredgewidth=2)
        ax.set_aspect('equal')
        ax.set_title(f'{method} Regularization')
    
    # Compare boundary fits
    ax = axes[2]
    theta = forward.theta
    ax.plot(theta, u_centered, 'b-', linewidth=2, label='Measured')
    ax.plot(theta, linear.G @ q_l1, 'r--', linewidth=2, label='L1')
    ax.plot(theta, linear.G @ q_l2, 'g:', linewidth=2, label='L2')
    ax.set_xlabel('θ')
    ax.set_ylabel('u')
    ax.legend()
    ax.set_title('Boundary Fits')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/example3_regularization_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved: results/example3_regularization_comparison.png")


def example_config_usage():
    """Example 4: Configuration system."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Configuration System")
    print("="*60)
    
    # List available templates
    print("\nAvailable configuration templates:")
    for name in config.TEMPLATES.keys():
        print(f"  - {name}")
    
    # Get and modify config
    cfg = config.get_template('default')
    print(f"\nDefault config:")
    print(f"  Forward: {cfg.forward.method}, {cfg.forward.n_boundary_points} boundary points")
    print(f"  Inverse: {cfg.inverse.method}, {cfg.inverse.regularization} regularization")
    
    # Save custom config
    cfg.inverse.alpha = 1e-3
    cfg.grid.n_radial = 15
    cfg.save('results/custom_config.json')
    
    # Load it back
    cfg2 = config.get_config('results/custom_config.json')
    print(f"\nLoaded custom config:")
    print(f"  Alpha: {cfg2.inverse.alpha}")
    print(f"  Grid radial: {cfg2.grid.n_radial}")


def main():
    """Run all examples."""
    print("="*60)
    print("INVERSE SOURCE LOCALIZATION - COMPLETE EXAMPLES")
    print("="*60)
    
    # Create output directory
    Path('results').mkdir(exist_ok=True)
    
    # Run examples
    np.random.seed(42)
    
    example_bem_disk()
    example_conformal_ellipse()
    example_regularization_comparison()
    example_config_usage()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETE")
    print("="*60)
    print("\nOutput files saved to results/")
    
    plt.show()


if __name__ == "__main__":
    main()
