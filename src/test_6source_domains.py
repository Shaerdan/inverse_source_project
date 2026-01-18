"""
6-Source Configuration Test Across Multiple Domains
====================================================

Tests inverse source recovery with n_sources=6 on:
- Disk (analytical solver)
- Ellipse (conformal solver)
- Brain (conformal solver)
- Star (conformal solver)

Uses visualization tools to diagnose any failures.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
from typing import List, Tuple, Dict

# Output directory
OUTPUT_DIR = 'test_6source_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("6-SOURCE INVERSE PROBLEM TEST")
print("=" * 70)
print(f"Output directory: {OUTPUT_DIR}\n")

# Import solvers
from analytical_solver import (
    AnalyticalForwardSolver,
    AnalyticalNonlinearInverseSolver,
    Source
)
from conformal_solver import (
    create_conformal_map,
    ConformalForwardSolver,
    ConformalNonlinearInverseSolver,
    MFSConformalMap
)

# Import visualization tools
from visualization import (
    diagnostic_dashboard,
    plot_convergence,
    plot_source_trajectory,
    plot_source_configuration,
    plot_boundary_fit,
    plot_source_recovery,
    plot_barrier_diagnostics,
    get_domain_boundary,
    compute_source_errors,
    sources_to_arrays,
    COLORS, FIGSIZE
)

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

N_SOURCES = 6
N_BOUNDARY = 100
SEED = 42

# 6 sources that sum to zero (for Neumann compatibility)
# Placed to be challenging but recoverable
SOURCES_TRUE = [
    ((0.35, 0.45), 1.2),    # +
    ((-0.40, 0.35), 0.8),   # +
    ((0.50, -0.20), 1.0),   # +
    ((-0.30, -0.45), -1.1), # -
    ((0.10, -0.50), -0.9),  # -
    ((-0.50, 0.05), -1.0),  # -
]

# Verify sum is zero
total_intensity = sum(s[1] for s in SOURCES_TRUE)
print(f"Source configuration: {N_SOURCES} sources")
print(f"Total intensity (should be ~0): {total_intensity:.6f}")
for i, ((x, y), q) in enumerate(SOURCES_TRUE):
    print(f"  Source {i+1}: ({x:+.2f}, {y:+.2f}), q={q:+.2f}")

# =============================================================================
# DOMAIN CONFIGURATIONS
# =============================================================================

DOMAINS = {
    'disk': {
        'type': 'disk',
        'params': {'radius': 1.0},
        'solver': 'analytical',
        'sources': SOURCES_TRUE,  # Use as-is for disk
    },
    'ellipse': {
        'type': 'ellipse',
        'params': {'a': 1.3, 'b': 0.7},
        'solver': 'conformal',
        # Scale sources to fit ellipse
        'sources': [((x*1.1, y*0.6), q) for (x, y), q in SOURCES_TRUE],
    },
    'brain': {
        'type': 'custom',
        'params': {
            'boundary_func': lambda t: (1.0 + 0.15*np.cos(2*t) - 0.1*np.cos(4*t) + 0.05*np.cos(3*t)) * (1 - 0.1*np.sin(t)**4) * (np.cos(t) + 1j * 0.8 * np.sin(t))
        },
        'solver': 'conformal',
        # Scale sources to fit brain shape
        'sources': [((x*0.7, y*0.5), q) for (x, y), q in SOURCES_TRUE],
    },
    'star': {
        'type': 'custom',
        'params': {
            'boundary_func': lambda t: (0.6 + 0.3*np.cos(5*t)) * np.exp(1j*t)
        },
        'solver': 'conformal',
        # Scale sources to fit inside star (inner radius ~0.3)
        'sources': [((x*0.4, y*0.4), q) for (x, y), q in SOURCES_TRUE],
    },
}

# =============================================================================
# RUN TESTS
# =============================================================================

results = {}

for domain_name, config in DOMAINS.items():
    print("\n" + "=" * 70)
    print(f"TESTING DOMAIN: {domain_name.upper()}")
    print("=" * 70)

    sources_true = config['sources']
    result = {
        'domain': domain_name,
        'sources_true': sources_true,
        'success': False,
        'error_msg': None,
    }

    try:
        # Get domain boundary for visualization
        if config['type'] == 'custom':
            boundary_func = config['params']['boundary_func']
            t = np.linspace(0, 2*np.pi, N_BOUNDARY, endpoint=False)
            z_boundary = np.array([boundary_func(ti) for ti in t])
            domain_boundary = np.column_stack([z_boundary.real, z_boundary.imag])
        else:
            domain_boundary = get_domain_boundary(config['type'], config['params'], n_points=N_BOUNDARY)

        result['domain_boundary'] = domain_boundary

        # Create solvers
        print(f"\n1. Creating {config['solver']} solver...")

        if config['solver'] == 'analytical':
            forward_solver = AnalyticalForwardSolver(n_boundary_points=N_BOUNDARY)
            inverse_solver = AnalyticalNonlinearInverseSolver(
                n_sources=N_SOURCES,
                n_boundary=N_BOUNDARY
            )
            theta = forward_solver.theta
            sensor_locations = forward_solver.sensor_locations

        else:  # conformal
            if config['type'] == 'custom':
                cmap = MFSConformalMap(
                    config['params']['boundary_func'],
                    n_boundary=256,
                    n_charge=200
                )
            else:
                cmap = create_conformal_map(config['type'], **config['params'])

            forward_solver = ConformalForwardSolver(cmap, n_boundary=N_BOUNDARY)
            inverse_solver = ConformalNonlinearInverseSolver(
                cmap,
                n_sources=N_SOURCES,
                n_boundary=N_BOUNDARY
            )

            # Get theta from sensor locations
            sensor_locations = forward_solver.sensor_locations
            theta = np.arctan2(sensor_locations[:, 1], sensor_locations[:, 0])
            theta = np.mod(theta, 2*np.pi)

        result['forward_solver'] = forward_solver
        result['theta'] = theta
        result['sensor_locations'] = sensor_locations

        # Forward solve
        print("2. Running forward solver...")
        u_measured = forward_solver.solve(sources_true)
        result['u_measured'] = u_measured
        print(f"   Boundary data range: [{u_measured.min():.4f}, {u_measured.max():.4f}]")

        # Add noise (realistic measurement noise)
        noise_level = 0.01 * np.std(u_measured)
        np.random.seed(SEED)
        u_noisy = u_measured + noise_level * np.random.randn(len(u_measured))
        result['u_noisy'] = u_noisy
        print(f"   Added noise level: {noise_level:.2e}")

        # Inverse solve
        print("3. Running inverse solver (this may take a while)...")
        t_start = time.time()

        if config['solver'] == 'analytical':
            inverse_result = inverse_solver.solve(
                u_noisy,
                method='differential_evolution',
                seed=SEED
            )
            sources_recovered = [((s.x, s.y), s.intensity) for s in inverse_result.sources]
            history = inverse_result.history if hasattr(inverse_result, 'history') else []

        else:  # conformal
            sources_recovered, residual = inverse_solver.solve(
                u_noisy,
                method='differential_evolution',
                seed=SEED
            )
            history = []  # Conformal solver doesn't track history yet

        t_elapsed = time.time() - t_start
        result['time'] = t_elapsed
        result['sources_recovered'] = sources_recovered
        result['history'] = history

        print(f"   Completed in {t_elapsed:.1f}s")

        # Compute recovered boundary data
        u_recovered = forward_solver.solve(sources_recovered)
        result['u_recovered'] = u_recovered

        # Compute errors
        errors = compute_source_errors(sources_true, sources_recovered)
        result['errors'] = errors
        result['success'] = True

        print(f"\n4. Results:")
        print(f"   Position RMSE: {errors['position_rmse']:.4f}")
        print(f"   Position Max:  {errors['position_max']:.4f}")
        print(f"   Intensity RMSE: {errors['intensity_rmse']:.4f}")
        print(f"   Boundary RMS: {np.sqrt(np.mean((u_noisy - u_recovered)**2)):.2e}")

        # Detailed source comparison
        print(f"\n   Source-by-source comparison:")
        pos_true, int_true = sources_to_arrays(sources_true)
        pos_rec, int_rec = sources_to_arrays(sources_recovered)

        for true_idx, rec_idx, dist in errors['matching']:
            int_err = abs(int_true[true_idx] - int_rec[rec_idx])
            print(f"   True {true_idx+1} -> Rec {rec_idx+1}: "
                  f"pos_err={dist:.4f}, int_err={int_err:.4f}")

    except Exception as e:
        result['error_msg'] = str(e)
        result['success'] = False
        print(f"\n   ERROR: {e}")
        import traceback
        traceback.print_exc()

    results[domain_name] = result

# =============================================================================
# GENERATE VISUALIZATIONS
# =============================================================================

print("\n" + "=" * 70)
print("GENERATING DIAGNOSTIC VISUALIZATIONS")
print("=" * 70)

for domain_name, result in results.items():
    print(f"\nGenerating visualizations for {domain_name}...")

    domain_dir = os.path.join(OUTPUT_DIR, domain_name)
    os.makedirs(domain_dir, exist_ok=True)

    if not result['success']:
        # Create error report
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"FAILED\n\n{result.get('error_msg', 'Unknown error')}",
               ha='center', va='center', fontsize=14, color='red',
               transform=ax.transAxes, wrap=True)
        ax.set_title(f'{domain_name.upper()} - Error Report')
        ax.axis('off')
        fig.savefig(os.path.join(domain_dir, '00_error_report.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved error report")
        continue

    # Extract data
    sources_true = result['sources_true']
    sources_recovered = result['sources_recovered']
    u_measured = result['u_noisy']
    u_recovered = result['u_recovered']
    theta = result['theta']
    domain_boundary = result['domain_boundary']
    history = result['history']
    sensor_locations = result['sensor_locations']
    errors = result['errors']

    # 1. Full diagnostic dashboard
    try:
        fig = diagnostic_dashboard(
            sources_true, sources_recovered,
            u_measured, u_recovered, theta,
            domain_boundary, history, sensor_locations,
            title=f"{domain_name.upper()} Domain - Diagnostic Dashboard"
        )
        fig.savefig(os.path.join(domain_dir, '01_diagnostic_dashboard.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: 01_diagnostic_dashboard.png")
    except Exception as e:
        print(f"  Failed to create dashboard: {e}")

    # 2. Source configuration with matching
    try:
        fig, ax = plt.subplots(figsize=FIGSIZE['square'])
        plot_source_configuration(sources_true, domain_boundary,
                                 sources_recovered=sources_recovered,
                                 sensor_locations=sensor_locations,
                                 show_matching=True,
                                 title=f"{domain_name.upper()} - Source Recovery", ax=ax)
        fig.savefig(os.path.join(domain_dir, '02_source_configuration.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: 02_source_configuration.png")
    except Exception as e:
        print(f"  Failed to create source config: {e}")

    # 3. Detailed source recovery analysis
    try:
        fig = plot_source_recovery(sources_true, sources_recovered, domain_boundary,
                                  title=f"{domain_name.upper()} - Source Recovery Analysis")
        fig.savefig(os.path.join(domain_dir, '03_source_recovery_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: 03_source_recovery_analysis.png")
    except Exception as e:
        print(f"  Failed to create recovery analysis: {e}")

    # 4. Boundary fit
    try:
        fig = plt.figure(figsize=FIGSIZE['wide'])
        plot_boundary_fit(theta, u_measured, u_recovered, show_residual=True,
                         title=f"{domain_name.upper()} - Boundary Fit", fig=fig)
        fig.savefig(os.path.join(domain_dir, '04_boundary_fit.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: 04_boundary_fit.png")
    except Exception as e:
        print(f"  Failed to create boundary fit: {e}")

    # 5. Convergence (if history available)
    if history and len(history) > 1:
        try:
            fig, ax = plt.subplots(figsize=FIGSIZE['wide'])
            plot_convergence(history, title=f"{domain_name.upper()} - Convergence", ax=ax)
            fig.savefig(os.path.join(domain_dir, '05_convergence.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: 05_convergence.png")
        except Exception as e:
            print(f"  Failed to create convergence plot: {e}")

    # 6. Individual source errors
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        pos_errors = errors['position_errors']
        int_errors = errors['intensity_errors']
        x = np.arange(len(pos_errors))

        ax1.bar(x, pos_errors, color=COLORS['trajectory'], alpha=0.8)
        ax1.axhline(errors['position_rmse'], color=COLORS['warning'], linestyle='--',
                   linewidth=2, label=f'RMSE = {errors["position_rmse"]:.4f}')
        ax1.set_xlabel('Source Pair')
        ax1.set_ylabel('Position Error')
        ax1.set_title('Position Errors')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.bar(x, int_errors, color=COLORS['recovered'], alpha=0.8)
        ax2.axhline(errors['intensity_rmse'], color=COLORS['warning'], linestyle='--',
                   linewidth=2, label=f'RMSE = {errors["intensity_rmse"]:.4f}')
        ax2.set_xlabel('Source Pair')
        ax2.set_ylabel('Intensity Error')
        ax2.set_title('Intensity Errors')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f"{domain_name.upper()} - Individual Source Errors", fontsize=14)
        plt.tight_layout()
        fig.savefig(os.path.join(domain_dir, '06_source_errors.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: 06_source_errors.png")
    except Exception as e:
        print(f"  Failed to create error plot: {e}")

# =============================================================================
# SUMMARY COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("GENERATING SUMMARY COMPARISON")
print("=" * 70)

# Create comparison figure
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for col, (domain_name, result) in enumerate(results.items()):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]

    if not result['success']:
        ax_top.text(0.5, 0.5, 'FAILED', ha='center', va='center', fontsize=20, color='red')
        ax_top.set_title(domain_name.upper())
        ax_top.axis('off')
        ax_bot.axis('off')
        continue

    # Top: Source configuration
    domain_boundary = result['domain_boundary']
    sources_true = result['sources_true']
    sources_recovered = result['sources_recovered']

    # Plot boundary
    boundary_closed = np.vstack([domain_boundary, domain_boundary[0:1]])
    ax_top.fill(boundary_closed[:, 0], boundary_closed[:, 1],
               facecolor=COLORS['domain_fill'], alpha=0.3, edgecolor=COLORS['boundary'], linewidth=2)

    # Plot sources
    pos_true, int_true = sources_to_arrays(sources_true)
    pos_rec, int_rec = sources_to_arrays(sources_recovered)

    for pos, intensity in zip(pos_true, int_true):
        color = COLORS['source_positive'] if intensity > 0 else COLORS['source_negative']
        ax_top.scatter(pos[0], pos[1], c=color, s=100, marker='o', edgecolors='white', linewidths=2, zorder=10)

    for pos, intensity in zip(pos_rec, int_rec):
        color = COLORS['source_positive'] if intensity > 0 else COLORS['source_negative']
        ax_top.scatter(pos[0], pos[1], c='none', s=150, marker='s', edgecolors=color, linewidths=2, zorder=9)

    ax_top.set_aspect('equal')
    ax_top.set_title(f"{domain_name.upper()}\nPos RMSE: {result['errors']['position_rmse']:.4f}")
    ax_top.grid(True, alpha=0.3)

    # Bottom: Boundary fit
    theta = result['theta']
    u_meas = result['u_noisy']
    u_rec = result['u_recovered']
    sort_idx = np.argsort(theta)

    ax_bot.plot(theta[sort_idx], u_meas[sort_idx], color=COLORS['measured'], linewidth=1.5, label='Measured')
    ax_bot.plot(theta[sort_idx], u_rec[sort_idx], color=COLORS['recovered'], linewidth=1.5, linestyle='--', label='Recovered')

    rms = np.sqrt(np.mean((u_meas - u_rec)**2))
    ax_bot.set_title(f"RMS: {rms:.2e}")
    ax_bot.set_xlabel('θ')
    ax_bot.grid(True, alpha=0.3)
    if col == 0:
        ax_bot.legend(fontsize=8)

fig.suptitle("6-Source Recovery: Domain Comparison", fontsize=16, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'summary_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: summary_comparison.png")

# =============================================================================
# FINAL REPORT
# =============================================================================

print("\n" + "=" * 70)
print("FINAL REPORT")
print("=" * 70)

print("\n{:<12} {:>12} {:>12} {:>12} {:>10} {:>10}".format(
    "Domain", "Pos RMSE", "Pos Max", "Int RMSE", "Time (s)", "Status"
))
print("-" * 70)

for domain_name, result in results.items():
    if result['success']:
        errors = result['errors']
        status = "OK" if errors['position_rmse'] < 0.1 else "POOR"
        if errors['position_rmse'] > 0.2:
            status = "FAIL"
        print("{:<12} {:>12.4f} {:>12.4f} {:>12.4f} {:>10.1f} {:>10}".format(
            domain_name,
            errors['position_rmse'],
            errors['position_max'],
            errors['intensity_rmse'],
            result['time'],
            status
        ))
    else:
        print("{:<12} {:>12} {:>12} {:>12} {:>10} {:>10}".format(
            domain_name, "N/A", "N/A", "N/A", "N/A", "ERROR"
        ))

print("\nLegend: OK = RMSE < 0.1, POOR = 0.1 <= RMSE < 0.2, FAIL = RMSE >= 0.2")

# Identify failures and diagnoses
print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

for domain_name, result in results.items():
    if not result['success']:
        print(f"\n{domain_name.upper()}: SOLVER FAILED")
        print(f"  Error: {result['error_msg']}")
    elif result['errors']['position_rmse'] > 0.1:
        print(f"\n{domain_name.upper()}: POOR RECOVERY")
        errors = result['errors']

        # Analyze which sources are problematic
        pos_errors = errors['position_errors']
        int_errors = errors['intensity_errors']

        worst_pos_idx = np.argmax(pos_errors)
        worst_int_idx = np.argmax(int_errors)

        print(f"  Worst position error: Source pair {worst_pos_idx+1} (error={pos_errors[worst_pos_idx]:.4f})")
        print(f"  Worst intensity error: Source pair {worst_int_idx+1} (error={int_errors[worst_int_idx]:.4f})")

        # Check if sources are near boundary
        pos_true, _ = sources_to_arrays(result['sources_true'])
        pos_rec, _ = sources_to_arrays(result['sources_recovered'])

        domain_boundary = result['domain_boundary']
        centroid = domain_boundary.mean(axis=0)
        max_dist = np.max(np.linalg.norm(domain_boundary - centroid, axis=1))

        for i, (pt, pr) in enumerate(zip(pos_true, pos_rec)):
            dist_true = np.linalg.norm(pt - centroid) / max_dist
            dist_rec = np.linalg.norm(pr - centroid) / max_dist
            if dist_true > 0.8 or dist_rec > 0.8:
                print(f"  Source {i+1}: near boundary (r_norm={dist_true:.2f}/{dist_rec:.2f})")
    else:
        print(f"\n{domain_name.upper()}: GOOD RECOVERY")
        print(f"  All sources recovered within tolerance")

print(f"\nAll visualizations saved to: {OUTPUT_DIR}/")
