"""
Comprehensive Test for Visualization Toolbox
=============================================

This script generates all visualizations and saves them to test_output/
Run from src/ directory: python visualization/test_all_visualizations.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt

# Create output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'test_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Saving visualizations to: {OUTPUT_DIR}\n")

# Import visualization toolbox
from visualization import *

# Import solvers for realistic tests
try:
    from analytical_solver import AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver
    HAS_ANALYTICAL = True
except ImportError:
    HAS_ANALYTICAL = False
    print("Warning: analytical_solver not found, using mock data")

try:
    from conformal_solver import create_conformal_map, ConformalForwardSolver
    HAS_CONFORMAL = True
except ImportError:
    HAS_CONFORMAL = False
    print("Warning: conformal_solver not found, skipping conformal tests")

# =============================================================================
# TEST DATA SETUP
# =============================================================================

print("=" * 60)
print("Setting up test data...")
print("=" * 60)

# Test sources (must sum to zero for Neumann problem)
sources_true = [
    ((0.4, 0.3), 1.5),
    ((-0.3, 0.5), 1.0),
    ((-0.4, -0.3), -1.2),
    ((0.3, -0.4), -1.3),
]

# Simulated recovered sources (with small errors)
np.random.seed(42)
sources_recovered = [
    ((0.42, 0.28), 1.45),
    ((-0.28, 0.52), 1.05),
    ((-0.38, -0.32), -1.18),
    ((0.32, -0.38), -1.32),
]

n_sources = len(sources_true)
n_boundary = 100

# Domain boundary
boundary_disk = get_domain_boundary('disk', n_points=n_boundary)
boundary_ellipse = get_domain_boundary('ellipse', {'a': 1.5, 'b': 0.8}, n_points=n_boundary)
boundary_star = get_domain_boundary('star', {'n_points': 5, 'r_outer': 1.0, 'r_inner': 0.5}, n_points=n_boundary)

# Boundary angles and simulated data
theta = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)

# Create realistic boundary data using forward solver if available
if HAS_ANALYTICAL:
    fwd_solver = AnalyticalForwardSolver(n_boundary_points=n_boundary)
    u_measured = fwd_solver.solve(sources_true)
    u_recovered = fwd_solver.solve(sources_recovered)
    sensor_locations = fwd_solver.sensor_locations
else:
    # Mock data
    u_measured = np.zeros(n_boundary)
    for (x, y), q in sources_true:
        source_angle = np.arctan2(y, x)
        u_measured += q * np.cos(theta - source_angle) / (2 * np.pi)
    u_measured -= np.mean(u_measured)
    u_recovered = u_measured + 0.02 * np.random.randn(n_boundary)
    sensor_locations = boundary_disk

# Simulated optimization trajectory
n_iterations = 50
trajectory = []
history = []

# Start from random positions, converge toward recovered
np.random.seed(123)
initial_params = np.zeros(3 * n_sources)
for i in range(n_sources):
    initial_params[2*i] = 0.3 * np.random.randn()
    initial_params[2*i + 1] = 0.3 * np.random.randn()
    initial_params[2*n_sources + i] = 0.5 * np.random.randn()

final_params = np.zeros(3 * n_sources)
for i, ((x, y), q) in enumerate(sources_recovered):
    final_params[2*i] = x
    final_params[2*i + 1] = y
    final_params[2*n_sources + i] = q

for t in range(n_iterations):
    alpha = t / (n_iterations - 1)
    params = (1 - alpha) * initial_params + alpha * final_params
    # Add some noise for realism
    params += 0.02 * np.exp(-3*alpha) * np.random.randn(len(params))
    trajectory.append(params.copy())
    # Simulated objective (exponential decay with noise)
    obj = 10 * np.exp(-5 * alpha) + 0.001 * np.random.rand()
    history.append(obj)

# Grid for linear solver visualization
n_grid = 200
grid_r = np.random.uniform(0.1, 0.9, n_grid)
grid_theta = np.random.uniform(0, 2*np.pi, n_grid)
grid_positions = np.column_stack([grid_r * np.cos(grid_theta), grid_r * np.sin(grid_theta)])

# Simulated linear solver output (sparse with peaks near true sources)
grid_intensities = np.zeros(n_grid)
for (x, y), q in sources_true:
    dist = np.sqrt((grid_positions[:, 0] - x)**2 + (grid_positions[:, 1] - y)**2)
    grid_intensities += q * np.exp(-dist**2 / 0.05)
grid_intensities += 0.1 * np.random.randn(n_grid)

print(f"  Sources: {n_sources}")
print(f"  Boundary points: {n_boundary}")
print(f"  Trajectory iterations: {n_iterations}")
print(f"  Grid points: {n_grid}")

# =============================================================================
# TEST 1: CONFIG MODULE
# =============================================================================
print("\n" + "=" * 60)
print("1. Testing config module...")
print("=" * 60)

fig, ax = plt.subplots(figsize=FIGSIZE['single'])
colors_to_show = ['source_positive', 'source_negative', 'measured', 'recovered',
                  'trajectory', 'boundary', 'sensor']
for i, name in enumerate(colors_to_show):
    ax.barh(i, 1, color=COLORS[name], label=name)
ax.set_yticks(range(len(colors_to_show)))
ax.set_yticklabels(colors_to_show)
ax.set_title('Color Palette')
ax.set_xlim(0, 1.5)
fig.savefig(os.path.join(OUTPUT_DIR, '01_config_colors.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_config_colors.png")

# =============================================================================
# TEST 2: UTILS MODULE
# =============================================================================
print("\n" + "=" * 60)
print("2. Testing utils module...")
print("=" * 60)

# Test domain boundaries
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
domains = ['disk', 'ellipse', 'square', 'rectangle', 'star', 'brain']
params_list = [
    {},
    {'a': 2, 'b': 1},
    {'half_side': 1},
    {'half_width': 1.5, 'half_height': 0.8},
    {'n_points': 5, 'r_outer': 1.0, 'r_inner': 0.5},
    {'scale': 1.0}
]

for ax, domain, params in zip(axes.flat, domains, params_list):
    boundary = get_domain_boundary(domain, params)
    add_domain_boundary(ax, boundary, fill=True)
    ax.set_aspect('equal')
    ax.set_title(f'{domain.capitalize()} Domain')
    ax.grid(True, alpha=0.3)

fig.suptitle('Domain Boundaries (utils.get_domain_boundary)', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '02_utils_domains.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_utils_domains.png")

# Test source matching
errors = compute_source_errors(sources_true, sources_recovered)
print(f"  Source matching - Position RMSE: {errors['position_rmse']:.4f}")
print(f"  Source matching - Intensity RMSE: {errors['intensity_rmse']:.4f}")

# =============================================================================
# TEST 3: FORWARD_VIZ MODULE
# =============================================================================
print("\n" + "=" * 60)
print("3. Testing forward_viz module...")
print("=" * 60)

# plot_boundary_values
fig, ax = plt.subplots(figsize=FIGSIZE['wide'])
plot_boundary_values(theta, u_measured, sources=sources_true,
                    title="Boundary Potential u(θ)", ax=ax)
fig.savefig(os.path.join(OUTPUT_DIR, '03a_forward_boundary_values.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03a_forward_boundary_values.png")

# plot_source_configuration
fig, ax = plt.subplots(figsize=FIGSIZE['square'])
plot_source_configuration(sources_true, boundary_disk,
                         sources_recovered=sources_recovered,
                         sensor_locations=sensor_locations,
                         title="Source Configuration", ax=ax)
fig.savefig(os.path.join(OUTPUT_DIR, '03b_forward_source_config.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03b_forward_source_config.png")

# plot_boundary_comparison
u_list = [u_measured, u_recovered, u_measured + 0.05*np.sin(3*theta)]
labels = ['Measured', 'Recovered', 'Alternative']
fig, ax = plt.subplots(figsize=FIGSIZE['wide'])
plot_boundary_comparison(theta, u_list, labels, ax=ax)
fig.savefig(os.path.join(OUTPUT_DIR, '03c_forward_boundary_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03c_forward_boundary_comparison.png")

# plot_interior_solution (if solver available)
if HAS_ANALYTICAL:
    fig, ax = plt.subplots(figsize=FIGSIZE['square'])
    plot_interior_solution(fwd_solver, sources_true, 'disk', resolution=40,
                          title="Interior Solution", ax=ax)
    fig.savefig(os.path.join(OUTPUT_DIR, '03d_forward_interior.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 03d_forward_interior.png")

# =============================================================================
# TEST 4: OPTIMIZATION_VIZ MODULE
# =============================================================================
print("\n" + "=" * 60)
print("4. Testing optimization_viz module...")
print("=" * 60)

# plot_convergence
fig, ax = plt.subplots(figsize=FIGSIZE['wide'])
plot_convergence(history, true_minimum=0.001, milestones=[10, 30], ax=ax)
fig.savefig(os.path.join(OUTPUT_DIR, '04a_opt_convergence.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04a_opt_convergence.png")

# plot_multistart_convergence
histories = [
    [10 * np.exp(-5 * t/50) + 0.001 * np.random.rand() for t in range(50)],
    [12 * np.exp(-4 * t/50) + 0.002 * np.random.rand() for t in range(50)],
    [8 * np.exp(-6 * t/50) + 0.0005 * np.random.rand() for t in range(50)],
]
fig, ax = plt.subplots(figsize=FIGSIZE['wide'])
plot_multistart_convergence(histories, labels=['Run 1', 'Run 2', 'Run 3'], ax=ax)
fig.savefig(os.path.join(OUTPUT_DIR, '04b_opt_multistart.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04b_opt_multistart.png")

# plot_source_trajectory
fig, ax = plt.subplots(figsize=FIGSIZE['square'])
plot_source_trajectory(trajectory, n_sources, boundary_disk,
                      sources_true=sources_true, ax=ax)
fig.savefig(os.path.join(OUTPUT_DIR, '04c_opt_trajectory.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04c_opt_trajectory.png")

# plot_parameter_evolution - positions
fig = plot_parameter_evolution(trajectory, n_sources, sources_true=sources_true,
                              param_type='positions')
fig.savefig(os.path.join(OUTPUT_DIR, '04d_opt_position_evolution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04d_opt_position_evolution.png")

# plot_parameter_evolution - intensities
fig = plot_parameter_evolution(trajectory, n_sources, sources_true=sources_true,
                              param_type='intensities')
fig.savefig(os.path.join(OUTPUT_DIR, '04e_opt_intensity_evolution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04e_opt_intensity_evolution.png")

# plot_optimization_summary
fig = plot_optimization_summary(trajectory, history, n_sources, boundary_disk,
                               sources_true=sources_true)
fig.savefig(os.path.join(OUTPUT_DIR, '04f_opt_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04f_opt_summary.png")

# =============================================================================
# TEST 5: BARRIER_VIZ MODULE
# =============================================================================
print("\n" + "=" * 60)
print("5. Testing barrier_viz module...")
print("=" * 60)

# plot_barrier_landscape - disk
fig, ax = plt.subplots(figsize=FIGSIZE['square'])
plot_barrier_landscape('disk', mu=1e-4, R_max=0.95, ax=ax)
fig.savefig(os.path.join(OUTPUT_DIR, '05a_barrier_landscape_disk.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05a_barrier_landscape_disk.png")

# plot_barrier_landscape - ellipse
fig, ax = plt.subplots(figsize=FIGSIZE['square'])
plot_barrier_landscape('ellipse', {'a': 1.5, 'b': 0.8}, mu=1e-4, R_max=0.9, ax=ax)
fig.savefig(os.path.join(OUTPUT_DIR, '05b_barrier_landscape_ellipse.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05b_barrier_landscape_ellipse.png")

# plot_barrier_radial_profile
fig, ax = plt.subplots(figsize=FIGSIZE['wide'])
plot_barrier_radial_profile(mu_values=[1e-3, 1e-4, 1e-5], R_max=0.95, ax=ax)
fig.savefig(os.path.join(OUTPUT_DIR, '05c_barrier_radial.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05c_barrier_radial.png")

# plot_constraint_violation
fig, ax = plt.subplots(figsize=FIGSIZE['wide'])
plot_constraint_violation(trajectory, n_sources, 'disk', R_max=0.95, ax=ax)
fig.savefig(os.path.join(OUTPUT_DIR, '05d_barrier_violation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05d_barrier_violation.png")

# plot_barrier_diagnostics
fig = plot_barrier_diagnostics(trajectory, history, n_sources, 'disk',
                              mu=1e-4, R_max=0.95)
fig.savefig(os.path.join(OUTPUT_DIR, '05e_barrier_diagnostics.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05e_barrier_diagnostics.png")

# =============================================================================
# TEST 6: INVERSE_VIZ MODULE
# =============================================================================
print("\n" + "=" * 60)
print("6. Testing inverse_viz module...")
print("=" * 60)

# plot_boundary_fit
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE['wide'], height_ratios=[3, 1])
plot_boundary_fit(theta, u_measured, u_recovered, show_residual=True, fig=fig)
fig.savefig(os.path.join(OUTPUT_DIR, '06a_inverse_boundary_fit.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 06a_inverse_boundary_fit.png")

# plot_source_recovery
fig = plot_source_recovery(sources_true, sources_recovered, boundary_disk)
fig.savefig(os.path.join(OUTPUT_DIR, '06b_inverse_source_recovery.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 06b_inverse_source_recovery.png")

# plot_linear_solution
fig, ax = plt.subplots(figsize=FIGSIZE['square'])
plot_linear_solution(grid_positions, grid_intensities, boundary_disk,
                    sources_true=sources_true, ax=ax)
fig.savefig(os.path.join(OUTPUT_DIR, '06c_inverse_linear_solution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 06c_inverse_linear_solution.png")

# plot_source_intensity_bar
fig, ax = plt.subplots(figsize=FIGSIZE['wide'])
plot_source_intensity_bar(sources_true, sources_recovered, ax=ax)
fig.savefig(os.path.join(OUTPUT_DIR, '06d_inverse_intensity_bar.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 06d_inverse_intensity_bar.png")

# =============================================================================
# TEST 7: DASHBOARD MODULE
# =============================================================================
print("\n" + "=" * 60)
print("7. Testing dashboard module...")
print("=" * 60)

# diagnostic_dashboard
fig = diagnostic_dashboard(sources_true, sources_recovered, u_measured, u_recovered,
                          theta, boundary_disk, history, sensor_locations)
fig.savefig(os.path.join(OUTPUT_DIR, '07a_dashboard_diagnostic.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 07a_dashboard_diagnostic.png")

# solver_comparison_dashboard
results = [
    {'name': 'L-BFGS-B', 'sources': sources_recovered, 'u_recovered': u_recovered,
     'history': history, 'time': 2.5},
    {'name': 'Diff. Evol.', 'sources': [((s[0][0]+0.02, s[0][1]-0.01), s[1]*0.98) for s in sources_recovered],
     'u_recovered': u_recovered + 0.01*np.random.randn(n_boundary),
     'history': [h*1.2 for h in history], 'time': 15.3},
]
fig = solver_comparison_dashboard(results, sources_true, u_measured, theta, boundary_disk)
fig.savefig(os.path.join(OUTPUT_DIR, '07b_dashboard_solver_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 07b_dashboard_solver_comparison.png")

# quick_diagnostic (using mock inverse result)
class MockInverseResult:
    def __init__(self):
        self.sources = [type('Source', (), {'x': s[0][0], 'y': s[0][1], 'intensity': s[1]})()
                       for s in sources_recovered]
        self.history = history

if HAS_ANALYTICAL:
    mock_result = MockInverseResult()
    fig = quick_diagnostic(fwd_solver, mock_result, sources_true, 'disk')
    fig.savefig(os.path.join(OUTPUT_DIR, '07c_dashboard_quick.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 07c_dashboard_quick.png")

# =============================================================================
# TEST 8: GREENS_VIZ MODULE
# =============================================================================
print("\n" + "=" * 60)
print("8. Testing greens_viz module...")
print("=" * 60)

# plot_greens_function
fig, ax = plt.subplots(figsize=FIGSIZE['square'])
plot_greens_function((0.3, 0.4), 'disk', resolution=60, ax=ax)
fig.savefig(os.path.join(OUTPUT_DIR, '08a_greens_function.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 08a_greens_function.png")

# plot_greens_boundary
fig, ax = plt.subplots(figsize=FIGSIZE['wide'])
plot_greens_boundary((0.3, 0.4), 'disk', ax=ax)
fig.savefig(os.path.join(OUTPUT_DIR, '08b_greens_boundary.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 08b_greens_boundary.png")

# plot_greens_matrix (create mock matrix)
n_sensors_g = 50
n_grid_g = 100
G_mock = np.random.randn(n_sensors_g, n_grid_g)
# Add some structure
for i in range(n_sensors_g):
    for j in range(n_grid_g):
        G_mock[i, j] += np.exp(-((i/n_sensors_g - j/n_grid_g)**2) * 10)

fig = plot_greens_matrix(G_mock)
fig.savefig(os.path.join(OUTPUT_DIR, '08c_greens_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 08c_greens_matrix.png")

# plot_greens_function_comparison
fig = plot_greens_function_comparison([(0.3, 0.0), (0.0, 0.5), (-0.4, -0.3)], 'disk')
fig.savefig(os.path.join(OUTPUT_DIR, '08d_greens_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 08d_greens_comparison.png")

# =============================================================================
# TEST 9: CONFORMAL_VIZ MODULE
# =============================================================================
print("\n" + "=" * 60)
print("9. Testing conformal_viz module...")
print("=" * 60)

if HAS_CONFORMAL:
    # Create conformal map for ellipse
    cmap = create_conformal_map('ellipse', a=1.5, b=0.8)

    # plot_domain_correspondence
    fig = plot_domain_correspondence(cmap, n_circles=4, n_radials=8)
    fig.savefig(os.path.join(OUTPUT_DIR, '09a_conformal_correspondence.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 09a_conformal_correspondence.png")

    # plot_mapping_jacobian
    fig, ax = plt.subplots(figsize=FIGSIZE['square'])
    plot_mapping_jacobian(cmap, resolution=40, ax=ax)
    fig.savefig(os.path.join(OUTPUT_DIR, '09b_conformal_jacobian.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 09b_conformal_jacobian.png")

    # plot_boundary_correspondence
    fig, ax = plt.subplots(figsize=FIGSIZE['square'])
    plot_boundary_correspondence(cmap, ax=ax)
    fig.savefig(os.path.join(OUTPUT_DIR, '09c_conformal_boundary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 09c_conformal_boundary.png")

    # plot_conformal_distortion
    fig = plot_conformal_distortion(cmap, resolution=25)
    fig.savefig(os.path.join(OUTPUT_DIR, '09d_conformal_distortion.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 09d_conformal_distortion.png")

    # plot_source_mapping
    sources_ellipse = [((0.5, 0.2), 1.0), ((-0.6, 0.1), -0.5), ((0.0, -0.4), -0.5)]
    fig = plot_source_mapping(cmap, sources_ellipse)
    fig.savefig(os.path.join(OUTPUT_DIR, '09e_conformal_source_mapping.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 09e_conformal_source_mapping.png")
else:
    print("  Skipped conformal tests (conformal_solver not available)")

# =============================================================================
# TEST 10: ANIMATION MODULE (save as static frames)
# =============================================================================
print("\n" + "=" * 60)
print("10. Testing animation module (static frames)...")
print("=" * 60)

# Create a few frames from animation manually
for frame_idx in [0, n_iterations//2, n_iterations-1]:
    fig, ax = plt.subplots(figsize=FIGSIZE['square'])
    add_domain_boundary(ax, boundary_disk, fill=True)

    params = trajectory[frame_idx]
    for i in range(n_sources):
        x, y = params[2*i], params[2*i + 1]
        q = params[2*n_sources + i]
        color = get_source_color(q)
        ax.scatter(x, y, c=color, s=100, edgecolors='white', linewidths=2, zorder=5)

        # Trail
        trail_start = max(0, frame_idx - 10)
        trail_x = [trajectory[f][2*i] for f in range(trail_start, frame_idx + 1)]
        trail_y = [trajectory[f][2*i + 1] for f in range(trail_start, frame_idx + 1)]
        ax.plot(trail_x, trail_y, '-', color=color, alpha=0.5, linewidth=1)

    # True sources
    for (x, y), q in sources_true:
        ax.scatter(x, y, c='none', s=150, marker='*', edgecolors='gold', linewidths=2, zorder=10)

    ax.set_aspect('equal')
    ax.set_title(f'Animation Frame {frame_idx} (Iteration {frame_idx})')
    fig.savefig(os.path.join(OUTPUT_DIR, f'10_animation_frame_{frame_idx:03d}.png'), dpi=150, bbox_inches='tight')
    plt.close()

print(f"  Saved: 10_animation_frame_000.png")
print(f"  Saved: 10_animation_frame_{n_iterations//2:03d}.png")
print(f"  Saved: 10_animation_frame_{n_iterations-1:03d}.png")

# Try to create actual animation (GIF)
try:
    anim = animate_source_movement(trajectory[:20], n_sources, boundary_disk,
                                   sources_true=sources_true,
                                   interval=200,
                                   save_path=os.path.join(OUTPUT_DIR, '10_animation.gif'))
    plt.close()
    print("  Saved: 10_animation.gif")
except Exception as e:
    print(f"  Animation GIF failed (may need pillow): {e}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("TEST COMPLETE!")
print("=" * 60)

# Count saved files
files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png') or f.endswith('.gif')]
print(f"\nTotal visualizations saved: {len(files)}")
print(f"Output directory: {OUTPUT_DIR}")
print("\nFiles created:")
for f in sorted(files):
    print(f"  - {f}")
