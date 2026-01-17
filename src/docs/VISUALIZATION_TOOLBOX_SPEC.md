# Inverse Source Visualization Toolbox — Implementation Specification

## Overview

Create a comprehensive visualization toolbox for debugging and understanding inverse source localization solvers. This module should be added to the existing `inverse_source_project` package.

## Directory Structure

```
src/visualization/
├── __init__.py           # Exports all public functions
├── config.py             # Colors, markers, figure sizes, style settings
├── utils.py              # Helper functions (domain geometry, source matching, etc.)
├── forward_viz.py        # Forward problem visualizations
├── inverse_viz.py        # Inverse problem visualizations  
├── optimization_viz.py   # Optimization diagnostics
├── barrier_viz.py        # Constraint/barrier visualizations
├── conformal_viz.py      # Conformal mapping visualizations
├── greens_viz.py         # Green's function visualizations
├── animation.py          # Animated visualizations
└── dashboard.py          # Combined diagnostic dashboards
```

---

## 1. config.py — Centralized Configuration

### Purpose
Single source of truth for all styling to ensure visual consistency.

### Contents

```python
COLORS = {
    # Sources
    'source_positive': '#d62728',      # Red for q > 0
    'source_negative': '#1f77b4',      # Blue for q < 0
    
    # Domain
    'boundary': '#2c3e50',             # Dark gray
    'domain_fill': '#ecf0f1',          # Light fill
    
    # Data comparison
    'measured': '#3498db',             # Blue
    'recovered': '#e74c3c',            # Red
    'residual': '#f39c12',             # Orange
    
    # Optimization
    'trajectory': '#8e44ad',           # Purple
    'initial': '#27ae60',              # Green
    'final': '#c0392b',                # Dark red
    
    # Sensors
    'sensor': '#7f8c8d',               # Gray
}

MARKERS = {
    'source_true': {'marker': 'o', 'markersize': 10, 'markeredgewidth': 2},
    'source_recovered': {'marker': 's', 'markersize': 10, 'fillstyle': 'none', 'markeredgewidth': 2},
    'sensor': {'marker': '.', 'markersize': 4},
}

FIGSIZE = {
    'single': (8, 6),
    'wide': (12, 5),
    'square': (8, 8),
    'dashboard': (16, 12),
}

COLORMAPS = {
    'solution': 'RdBu_r',      # Diverging for potential
    'barrier': 'YlOrRd',       # Sequential for barrier
    'error': 'hot',            # Sequential for errors
    'greens': 'PiYG',          # Diverging for Green's function
}

def get_source_color(intensity: float) -> str:
    """Return red for positive, blue for negative intensity."""
    return COLORS['source_positive'] if intensity > 0 else COLORS['source_negative']

def apply_style():
    """Apply consistent matplotlib style settings."""
    # Set rcParams for font sizes, grid, etc.
```

---

## 2. utils.py — Utility Functions

### Domain Geometry

```python
def get_domain_boundary(domain_type: str, domain_params: dict = None, 
                        n_points: int = 200) -> np.ndarray:
    """
    Get boundary points for domain.
    
    Supports: 'disk', 'ellipse', 'square', 'star', 'polygon', 'brain'
    
    Returns: ndarray shape (n_points, 2)
    """

def domain_mask(xx: np.ndarray, yy: np.ndarray, 
                domain_type: str, domain_params: dict) -> np.ndarray:
    """
    Create boolean mask for points inside domain.
    
    Used for masking heatmaps to domain interior.
    """

def point_in_domain(x: float, y: float, domain_type: str, 
                    domain_params: dict) -> bool:
    """Check if single point is inside domain."""

def get_bounding_box(domain_boundary: np.ndarray, 
                     margin: float = 0.1) -> Tuple[float, float, float, float]:
    """Get (x_min, x_max, y_min, y_max) with margin."""
```

### Source Utilities

```python
def match_sources(sources_true: List[Tuple], 
                  sources_recovered: List[Tuple]) -> List[Tuple[int, int, float]]:
    """
    Hungarian algorithm matching between true and recovered sources.
    
    Returns: list of (true_idx, recovered_idx, position_distance)
    """

def sources_to_arrays(sources: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert sources list to (positions, intensities) arrays."""

def compute_source_errors(sources_true, sources_recovered) -> Dict:
    """
    Compute error metrics after matching.
    
    Returns: {
        'position_rmse': float,
        'position_max': float,
        'intensity_rmse': float,
        'matching': list of (i, j, dist),
    }
    """
```

### Plotting Helpers

```python
def add_domain_boundary(ax: plt.Axes, boundary: np.ndarray, 
                        fill: bool = False, **kwargs):
    """Add domain boundary line (and optional fill) to axes."""

def add_source_markers(ax: plt.Axes, sources: List[Tuple], 
                       style: str = 'true', show_labels: bool = False):
    """
    Add source markers to axes.
    
    style: 'true' (filled circles) or 'recovered' (hollow squares)
    """

def add_sensor_markers(ax: plt.Axes, sensor_locations: np.ndarray):
    """Add small gray dots for sensor locations."""

def add_colorbar(ax, mappable, label: str = None):
    """Add colorbar with label."""

def set_domain_axes(ax, domain_boundary: np.ndarray):
    """Set equal aspect and appropriate limits."""
```

---

## 3. forward_viz.py — Forward Problem Visualizations

### 3.1 plot_boundary_values

```python
def plot_boundary_values(theta: np.ndarray, 
                         u_boundary: np.ndarray,
                         sources: List[Tuple] = None,
                         u_reference: np.ndarray = None,
                         title: str = "Boundary Potential u(θ)",
                         ax: plt.Axes = None) -> plt.Axes:
    """
    Plot boundary potential u(θ) vs θ.
    
    Features:
    - Main curve: u(θ) in blue
    - Reference curve (if provided): dashed gray, with shaded difference
    - Source angle markers: vertical lines at arctan2(y,x) for each source
      colored by sign of intensity
    - θ axis from 0 to 2π with π markers
    
    Use case: Verify forward solution looks correct, compare measured vs recovered.
    """
```

### 3.2 plot_interior_solution

```python
def plot_interior_solution(forward_solver,
                           sources: List[Tuple],
                           domain_type: str,
                           domain_params: dict = None,
                           resolution: int = 50,
                           show_sources: bool = True,
                           ax: plt.Axes = None) -> plt.Axes:
    """
    Heatmap of solution u(x,y) on domain interior.
    
    Implementation:
    1. Create meshgrid covering bounding box
    2. Mask points outside domain
    3. Evaluate forward_solver.solve_interior(sources, points) or equivalent
    4. Plot filled contours with RdBu_r colormap (symmetric around 0)
    5. Overlay domain boundary
    6. Mark sources with +/- markers, size proportional to |q|
    
    Use case: Visualize potential field, verify source singularities are correct.
    """
```

### 3.3 plot_source_configuration

```python
def plot_source_configuration(sources: List[Tuple],
                              domain_boundary: np.ndarray,
                              sources_recovered: List[Tuple] = None,
                              sensor_locations: np.ndarray = None,
                              show_matching: bool = True,
                              title: str = "Source Configuration",
                              ax: plt.Axes = None) -> plt.Axes:
    """
    Plot sources on domain.
    
    Features:
    - Domain boundary (solid black) with light fill
    - True sources: filled circles, red (+) / blue (-)
    - Recovered sources (if provided): hollow squares
    - Matching arrows from recovered to true (if show_matching)
    - Sensors on boundary (small gray dots)
    - Legend with source intensities
    
    Use case: Quick visual check of source positions and recovery quality.
    """
```

---

## 4. inverse_viz.py — Inverse Problem Visualizations

### 4.1 plot_boundary_fit

```python
def plot_boundary_fit(theta: np.ndarray,
                      u_measured: np.ndarray,
                      u_recovered: np.ndarray,
                      show_residual: bool = True,
                      ax: plt.Axes = None) -> Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]:
    """
    Compare measured vs recovered boundary data.
    
    If show_residual=True, creates 2-panel figure:
    
    Top panel: Overlay
    - Measured: solid blue with dots
    - Recovered: dashed red
    - Shaded region showing difference
    
    Bottom panel: Residual
    - Residual = u_measured - u_recovered
    - Horizontal zero line
    - RMS annotation
    
    Use case: Assess fit quality, identify problematic boundary regions.
    """
```

### 4.2 plot_source_recovery

```python
def plot_source_recovery(sources_true: List[Tuple],
                         sources_recovered: List[Tuple],
                         domain_boundary: np.ndarray) -> plt.Figure:
    """
    Comprehensive source recovery comparison (4-panel figure).
    
    Panel 1 (top-left): Spatial comparison
    - Domain with true (filled) and recovered (hollow) sources
    - Arrows from recovered to true
    - Arrow color intensity = error magnitude
    
    Panel 2 (top-right): Position error vs radius
    - x: true source radius (distance from origin)
    - y: position error
    - Reveals if errors correlate with source location
    
    Panel 3 (bottom-left): Intensity scatter
    - x: true intensity, y: recovered intensity
    - Diagonal line = perfect recovery
    - Annotate correlation coefficient
    
    Panel 4 (bottom-right): Error summary table
    - Table with: source idx, true pos, recovered pos, pos error, int error
    - Summary row with RMSE values
    
    Use case: Detailed diagnosis of source recovery accuracy.
    """
```

### 4.3 plot_linear_solution

```python
def plot_linear_solution(grid_positions: np.ndarray,
                         grid_intensities: np.ndarray,
                         domain_boundary: np.ndarray,
                         sources_true: List[Tuple] = None,
                         threshold: float = None,
                         ax: plt.Axes = None) -> plt.Axes:
    """
    Visualize grid-based linear solver output.
    
    Features:
    - Scatter plot of grid points, color = intensity (RdBu_r)
    - Size proportional to |intensity|
    - True sources marked with stars (if provided)
    - Detected peaks circled (points above threshold)
    - Domain boundary overlay
    - Sparsity annotation (fraction of points above threshold)
    
    Use case: Understand linear solver output, see where it places mass.
    """
```

---

## 5. optimization_viz.py — Optimization Diagnostics

### 5.1 plot_convergence

```python
def plot_convergence(history: List[float],
                     true_minimum: float = None,
                     log_scale: bool = True,
                     milestones: List[int] = None,
                     ax: plt.Axes = None) -> plt.Axes:
    """
    Plot objective value vs iteration.
    
    Features:
    - Main line: objective value (log scale by default)
    - True minimum: horizontal dashed line (if known)
    - Milestones: vertical dotted lines (e.g., restart points)
    - Final value annotation
    - Convergence rate annotation (slope of last N iterations)
    
    Use case: Diagnose convergence behavior, detect stuck optimization.
    """
```

### 5.2 plot_multistart_convergence

```python
def plot_multistart_convergence(histories: List[List[float]],
                                 labels: List[str] = None,
                                 highlight_best: bool = True,
                                 ax: plt.Axes = None) -> plt.Axes:
    """
    Compare convergence across multiple restarts.
    
    Features:
    - Each run: thin colored line
    - Best run: thick highlighted line
    - Final values: scatter points
    - Inset or annotation: best/worst/median final values
    
    Use case: Assess sensitivity to initialization.
    """
```

### 5.3 plot_source_trajectory

```python
def plot_source_trajectory(trajectory: List[np.ndarray],
                           n_sources: int,
                           domain_boundary: np.ndarray,
                           sources_true: List[Tuple] = None,
                           source_indices: List[int] = None,
                           ax: plt.Axes = None) -> plt.Axes:
    """
    Plot source position trajectories during optimization.
    
    Parameters:
    - trajectory: list of parameter vectors at each iteration
      Layout: [x0, y0, x1, y1, ..., q0, q1, ...]
    - source_indices: which sources to show (default: all)
    
    Features:
    - Domain boundary
    - True source positions (hollow markers)
    - Trajectory lines colored by iteration (early=light, late=dark)
    - Initial positions: green circles
    - Final positions: red circles
    - Arrows showing direction of movement
    
    Use case: Understand optimization dynamics, see which sources move.
    """
```

### 5.4 plot_parameter_evolution

```python
def plot_parameter_evolution(trajectory: List[np.ndarray],
                             n_sources: int,
                             sources_true: List[Tuple] = None,
                             param_type: str = 'positions') -> plt.Figure:
    """
    Plot individual parameter values vs iteration.
    
    param_type: 'positions' or 'intensities'
    
    For positions: 2*n subplots showing x_i(iter) and y_i(iter)
    For intensities: n subplots showing q_i(iter)
    
    True values shown as horizontal dashed lines.
    
    Use case: See which parameters converge first, detect stuck parameters.
    """
```

---

## 6. barrier_viz.py — Barrier/Constraint Visualizations

### 6.1 plot_barrier_landscape

```python
def plot_barrier_landscape(domain_type: str,
                           domain_params: dict = None,
                           mu: float = 1e-4,
                           R_max: float = 0.95,
                           resolution: int = 100,
                           ax: plt.Axes = None) -> plt.Axes:
    """
    Visualize log-barrier function over domain.
    
    For disk: barrier = -μ * log(R_max² - r²)  for r < R_max
              barrier = penalty(r - R_max)      for r >= R_max
    
    Features:
    - Heatmap of barrier value (log scale colormap)
    - Domain boundary at R_max (dashed)
    - True boundary at R=1 (solid)
    - Contour lines at key barrier levels
    - Outside region shown with penalty colormap
    - Annotation: μ value, barrier at r=0, barrier at r=0.9
    
    Use case: Understand constraint strength, tune μ parameter.
    """
```

### 6.2 plot_objective_slice

```python
def plot_objective_slice(objective_fn: Callable,
                         base_params: np.ndarray,
                         vary_indices: Tuple[int, int],
                         ranges: Tuple[Tuple[float, float], Tuple[float, float]],
                         resolution: int = 50,
                         true_values: Tuple[float, float] = None,
                         current_values: Tuple[float, float] = None,
                         log_scale: bool = True,
                         ax: plt.Axes = None) -> plt.Axes:
    """
    2D slice of objective function.
    
    Fix all parameters at base_params except two (vary_indices).
    Vary those two over given ranges.
    
    Features:
    - Filled contours of objective (log scale optional)
    - True parameter values: star marker
    - Current values: circle marker
    - Domain boundary (if slice is through positions)
    - Global minimum marker (if detectable from slice)
    
    Example vary_indices:
    - (0, 1): x₀, y₀ of first source
    - (12, 13): q₀, q₁ intensities for 6-source problem
    
    Use case: Visualize optimization landscape, diagnose local minima.
    """
```

### 6.3 plot_barrier_radial_profile

```python
def plot_barrier_radial_profile(mu_values: List[float] = [1e-3, 1e-4, 1e-5],
                                 R_max: float = 0.95,
                                 ax: plt.Axes = None) -> plt.Axes:
    """
    Plot barrier value vs radius for different μ values.
    
    Features:
    - Multiple lines, one per μ
    - X-axis: r from 0 to 1
    - Y-axis: barrier value (log scale)
    - Vertical line at R_max
    - Legend with μ values
    - Annotation explaining μ effect
    
    Use case: Choose appropriate μ value.
    """
```

---

## 7. conformal_viz.py — Conformal Mapping Visualizations

### 7.1 plot_domain_correspondence

```python
def plot_domain_correspondence(conformal_map,
                                n_circles: int = 5,
                                n_radials: int = 8,
                                n_boundary: int = 100) -> plt.Figure:
    """
    Show mapping between canonical (unit disk) and physical domain.
    
    Creates 2-panel figure:
    
    Left: Canonical domain (unit disk)
    - Concentric circles at r = 0.2, 0.4, 0.6, 0.8
    - Radial lines at θ = 0, π/4, π/2, ...
    
    Right: Physical domain
    - Images of circles under the map
    - Images of radial lines
    - Should remain orthogonal for conformal map
    
    Color coding: same colors for corresponding curves
    
    Use case: Verify conformal map quality, understand distortion.
    """
```

### 7.2 plot_mapping_jacobian

```python
def plot_mapping_jacobian(conformal_map,
                          resolution: int = 50,
                          ax: plt.Axes = None) -> plt.Axes:
    """
    Heatmap of |f'(z)| (Jacobian/scale factor) over canonical domain.
    
    Features:
    - Heatmap of |f'(z)| over unit disk
    - Contour lines
    - Colorbar with label "Local scale factor"
    - Physical domain boundary (inset)
    
    Interpretation:
    - |f'(z)| > 1: local expansion
    - |f'(z)| < 1: local contraction
    - Constant: uniform scaling
    
    Use case: Understand where mapping distorts most.
    """
```

### 7.3 plot_boundary_correspondence

```python
def plot_boundary_correspondence(conformal_map,
                                  sensor_locations_physical: np.ndarray = None,
                                  ax: plt.Axes = None) -> plt.Axes:
    """
    Show how boundary angle maps between domains.
    
    Plot θ_physical vs θ_canonical.
    
    Features:
    - Main curve showing mapping
    - Diagonal line (identity reference)
    - Sensor locations marked (if provided)
    - Arc length scale on secondary axis
    
    Use case: Understand sensor spacing in canonical domain.
    """
```

---

## 8. greens_viz.py — Green's Function Visualizations

### 8.1 plot_greens_function

```python
def plot_greens_function(source_location: Tuple[float, float],
                         domain_type: str = 'disk',
                         domain_params: dict = None,
                         resolution: int = 100,
                         clip_singularity: float = None,
                         ax: plt.Axes = None) -> plt.Axes:
    """
    Heatmap of Neumann Green's function G(x, ξ) for fixed source ξ.
    
    Features:
    - Heatmap over domain interior
    - Source location marked with star
    - Domain boundary
    - Contour lines
    - Singularity handling: clip values within clip_singularity of source
    - Colorbar
    
    Use case: Understand kernel shape, boundary effects.
    """
```

### 8.2 plot_greens_boundary

```python
def plot_greens_boundary(source_location: Tuple[float, float],
                         domain_type: str = 'disk',
                         domain_params: dict = None,
                         n_boundary: int = 100,
                         ax: plt.Axes = None) -> plt.Axes:
    """
    Plot G(x_boundary, ξ) as function of boundary angle.
    
    This is essentially one column of the Green's matrix.
    
    Features:
    - G value vs θ
    - Source angle marked
    - Peak location (should be near source angle)
    
    Use case: Understand boundary influence of single source.
    """
```

### 8.3 plot_greens_matrix

```python
def plot_greens_matrix(G: np.ndarray,
                       sensor_angles: np.ndarray = None,
                       grid_positions: np.ndarray = None) -> plt.Figure:
    """
    Analyze Green's matrix structure (3-panel figure).
    
    Panel 1: Matrix heatmap
    - |G| as image (sensors × grid points)
    
    Panel 2: Singular value spectrum
    - log(σ_i) vs index
    - Condition number annotation
    
    Panel 3: Column coherence histogram
    - Distribution of |⟨g_i, g_j⟩| / (‖g_i‖ ‖g_j‖)
    - Vertical line at 0.33 (sparse recovery threshold)
    - Max coherence annotation
    
    Use case: Diagnose linear solver ill-conditioning.
    """
```

---

## 9. animation.py — Animated Visualizations

### 9.1 animate_optimization

```python
def animate_optimization(trajectory: List[np.ndarray],
                         objective_history: List[float],
                         n_sources: int,
                         domain_boundary: np.ndarray,
                         sources_true: List[Tuple] = None,
                         u_measured: np.ndarray = None,
                         theta: np.ndarray = None,
                         forward_solver = None,
                         interval: int = 100,
                         save_path: str = None) -> animation.FuncAnimation:
    """
    Full optimization animation (2x2 layout).
    
    Top-left: Source positions on domain
    - Sources move from initial to final
    - Trajectory trails (fading)
    - True sources (static)
    
    Top-right: Boundary fit
    - Measured data (static)
    - Current fit (animated)
    
    Bottom-left: Convergence curve
    - Growing as iterations progress
    
    Bottom-right: Info panel
    - Current iteration
    - Current objective
    - Position RMSE (if true sources known)
    
    Parameters:
    - interval: ms between frames
    - save_path: if provided, save as .gif or .mp4
    
    Use case: Visualize entire optimization process.
    """
```

### 9.2 animate_source_movement

```python
def animate_source_movement(trajectory: List[np.ndarray],
                            n_sources: int,
                            domain_boundary: np.ndarray,
                            sources_true: List[Tuple] = None,
                            trail_length: int = 20,
                            interval: int = 50,
                            save_path: str = None) -> animation.FuncAnimation:
    """
    Simple animation of source positions only.
    
    Features:
    - Sources as colored circles
    - Fading trajectory trails
    - True sources as hollow markers
    - Iteration counter
    
    Use case: Quick visualization of optimization dynamics.
    """
```

---

## 10. dashboard.py — Combined Dashboards

### 10.1 diagnostic_dashboard

```python
def diagnostic_dashboard(sources_true: List[Tuple],
                         sources_recovered: List[Tuple],
                         u_measured: np.ndarray,
                         u_recovered: np.ndarray,
                         theta: np.ndarray,
                         domain_boundary: np.ndarray,
                         history: List[float] = None,
                         sensor_locations: np.ndarray = None,
                         title: str = "Solver Diagnostic") -> plt.Figure:
    """
    Comprehensive single-page diagnostic (3x3 grid).
    
    Layout:
    [Source Config   ] [Interior Solution] [Boundary Fit    ]
    [Convergence     ] [Position Scatter ] [Intensity Scatter]
    [Residual Detail ] [Error Table      ] [Summary Stats   ]
    
    All key information at a glance for debugging a single solve.
    
    Use case: Quick comprehensive diagnosis after running solver.
    """
```

### 10.2 solver_comparison_dashboard

```python
def solver_comparison_dashboard(results: List[Dict],
                                 sources_true: List[Tuple],
                                 u_measured: np.ndarray,
                                 domain_boundary: np.ndarray) -> plt.Figure:
    """
    Compare multiple solvers side-by-side.
    
    results: list of dicts with keys:
        'name': solver name
        'sources': recovered sources
        'u_recovered': boundary data
        'history': convergence history (optional)
        'time': computation time
    
    Layout:
    - Row per solver
    - Columns: source recovery, boundary fit, metrics
    - Bottom: bar chart comparison of position RMSE
    
    Use case: Compare L-BFGS-B vs DE, or FEM vs Analytical.
    """
```

### 10.3 domain_comparison_dashboard

```python
def domain_comparison_dashboard(domain_results: Dict[str, Dict],
                                 domain_boundaries: Dict[str, np.ndarray]) -> plt.Figure:
    """
    Compare solver performance across domains.
    
    domain_results: dict mapping domain_type -> result dict
    
    Layout:
    - Column per domain
    - Rows: source config, boundary fit, metrics
    
    Use case: Verify solver works across disk, ellipse, star, etc.
    """
```

---

## 11. __init__.py — Public API

```python
"""
Inverse Source Visualization Toolbox
====================================

Comprehensive visualization tools for inverse source localization.

Quick Start:
    from visualization import diagnostic_dashboard, plot_convergence
    
    # After running solver...
    fig = diagnostic_dashboard(
        sources_true, sources_recovered,
        u_measured, u_recovered, theta,
        domain_boundary, history
    )
    fig.savefig('diagnostic.png')

Modules:
    forward_viz     - Forward problem visualizations
    inverse_viz     - Inverse problem visualizations
    optimization_viz - Optimization diagnostics
    barrier_viz     - Constraint/barrier visualizations
    conformal_viz   - Conformal mapping visualizations
    greens_viz      - Green's function visualizations
    animation       - Animated visualizations
    dashboard       - Combined diagnostic dashboards
"""

from .config import COLORS, FIGSIZE, COLORMAPS, apply_style

from .forward_viz import (
    plot_boundary_values,
    plot_interior_solution,
    plot_source_configuration,
)

from .inverse_viz import (
    plot_boundary_fit,
    plot_source_recovery,
    plot_linear_solution,
)

from .optimization_viz import (
    plot_convergence,
    plot_multistart_convergence,
    plot_source_trajectory,
    plot_parameter_evolution,
)

from .barrier_viz import (
    plot_barrier_landscape,
    plot_objective_slice,
    plot_barrier_radial_profile,
)

from .conformal_viz import (
    plot_domain_correspondence,
    plot_mapping_jacobian,
    plot_boundary_correspondence,
)

from .greens_viz import (
    plot_greens_function,
    plot_greens_boundary,
    plot_greens_matrix,
)

from .animation import (
    animate_optimization,
    animate_source_movement,
)

from .dashboard import (
    diagnostic_dashboard,
    solver_comparison_dashboard,
    domain_comparison_dashboard,
)

from .utils import (
    get_domain_boundary,
    match_sources,
    compute_source_errors,
)

# Apply style on import
apply_style()
```

---

## Implementation Notes for Claude Code

### Priority Order
1. **config.py, utils.py** — Foundation (needed by everything else)
2. **forward_viz.py** — Basic visualizations
3. **optimization_viz.py** — Critical for debugging (convergence, trajectories)
4. **barrier_viz.py** — Important for current barrier function debugging
5. **inverse_viz.py** — Source recovery analysis
6. **dashboard.py** — Combines everything
7. **greens_viz.py** — Matrix analysis
8. **conformal_viz.py** — Conformal mapping analysis
9. **animation.py** — Nice to have

### Integration with Existing Code

The visualization module should work with:
- `analytical_solver.py` — AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver
- `fem_solver.py` — FEMForwardSolver, FEMNonlinearInverseSolver
- `conformal_solver.py` — ConformalForwardSolver, ConformalNonlinearInverseSolver
- `comparison.py` — ComparisonResult

Solvers provide:
- `forward_solver.solve(sources)` → boundary values
- `forward_solver.sensor_locations` → sensor positions
- `inverse_result.sources` → list of Source objects with .x, .y, .intensity
- `inverse_result.history` → list of objective values

### Testing

After implementation, test with:
```python
# Quick test
from visualization import plot_boundary_values, diagnostic_dashboard
from analytical_solver import AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver

# Create test sources
sources = [((0.5, 0.0), 1.0), ((-0.5, 0.0), -1.0)]

# Forward solve
fwd = AnalyticalForwardSolver(n_boundary_points=100)
u = fwd.solve(sources)
theta = np.linspace(0, 2*np.pi, 100, endpoint=False)

# Plot
plot_boundary_values(theta, u, sources)
plt.show()
```
