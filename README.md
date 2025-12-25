# Inverse Source Localization

A Python package for recovering point source locations and intensities from boundary measurements using the Poisson equation with zero Neumann boundary conditions.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Installation](#installation)
- [Package Structure](#package-structure)
- [Quick Start](#quick-start)
- [Solver Types](#solver-types)
- [Regularization Methods](#regularization-methods)
- [Configuration Files](#configuration-files)
- [API Reference](#api-reference)
- [Command Line Interface](#command-line-interface)
- [Examples](#examples)

---

## Problem Statement

Given boundary measurements `u(x)` on `∂Ω`, recover point sources `{(ξₖ, qₖ)}` satisfying:

```
-Δu = Σₖ qₖ δ(x - ξₖ)   in Ω
∂u/∂n = 0              on ∂Ω
```

with the compatibility condition `Σₖ qₖ = 0`.

---

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/Shaerdan/inverse_source_project.git
cd inverse_source_project

# Install in development mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### Dependencies

**Required:**
- numpy
- scipy
- matplotlib
- pyyaml

**Optional:**
- gmsh (for advanced mesh generation)
- scikit-fem (for FEM solver)

### Verify Installation

```python
import inverse_source
print(inverse_source.__version__)  # Should print "2.0.0"
```

---

## Package Structure

```
inverse_source/
├── analytical_solver.py   # Exact solution for unit disk (fastest)
├── bem_solver.py          # True BEM with numerical integration
├── conformal_solver.py    # General domains via conformal mapping
├── fem_solver.py          # Finite element method
├── mesh.py                # Mesh generation utilities
├── regularization.py      # L1, L2, TV regularization algorithms
├── parameter_study.py     # L-curve analysis and parameter sweeps
├── comparison.py          # Solver comparison utilities
├── config.py              # JSON/YAML configuration system
├── utils.py               # Plotting and analysis utilities
├── cli.py                 # Command-line interface
└── __init__.py            # Package exports
```

---

## Quick Start

### Basic Usage

```python
from inverse_source import (
    AnalyticalForwardSolver,
    AnalyticalLinearInverseSolver,
    generate_synthetic_data
)
import numpy as np

# 1. Define true sources (must sum to zero)
true_sources = [
    ((-0.3, 0.4), 1.0),   # ((x, y), intensity)
    ((0.5, 0.3), 1.0),
    ((-0.4, -0.4), -1.0),
    ((0.3, -0.5), -1.0),
]

# 2. Generate synthetic boundary measurements
forward = AnalyticalForwardSolver(n_boundary_points=100)
u_measured = forward.solve(true_sources)
u_measured += 0.01 * np.random.randn(len(u_measured))  # Add noise

# 3. Solve inverse problem
inverse = AnalyticalLinearInverseSolver(n_boundary=100, source_resolution=0.15)
inverse.build_greens_matrix()
q_recovered = inverse.solve_l1(u_measured, alpha=1e-4)

# 4. Find significant sources
positions = inverse.get_interior_positions()
threshold = 0.1 * np.abs(q_recovered).max()
significant = np.where(np.abs(q_recovered) > threshold)[0]

print(f"Found {len(significant)} significant sources")
for i in significant:
    print(f"  Position: ({positions[i,0]:.3f}, {positions[i,1]:.3f}), "
          f"Intensity: {q_recovered[i]:.3f}")
```

### Using Configuration Files

```python
from inverse_source import Config, get_template

# Load from file
config = Config.load("config.yaml")

# Or use a template
config = get_template("high_resolution")

# Access parameters
print(config.inverse.alpha)
print(config.forward.n_boundary_points)
```

---

## Solver Types

### 1. Analytical Solver (Recommended for Unit Disk)

Uses the exact closed-form Neumann Green's function. **Fastest and most accurate** for point sources on the unit disk.

```python
from inverse_source import (
    AnalyticalForwardSolver,
    AnalyticalLinearInverseSolver,
    AnalyticalNonlinearInverseSolver
)

# Forward solver
forward = AnalyticalForwardSolver(n_boundary_points=100)
u = forward.solve(sources)

# Linear inverse (sources on grid)
linear = AnalyticalLinearInverseSolver(
    n_boundary=100,
    source_resolution=0.15,  # Grid spacing
    verbose=True
)
linear.build_greens_matrix()
q = linear.solve_l1(u, alpha=1e-4)

# Nonlinear inverse (continuous positions)
nonlinear = AnalyticalNonlinearInverseSolver(n_sources=4, n_boundary=100)
nonlinear.set_measured_data(u)
result = nonlinear.solve(method='differential_evolution', maxiter=200)
```

### 2. BEM Solver (Numerical Integration)

True Boundary Element Method with numerical integration. Use for validation or distributed sources.

```python
from inverse_source import BEMNumericalForwardSolver, BEMNumericalLinearInverseSolver

# Forward solver
bem = BEMNumericalForwardSolver(n_elements=64, quadrature_order=6)
u = bem.solve(sources)

# For distributed sources
def source_func(x, y):
    return np.exp(-10*(x**2 + y**2))

u = bem.solve_distributed_source(source_func, n_quad=20)
```

### 3. FEM Solver (Finite Element Method)

Mesh-based discretization. Works for any domain geometry.

```python
from inverse_source import FEMForwardSolver, FEMLinearInverseSolver

# Forward solver
fem = FEMForwardSolver(mesh_resolution=0.1)
u = fem.solve(sources)

# Inverse solver
inverse = FEMLinearInverseSolver(mesh_resolution=0.1)
q = inverse.solve_l1(u, alpha=1e-4)
```

### 4. Conformal Solver (General Domains)

Maps general simply-connected domains to the unit disk, then uses the analytical solution.

```python
from inverse_source import (
    EllipseMap, DiskMap, StarShapedMap,
    ConformalForwardSolver,
    ConformalLinearInverseSolver
)

# Ellipse domain with semi-axes a=2, b=1
ellipse = EllipseMap(a=2.0, b=1.0)
solver = ConformalForwardSolver(ellipse, n_boundary=100)

# Sources inside ellipse
sources = [((-0.5, 0.3), 1.0), ((0.8, -0.2), -1.0)]
u = solver.solve(sources)

# Inverse solver
inverse = ConformalLinearInverseSolver(ellipse, n_boundary=100)
q = inverse.solve_l1(u, alpha=1e-4)

# Star-shaped domain r = R(θ)
def radius_func(theta):
    return 1.0 + 0.2 * np.cos(5 * theta)

star = StarShapedMap(radius_func)
solver = ConformalForwardSolver(star, n_boundary=100)
```

---

## Regularization Methods

### L1 Regularization (Sparsity-Promoting)

**Best for point source recovery.** Promotes sparse solutions.

```python
q = inverse.solve_l1(u, alpha=1e-4, max_iter=50)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 1e-4 | Regularization strength |
| `max_iter` | int | 50 | Maximum IRLS iterations |

### L2 Regularization (Tikhonov)

Produces smooth solutions. Good for distributed sources.

```python
q = inverse.solve_l2(u, alpha=1e-4)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 1e-4 | Regularization strength |

### Total Variation (TV)

Promotes piecewise constant solutions.

```python
# ADMM algorithm
q = inverse.solve_tv(u, alpha=1e-4, method='admm', rho=1.0, max_iter=100)

# Chambolle-Pock algorithm
q = inverse.solve_tv(u, alpha=1e-4, method='chambolle_pock', max_iter=200)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 1e-4 | Regularization strength |
| `method` | str | 'admm' | Algorithm: 'admm' or 'chambolle_pock' |
| `rho` | float | 1.0 | ADMM penalty parameter |
| `max_iter` | int | 100 | Maximum iterations |

---

## Configuration Files

The package supports JSON and YAML configuration files for parameter management.

### Creating a Configuration File

```python
from inverse_source import Config, create_default_config

# Create default config.json
config = create_default_config("config.json")

# Or create YAML
config = Config()
config.save("config.yaml")
```

### Loading Configuration

```python
from inverse_source import Config, get_config

# Load from file
config = Config.load("config.yaml")

# Or with fallback to defaults
config = get_config("config.yaml")  # Returns defaults if file not found
```

### Configuration Structure

#### JSON Format (`config.json`)

```json
{
  "forward": {
    "method": "analytical",
    "n_boundary_points": 100,
    "domain_type": "disk",
    "domain_params": {"radius": 1.0}
  },
  "inverse": {
    "method": "linear",
    "regularization": "l1",
    "alpha": 0.0001,
    "n_sources": 4,
    "optimizer": "L-BFGS-B",
    "max_iter": 200,
    "tolerance": 1e-6
  },
  "grid": {
    "n_radial": 10,
    "n_angular": 20,
    "r_min": 0.1,
    "r_max": 0.9
  },
  "tv": {
    "algorithm": "chambolle_pock",
    "tau": 0.1,
    "sigma": 0.1,
    "theta": 1.0,
    "rho": 1.0,
    "max_iter": 500,
    "tol": 1e-5
  },
  "visualization": {
    "live_plot": false,
    "plot_interval": 10,
    "save_figures": true,
    "figure_dir": "results",
    "dpi": 150
  }
}
```

#### YAML Format (`config.yaml`)

```yaml
forward:
  method: "analytical"        # "analytical", "bem", or "fem"
  n_boundary_points: 100      # Boundary discretization
  domain_type: "disk"         # "disk", "ellipse", or "star"
  domain_params:
    radius: 1.0               # For disk
    a: 2.0                    # For ellipse (semi-major axis)
    b: 1.0                    # For ellipse (semi-minor axis)

inverse:
  method: "linear"            # "linear" or "nonlinear"
  regularization: "l1"        # "l1", "l2", or "tv"
  alpha: 1.0e-4               # Regularization parameter
  n_sources: 4                # For nonlinear method
  optimizer: "L-BFGS-B"       # Optimization algorithm
  max_iter: 200
  tolerance: 1.0e-6

grid:
  n_radial: 10                # Radial grid divisions
  n_angular: 20               # Angular grid divisions
  r_min: 0.1                  # Minimum radius
  r_max: 0.9                  # Maximum radius

tv:
  algorithm: "chambolle_pock" # "chambolle_pock" or "admm"
  tau: 0.1                    # Primal step size
  sigma: 0.1                  # Dual step size
  theta: 1.0                  # Extrapolation parameter
  rho: 1.0                    # ADMM penalty
  max_iter: 500
  tol: 1.0e-5

visualization:
  live_plot: false
  plot_interval: 10
  save_figures: true
  figure_dir: "results"
  dpi: 150
```

### Configuration Options Reference

#### Forward Configuration (`forward`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `method` | str | "analytical" | Solver type: "analytical", "bem", "fem" |
| `n_boundary_points` | int | 100 | Number of boundary measurement points |
| `domain_type` | str | "disk" | Domain geometry: "disk", "ellipse", "star" |
| `domain_params` | dict | {"radius": 1.0} | Domain-specific parameters |

#### Inverse Configuration (`inverse`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `method` | str | "linear" | "linear" (grid-based) or "nonlinear" (continuous) |
| `regularization` | str | "l1" | "l1", "l2", or "tv" |
| `alpha` | float | 1e-4 | Regularization parameter |
| `n_sources` | int | 4 | Number of sources (nonlinear only) |
| `optimizer` | str | "L-BFGS-B" | Optimization algorithm (nonlinear only) |
| `max_iter` | int | 200 | Maximum iterations |
| `tolerance` | float | 1e-6 | Convergence tolerance |

**Optimizer Options:**
- `"L-BFGS-B"` - Fast gradient-based (may find local minimum)
- `"differential_evolution"` - Global optimizer (slower but robust)
- `"basinhopping"` - Hybrid global/local
- `"SLSQP"` - Sequential Least Squares
- `"trust-constr"` - Trust region constrained

#### Grid Configuration (`grid`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `n_radial` | int | 10 | Radial divisions in polar grid |
| `n_angular` | int | 20 | Angular divisions |
| `r_min` | float | 0.1 | Minimum radius (avoid origin singularity) |
| `r_max` | float | 0.9 | Maximum radius (stay inside boundary) |

#### TV Configuration (`tv`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `algorithm` | str | "chambolle_pock" | "chambolle_pock" or "admm" |
| `tau` | float | 0.1 | Primal step size (CP only) |
| `sigma` | float | 0.1 | Dual step size (CP only) |
| `theta` | float | 1.0 | Extrapolation parameter |
| `rho` | float | 1.0 | ADMM penalty parameter |
| `max_iter` | int | 500 | TV-specific iteration limit |
| `tol` | float | 1e-5 | Convergence tolerance |

#### Visualization Configuration (`visualization`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `live_plot` | bool | false | Show live updates during optimization |
| `plot_interval` | int | 10 | Update every N iterations |
| `save_figures` | bool | true | Auto-save figures |
| `figure_dir` | str | "results" | Output directory |
| `dpi` | int | 150 | Figure resolution (150=screen, 300=publication) |

### Pre-defined Templates

```python
from inverse_source import get_template, TEMPLATES

# Available templates
print(TEMPLATES.keys())
# dict_keys(['default', 'high_resolution', 'fast', 'tv_chambolle_pock', 
#            'tv_admm', 'nonlinear', 'ellipse'])

# Use a template
config = get_template("high_resolution")
config = get_template("nonlinear")
config = get_template("tv_admm")
```

---

## API Reference

### AnalyticalForwardSolver

```python
class AnalyticalForwardSolver(n_boundary_points: int = 100)
```

**Methods:**
- `solve(sources)` → `np.ndarray`: Compute boundary values
- `solve_interior(sources, x_eval)` → `np.ndarray`: Evaluate at interior points
- `solve_with_gradient(sources)` → `Tuple[np.ndarray, np.ndarray]`: Values and gradients

### AnalyticalLinearInverseSolver

```python
class AnalyticalLinearInverseSolver(
    n_boundary: int = 100,
    source_resolution: float = 0.15,
    verbose: bool = True
)
```

**Methods:**
- `build_greens_matrix()`: Build forward operator G
- `solve_l1(u, alpha)` → `np.ndarray`: L1 regularized solution
- `solve_l2(u, alpha)` → `np.ndarray`: L2 regularized solution
- `solve_tv(u, alpha, method, ...)` → `np.ndarray`: TV regularized solution
- `get_interior_positions()` → `np.ndarray`: Source candidate grid

### AnalyticalNonlinearInverseSolver

```python
class AnalyticalNonlinearInverseSolver(
    n_sources: int,
    n_boundary: int = 100
)
```

**Methods:**
- `set_measured_data(u)`: Set boundary measurements
- `solve(method, maxiter, n_restarts)` → `InverseResult`: Solve optimization

### Conformal Maps

```python
class DiskMap(radius: float = 1.0)
class EllipseMap(a: float = 2.0, b: float = 1.0)
class StarShapedMap(radius_func: Callable, n_terms: int = 32)
```

**Methods:**
- `to_disk(z)` → complex array: Map to unit disk
- `from_disk(w)` → complex array: Map from unit disk
- `boundary_physical(n_points)` → complex array: Physical boundary
- `is_inside(z)` → bool array: Check if points inside domain

---

## Command Line Interface

```bash
# Run with default config
python -m inverse_source.cli run

# Run with custom config
python -m inverse_source.cli run --config my_config.yaml

# Parameter sweep (L-curve analysis)
python -m inverse_source.cli sweep --plot

# Compare solvers
python -m inverse_source.cli compare

# Generate default config file
python -m inverse_source.cli init-config --output config.yaml
```

---

## Examples

### Example 1: Complete Workflow

```python
import numpy as np
import matplotlib.pyplot as plt
from inverse_source import (
    AnalyticalForwardSolver,
    AnalyticalLinearInverseSolver,
    create_test_sources,
    plot_recovery_comparison
)

# Create test sources
sources = create_test_sources(n_sources=4, seed=42)

# Forward solve with noise
forward = AnalyticalForwardSolver(100)
u = forward.solve(sources)
u_noisy = u + 0.01 * np.std(u) * np.random.randn(len(u))

# Inverse solve
inverse = AnalyticalLinearInverseSolver(100, source_resolution=0.1)
inverse.build_greens_matrix()
q = inverse.solve_l1(u_noisy, alpha=1e-4)

# Visualize
positions = inverse.get_interior_positions()
plot_recovery_comparison(sources, positions, q)
plt.savefig("recovery.png")
```

### Example 2: L-Curve Parameter Selection

```python
from inverse_source import parameter_sweep, find_l_curve_corner
from inverse_source.regularization import solve_l1

# Parameter sweep
alphas = np.logspace(-6, -1, 30)
result = parameter_sweep(G, u, solve_l1, alpha_range=alphas)

print(f"Optimal alpha: {result.optimal_alpha:.2e}")

# Use optimal solution
q_optimal = result.solutions[result.optimal_index]
```

### Example 3: Ellipse Domain

```python
from inverse_source import EllipseMap, ConformalForwardSolver

# Create ellipse
ellipse = EllipseMap(a=2.0, b=1.0)

# Sources must be inside ellipse
sources = [
    ((-0.8, 0.3), 1.0),
    ((1.2, -0.2), 1.0),
    ((0.0, 0.0), -2.0),
]

# Solve
solver = ConformalForwardSolver(ellipse, n_boundary=100)
u = solver.solve(sources)
```

---

## License

MIT License

## Author

Serdan - [GitHub](https://github.com/Shaerdan)

## References

- Shataer, S. (2021) - Derivation of closed-form Neumann Green's function for unit disk
- Stakgold, I. "Green's Functions and Boundary Value Problems"
- Sauter & Schwab, "Boundary Element Methods"
