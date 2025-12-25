# Inverse Source Localization Package

A comprehensive toolkit for inverse source localization in 2D domains using Boundary Element Methods (BEM) and Finite Element Methods (FEM).

## Mathematical Problem

We solve the inverse problem for the Poisson equation with point sources:

```
-Δu = Σₖ qₖ δ(x - ξₖ)    in Ω
∂u/∂n = 0                 on ∂Ω
```

**Compatibility condition**: Σₖ qₖ = 0 (conservation)

**Goal**: Given boundary measurements u|∂Ω, recover source locations ξₖ and intensities qₖ.

## Package Architecture

### Two Formulations × Two Methods = Four Solver Combinations

|  | **BEM** (Analytical) | **FEM** (Numerical) |
|--|----------------------|---------------------|
| **Linear** (Grid) | `BEMLinearInverseSolver` | `FEMLinearInverseSolver` |
| **Nonlinear** (Continuous) | `BEMNonlinearInverseSolver` | `FEMNonlinearInverseSolver` |

### Linear vs Nonlinear Formulation

| Aspect | Linear (Distributional) | Nonlinear (Continuous) |
|--------|------------------------|------------------------|
| Source positions | Fixed to grid | Continuous optimization |
| Variables | Intensities q ∈ ℝᴹ | Positions + intensities (ξₖ, qₖ) |
| Problem type | Convex (unique minimum) | Non-convex (local minima) |
| # Sources | Don't need to know | Must specify N |
| Regularization | Essential (ill-posed) | Optional |
| Accuracy | Limited by grid | Exact positions possible |

### BEM vs FEM

| Aspect | BEM | FEM |
|--------|-----|-----|
| Green's function | Analytical | Computed numerically |
| Mesh required | No (boundary only) | Yes (domain mesh) |
| Speed | Fast | Slower |
| Domain flexibility | Unit disk + conformal maps | Any mesh |
| Source handling | Always continuous | Barycentric interpolation |

## Features

### Solvers
- **BEM**: Analytical Green's function for unit disk (mesh-free)
- **Conformal BEM**: General simply connected domains via conformal mapping
  - Ellipse (Joukowsky map)
  - Star-shaped domains (numerical conformal map)
- **FEM**: Triangular mesh with P1 elements (scipy-based, DOLFINx optional)

### Regularization (for Linear Inverse)
- **L2 (Tikhonov)**: Smooth solutions - closed form
- **L1 (Sparsity)**: Sparse solutions via IRLS algorithm (best for point sources)
- **TV (Total Variation)**: Piecewise constant solutions
  - Chambolle-Pock primal-dual algorithm
  - ADMM (Alternating Direction Method of Multipliers)

### Tools
- JSON/YAML configuration system
- Parameter sweep and L-curve analysis
- Source error metrics computation
- Comprehensive plotting utilities
- Command-line interface

## Installation

```bash
# Clone the repository
git clone https://github.com/Shaerdan/inverse_source_project.git
cd inverse_source_project

# Install dependencies
pip install numpy scipy matplotlib pyyaml

# Install package in development mode
pip install -e .
```

## Quick Start

### BEM Nonlinear Inverse (Recommended for Point Sources)

```python
from inverse_source import bem_solver

# Define true sources (position, intensity) - must sum to zero
sources_true = [
    ((-0.3, 0.4), 1.0),
    ((0.5, 0.3), 1.0),
    ((-0.4, -0.4), -1.0),
    ((0.3, -0.5), -1.0),
]

# Forward solve
forward = bem_solver.BEMForwardSolver(n_boundary_points=100)
u_measured = forward.solve(sources_true)

# Add noise
import numpy as np
u_measured += 0.001 * np.random.randn(len(u_measured))

# Nonlinear inverse (continuous source positions)
inverse = bem_solver.BEMNonlinearInverseSolver(n_sources=4, n_boundary=100)
inverse.set_measured_data(u_measured)
result = inverse.solve(method='L-BFGS-B')

# Access recovered sources
for s in result.sources:
    print(f"Position: ({s.x:.3f}, {s.y:.3f}), Intensity: {s.intensity:.3f}")
```

### BEM Linear Inverse (Grid-Based with Regularization)

```python
from inverse_source import bem_solver

# Forward solve (same as above)
forward = bem_solver.BEMForwardSolver(n_boundary_points=100)
u_measured = forward.solve(sources_true)

# Linear inverse (sources on grid)
linear = bem_solver.BEMLinearInverseSolver(
    n_boundary=100,
    n_interior_radial=10,
    n_interior_angular=20
)
linear.build_greens_matrix()

# Solve with L1 regularization (sparsity-promoting)
q = linear.solve_l1(u_measured, alpha=1e-4)

# Find significant sources
threshold = 0.1 * np.max(np.abs(q))
significant_idx = np.where(np.abs(q) > threshold)[0]
positions = linear.interior_points[significant_idx]
```

### FEM Nonlinear Inverse (Continuous Positions)

```python
from inverse_source import fem_solver

# Forward solve with FEM
forward = fem_solver.FEMForwardSolver(n_radial=15, n_angular=30)
u_measured = forward.solve(sources_true)

# Nonlinear inverse
inverse = fem_solver.FEMNonlinearInverseSolver(n_sources=4, n_radial=15, n_angular=30)
inverse.set_measured_data(u_measured)
result = inverse.solve(method='L-BFGS-B')
```

### FEM Linear Inverse (Grid-Based)

```python
from inverse_source import fem_solver

# Linear inverse with FEM
linear = fem_solver.FEMLinearInverseSolver(n_radial=15, n_angular=30)
q = linear.solve_l1(u_measured, alpha=1e-3)
```

### Conformal BEM (General Domains)

```python
from inverse_source import conformal_bem

# Ellipse domain (a=2, b=1)
ellipse = conformal_bem.EllipseMap(a=2.0, b=1.0)

# Forward solve
solver = conformal_bem.ConformalBEMSolver(ellipse, n_boundary=100)
u_measured = solver.solve_forward(sources_true)

# Nonlinear inverse
inverse = conformal_bem.ConformalNonlinearInverse(ellipse, n_sources=4)
inverse.set_measured_data(u_measured)
result = inverse.solve()
```

### Command Line Interface

```bash
# Run demo
python -m inverse_source.cli demo --type bem

# Solve with specific parameters
python -m inverse_source.cli solve --method l1 --alpha 1e-4

# Parameter sweep with L-curve analysis
python -m inverse_source.cli sweep --method all --plot

# Show available config templates
python -m inverse_source.cli config --list-templates

# Package info
python -m inverse_source.cli info
```

### Using Configuration Files

```python
from inverse_source.config import Config

# Load YAML config
config = Config.load('Config/config.yaml')

# Access parameters
print(config.inverse.regularization)  # 'l1'
print(config.inverse.alpha)           # 1e-4
print(config.grid.n_radial)           # 10

# Use with solver
from inverse_source import bem_solver
forward = bem_solver.BEMForwardSolver(n_boundary_points=config.forward.n_boundary_points)
```

## Project Structure

```
inverse_source_project/
├── src/
│   ├── __init__.py           # Package exports
│   ├── bem_solver.py         # BEM forward + linear/nonlinear inverse
│   ├── fem_solver.py         # FEM forward + linear/nonlinear inverse
│   ├── conformal_bem.py      # Conformal mapping for general domains
│   ├── regularization.py     # L1, L2, TV algorithms
│   ├── parameter_study.py    # Parameter sweeps and L-curve analysis
│   ├── config.py             # JSON/YAML configuration system
│   ├── utils.py              # Plotting and utilities
│   └── cli.py                # Command-line interface
├── Config/
│   └── config.yaml           # Configuration file with all options
├── docs/
│   ├── main.tex              # LaTeX documentation
│   ├── references.bib        # Bibliography
│   └── main.pdf              # Compiled documentation
├── examples/
│   └── complete_example.py   # Comprehensive usage examples
├── results/                  # Output directory for figures
├── pyproject.toml            # Package configuration
├── requirements.txt          # Dependencies
└── README.md
```

## Solver Reference

### Forward Solvers

| Class | Description |
|-------|-------------|
| `BEMForwardSolver` | Analytical Green's function for unit disk |
| `FEMForwardSolver` | Scipy-based FEM (DOLFINx optional) |
| `ConformalBEMSolver` | BEM for general domains via conformal map |

### Inverse Solvers

| Class | Type | Description |
|-------|------|-------------|
| `BEMLinearInverseSolver` | Linear | Grid-based, L1/L2/TV regularization |
| `BEMNonlinearInverseSolver` | Nonlinear | Continuous positions, gradient optimization |
| `FEMLinearInverseSolver` | Linear | FEM-based Green's matrix |
| `FEMNonlinearInverseSolver` | Nonlinear | FEM forward in optimization loop |
| `ConformalLinearInverse` | Linear | General domains |
| `ConformalNonlinearInverse` | Nonlinear | General domains |

### Regularization Methods

| Method | Function | Best For |
|--------|----------|----------|
| L2 (Tikhonov) | `solve_l2()` | Smooth distributed sources |
| L1 (Sparsity) | `solve_l1()` | Point sources (recommended) |
| TV (Total Variation) | `solve_tv()` | Piecewise constant regions |

## Mathematical Details

### BEM: Analytical Green's Function

For the unit disk with Neumann boundary conditions:

```
G(x, ξ) = -1/(2π) [ln|x - ξ| + ln|x - ξ*| - ln|ξ|]
```

where ξ* = ξ/|ξ|² is the image point (method of images).

### FEM: Continuous Source Positions

Sources at arbitrary positions ξₖ are handled via barycentric interpolation:

1. Find element containing ξₖ
2. Compute barycentric coordinates (λ₁, λ₂, λ₃)
3. Distribute source to vertices: fᵢ = qₖ · λᵢ

This allows **continuous optimization** over source positions.

### Linear Inverse: Regularized Least Squares

```
min_q ||Gq - u||² + α·R(q)
```

where R(q) is:
- L2: ||q||²
- L1: ||q||₁
- TV: ||Dq||₁

### Nonlinear Inverse: Direct Optimization

```
min_{ξ,q} ||u(ξ,q) - u_measured||²

subject to: Σqₖ = 0, ξₖ ∈ Ω
```

## Citation

```bibtex
@software{inverse_source_localization,
  author = {Serdan},
  title = {Inverse Source Localization Package},
  year = {2025},
  url = {https://github.com/Shaerdan/inverse_source_project}
}
```

## License

MIT License

## Author

Serdan (https://github.com/Shaerdan)
