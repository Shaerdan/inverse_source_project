# Inverse Source Localization

A Python package for recovering point source locations and intensities from boundary measurements using the Poisson equation with zero Neumann boundary conditions.

## Problem Statement

Given boundary measurements `u(x)` on `∂Ω`, recover point sources `{(ξₖ, qₖ)}` satisfying:

```
-Δu = Σₖ qₖ δ(x - ξₖ)  in Ω
∂u/∂n = 0              on ∂Ω
```

with the compatibility condition `Σₖ qₖ = 0`.

## Installation

```bash
# Clone the repository
git clone https://github.com/Shaerdan/inverse_source_project.git
cd inverse_source_project

# Install in development mode
pip install -e .

# Optional: Install gmsh for better mesh generation
pip install gmsh
```

## Quick Start

```bash
# Run comparison with optimal regularization parameters (recommended)
python -m inverse_source.cli compare --optimal

# Quick comparison (fixed α, faster)
python -m inverse_source.cli compare --quick

# Show package info
python -m inverse_source.cli info
```

## Architecture

### Two Approaches

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Linear (Distributed)** | Sources on fixed grid, solve for intensities `q ∈ ℝᴹ` | Convex problem, global optimum | Grid-constrained positions |
| **Nonlinear (Continuous)** | Optimize both positions `ξ` and intensities `q` | Exact positions possible | Non-convex, local minima |

### Two Forward Methods

| Method | Description | When to Use |
|--------|-------------|-------------|
| **BEM** | Analytical Green's function | Unit disk (exact, fast) |
| **FEM** | Finite element discretization | General domains, validation |

### Solver Classes

```
┌─────────────────────────────────────────────────────────────────┐
│                        mesh.py (shared)                         │
│  Uniform triangular mesh generation (gmsh or fallback)          │
└─────────────────────────────────────────────────────────────────┘
                    │                           │
                    ▼                           ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐
│        fem_solver.py        │   │        bem_solver.py        │
│                             │   │                             │
│  FEMForwardSolver           │   │  BEMForwardSolver           │
│  FEMLinearInverseSolver     │   │  BEMLinearInverseSolver     │
│  FEMNonlinearInverseSolver  │   │  BEMNonlinearInverseSolver  │
└─────────────────────────────┘   └─────────────────────────────┘
```

## Regularization Methods

### Linear Solvers

| Method | Algorithm | Best For | Function |
|--------|-----------|----------|----------|
| **L2 (Tikhonov)** | Closed-form | Smooth solutions | `solve_l2(u, alpha)` |
| **L1 (Sparsity)** | IRLS | Sparse point sources | `solve_l1(u, alpha)` |
| **TV-ADMM** | ADMM | Piecewise constant | `solve_tv(u, alpha, method='admm')` |
| **TV-CP** | Chambolle-Pock | Piecewise constant | `solve_tv(u, alpha, method='chambolle_pock')` |

### Nonlinear Solvers

| Optimizer | Type | Characteristics |
|-----------|------|-----------------|
| `L-BFGS-B` | Local | Fast, may get stuck in local minima |
| `differential_evolution` | Global | Slower, more robust |
| `basinhopping` | Hybrid | Global search + local polish |

## CLI Reference

### Commands

```bash
# Compare all solvers
python -m inverse_source.cli compare [options]

# Run demos
python -m inverse_source.cli demo --type [bem|conformal|all]

# Single solve
python -m inverse_source.cli solve --method [l1|l2|tv] --alpha 1e-4

# Parameter sweep
python -m inverse_source.cli sweep --method all --plot

# Configuration
python -m inverse_source.cli config --template default --output config.json

# Package info
python -m inverse_source.cli info
```

### Compare Options

```bash
# Optimal α comparison (L-curve selection for each method) - RECOMMENDED
python -m inverse_source.cli compare --optimal

# Quick comparison (fixed α, L-BFGS-B only for nonlinear)
python -m inverse_source.cli compare --quick

# Specific methods only
python -m inverse_source.cli compare --optimal --methods l1 tv_admm tv_cp

# Skip nonlinear solvers
python -m inverse_source.cli compare --optimal --no-nonlinear

# Custom noise level
python -m inverse_source.cli compare --optimal --noise 0.01

# Save figure
python -m inverse_source.cli compare --optimal --save results/comparison.png
```

## Python API

### Basic Usage

```python
from inverse_source import (
    BEMForwardSolver, BEMLinearInverseSolver, BEMNonlinearInverseSolver,
    FEMForwardSolver, FEMLinearInverseSolver, FEMNonlinearInverseSolver
)
import numpy as np

# Define true sources: [((x, y), intensity), ...]
sources_true = [
    ((-0.3, 0.4), 1.0),
    ((0.5, 0.3), 1.0),
    ((-0.4, -0.4), -1.0),
    ((0.3, -0.5), -1.0),
]

# Generate synthetic measurements
forward = BEMForwardSolver(n_boundary_points=100)
u_measured = forward.solve(sources_true)
u_measured += 0.001 * np.random.randn(len(u_measured))  # Add noise
```

### Linear Inverse (BEM)

```python
# Setup solver
linear = BEMLinearInverseSolver(n_boundary=100, source_resolution=0.15)
linear.build_greens_matrix()

# Solve with different regularizations
q_l1 = linear.solve_l1(u_measured, alpha=1e-4)
q_l2 = linear.solve_l2(u_measured, alpha=1e-4)
q_tv_admm = linear.solve_tv(u_measured, alpha=1e-4, method='admm')
q_tv_cp = linear.solve_tv(u_measured, alpha=1e-4, method='chambolle_pock')

# Extract significant sources
threshold = 0.1 * np.abs(q_l1).max()
significant = np.where(np.abs(q_l1) > threshold)[0]
for idx in significant:
    pos = linear.interior_points[idx]
    print(f"Source at ({pos[0]:.3f}, {pos[1]:.3f}), q = {q_l1[idx]:.3f}")
```

### Nonlinear Inverse (BEM)

```python
# Setup solver
nonlinear = BEMNonlinearInverseSolver(n_sources=4, n_boundary=100)
nonlinear.set_measured_data(u_measured)

# Solve (choose optimizer)
result = nonlinear.solve(method='differential_evolution', maxiter=200)
# Or with multi-start local optimizer:
# result = nonlinear.solve(method='L-BFGS-B', n_restarts=5)

# Print results
for s in result.sources:
    print(f"Source at ({s.x:.3f}, {s.y:.3f}), q = {s.intensity:.3f}")
print(f"Residual: {result.residual:.6e}")
```

### Linear Inverse (FEM)

```python
# FEM uses two mesh resolutions: forward (finer) and source candidates (coarser)
fem_linear = FEMLinearInverseSolver(
    forward_resolution=0.1,   # FEM discretization
    source_resolution=0.15    # Source candidate grid
)
fem_linear.build_greens_matrix()

# Same solve methods as BEM
q = fem_linear.solve_l1(u_fem, alpha=1e-3)
```

### Optimal α Selection

```python
from inverse_source.comparison import find_optimal_alpha

# Find optimal α via L-curve
alpha_opt, sweep_data = find_optimal_alpha(
    linear, u_measured, method='l1', verbose=True
)
print(f"Optimal α: {alpha_opt:.2e}")

# Solve with optimal α
q = linear.solve_l1(u_measured, alpha=alpha_opt)
```

### Full Comparison with Optimal α

```python
from inverse_source.comparison import (
    compare_with_optimal_alpha,
    print_comparison_table, 
    plot_comparison
)

# Comparison with optimal α for each method (recommended)
results = compare_with_optimal_alpha(
    sources_true, 
    noise_level=0.001,
    methods=['l1', 'l2', 'tv_admm', 'tv_cp'],
    include_nonlinear=True
)

print_comparison_table(results)
plot_comparison(results, sources_true, save_path='comparison.png')
```

## File Structure

```
inverse_source_project/
├── src/
│   ├── __init__.py         # Package exports
│   ├── mesh.py             # Uniform triangular mesh generation
│   ├── bem_solver.py       # BEM forward and inverse solvers
│   ├── fem_solver.py       # FEM forward and inverse solvers
│   ├── conformal_bem.py    # Conformal mapping for general domains
│   ├── regularization.py   # L1, L2, TV algorithms (standalone)
│   ├── comparison.py       # Solver comparison utilities
│   ├── parameter_study.py  # L-curve analysis
│   ├── config.py           # Configuration management
│   ├── utils.py            # Plotting and helper functions
│   └── cli.py              # Command-line interface
├── Config/
│   └── config.yaml         # Default configuration
├── tests/                  # Unit tests
├── examples/               # Example scripts
├── docs/                   # Documentation
├── pyproject.toml          # Package metadata
└── README.md
```

## Key Findings

Based on parameter studies:

1. **L1 outperforms TV for point sources**: L1 promotes sparsity (discrete sources), TV promotes piecewise constant regions which doesn't match point source physics
2. **Optimal α selection is critical**: Use L-curve method (`--optimal` flag), not arbitrary values
3. **Nonlinear solvers need global search**: `differential_evolution` significantly outperforms `L-BFGS-B` for avoiding local minima
4. **BEM ≈ FEM accuracy**: Both give similar results on unit disk; BEM is faster due to analytical Green's function

## Dependencies

- **Required**: numpy, scipy, matplotlib
- **Optional**: gmsh (better mesh generation), pyyaml (config files)

## References

- Stakgold, I. "Green's Functions and Boundary Value Problems"
- Chambolle, A., & Pock, T. (2011). "A first-order primal-dual algorithm"
- Boyd, S., et al. (2011). "Distributed Optimization and Statistical Learning via ADMM"

## License

MIT License

## Author

Serdan - [GitHub](https://github.com/Shaerdan)
