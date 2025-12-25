# Inverse Source Localization Package

A comprehensive toolkit for inverse source localization in 2D domains using Boundary Element Methods (BEM) and Finite Element Methods (FEM).

## Features

### Solvers
- **BEM (Boundary Element Method)**: Analytical Green's function for unit disk
- **Conformal BEM**: General simply connected domains via conformal mapping
  - Ellipse (Joukowsky map)
  - Star-shaped domains (numerical conformal map)
- **FEM (Finite Element Method)**: Fallback for complex domains

### Regularization
- **L2 (Tikhonov)**: Smooth solutions - closed form
- **L1 (Sparsity)**: Sparse solutions via IRLS algorithm
- **TV (Total Variation)**: Piecewise constant solutions
  - Chambolle-Pock primal-dual algorithm
  - ADMM (Alternating Direction Method of Multipliers)

### Tools
- JSON-based configuration system
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
pip install numpy scipy matplotlib

# Install package
pip install -e .
```

## Quick Start

### Python API

```python
from inverse_source import bem_solver

# Define sources (position, intensity) - must sum to zero
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

# Nonlinear inverse solve (continuous source positions)
inverse = bem_solver.BEMNonlinearInverseSolver(n_sources=4, n_boundary=100)
inverse.set_measured_data(u_measured)
sources_recovered, result = inverse.solve(method='L-BFGS-B')

# Linear inverse solve (grid-based, with regularization)
linear = bem_solver.BEMLinearInverseSolver(n_boundary=100)
q = linear.solve_l1(u_measured, alpha=1e-4)
```

### Command Line

```bash
# Run demo
python -m inverse_source.cli demo --type bem

# Solve with specific parameters
python -m inverse_source.cli solve --method l1 --alpha 1e-4

# Parameter sweep
python -m inverse_source.cli sweep --method all --plot

# Show available templates
python -m inverse_source.cli config --list-templates
```

## Mathematical Background

The package solves the inverse problem for the Poisson equation:

```
-Δu = Σₖ qₖ δ(x - ξₖ)    in Ω
∂u/∂n = 0                 on ∂Ω
```

**Compatibility condition**: Σₖ qₖ = 0 (conservation)

### Two Formulations

1. **Nonlinear**: Optimize over continuous source positions (ξₖ, qₖ)
   - Low-dimensional (3N-1 parameters for N sources)
   - Smooth objective with analytical gradients
   - Uses gradient-based optimization

2. **Linear**: Fix source positions to grid, solve for intensities qₖ
   - High-dimensional but convex
   - Regularization (L1/L2/TV) for stability
   - L-curve analysis for parameter selection

See `docs/main.pdf` for complete mathematical derivation.

## Project Structure

```
inverse_source/
├── src/
│   ├── __init__.py         # Package initialization
│   ├── bem_solver.py       # BEM solvers for unit disk
│   ├── conformal_bem.py    # Conformal mapping for general domains
│   ├── fem_solver.py       # FEM-based solvers
│   ├── regularization.py   # L1, L2, TV regularization algorithms
│   ├── parameter_study.py  # Parameter sweeps and analysis
│   ├── config.py           # JSON configuration system
│   ├── utils.py            # Plotting and utilities
│   └── cli.py              # Command-line interface
├── docs/
│   ├── main.tex            # LaTeX documentation
│   ├── references.bib      # Bibliography
│   └── main.pdf            # Compiled documentation
├── examples/
│   └── complete_example.py # Comprehensive usage example
├── results/                # Output directory for figures
├── requirements.txt
├── setup.py
└── README.md
```

## Citation

If you use this package in your research, please cite:

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
