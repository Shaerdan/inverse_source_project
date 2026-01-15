# Inverse Source Localization from Boundary Measurements

A Python package for recovering acoustic point source locations and intensities from boundary measurements using the Poisson equation with Neumann boundary conditions.

## Overview

This package implements multiple numerical approaches to the inverse source problem:

| Solver Type | Method | Best For |
|-------------|--------|----------|
| **Analytical** | Closed-form Green's function | Unit disk (fastest, most accurate) |
| **Conformal** | Joukowsky/numerical mapping | Ellipse, star-shaped domains |
| **FEM** | Finite Element Method | Any polygon, complex geometries |

### Supported Domains

- **Disk**: Unit disk with analytical Green's function
- **Ellipse**: Via conformal mapping or FEM
- **Star**: Star-shaped domains r(θ) = 1 + A·cos(nθ)
- **Square**: Unit square [-1,1]²
- **Brain**: Realistic brain-shaped domain

> **Note**: Polygon support (L-shaped domains) is available but archived due to fundamental limitations with non-convex geometries.

## Installation

### Quick Start (pip)

```bash
pip install -e .
```

### Full Installation with FEniCSx (conda)

```bash
conda create -n inverse_source python=3.11
conda activate inverse_source
conda install -c conda-forge fenics-dolfinx mpich
pip install -e .
```

## Quick Start

```python
from analytical_solver import AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver

# Define sources: list of ((x, y), intensity)
sources_true = [
    ((-0.5, 0.5), 1.0),
    ((0.5, 0.5), 1.0),
    ((-0.5, -0.5), -1.0),
    ((0.5, -0.5), -1.0),
]

# Generate synthetic boundary data
forward = AnalyticalForwardSolver(n_boundary_points=100)
u_measured = forward.solve(sources_true)

# Recover sources (nonlinear solver)
inverse = AnalyticalNonlinearInverseSolver(n_sources=4, n_boundary=100)
inverse.set_measured_data(u_measured)
result = inverse.solve(method='differential_evolution', seed=42)

for src in result.sources:
    print(f"Position: ({src.x:.3f}, {src.y:.3f}), Intensity: {src.intensity:.3f}")
```

## Command-Line Interface

### Compare Solvers Across Domains

```bash
cd src/

# Compare all active domains
python cli.py compare --domains disk ellipse star square brain

# Quick mode (L-BFGS-B only, skip differential evolution)
python cli.py compare --domains disk ellipse --quick

# Use a test preset
python cli.py compare --domains disk --preset easy_validation

# List available presets
python cli.py compare --list-presets
```

### Test Presets

Configure tests via JSON presets in `test_configurations.json`:

```bash
# List all presets
python cli.py compare --list-presets

# Use specific preset
python cli.py compare --domains disk ellipse star --preset six_sources

# Override preset settings
python cli.py compare --domains disk --preset default --n-sources 6 --noise 0.01
```

Available presets include:
- `default`: 4 sources, standard settings
- `easy_validation`: 2 well-separated sources
- `four_sources`: Standard 4-source test
- `six_sources`: 6-source challenge
- `high_noise`: Noise robustness testing
- `stress_test`: 8 sources with noise

### Calibration

```bash
# Calibrate optimal parameters for all domains
python cli.py calibrate --domains disk ellipse star square brain

# Use calibration in comparison
python cli.py compare --domains disk ellipse --use-calibration results/calibration/calibration_config.json
```

## Configuration System

### JSON Test Configuration

Create custom test configurations in `test_configurations.json`:

```json
{
  "active_preset": "default",
  "presets": {
    "custom_test": {
      "description": "My custom test configuration",
      "sources": {
        "n_sources": 4,
        "placement": {
          "method": "angular_spread",
          "depth_range": [0.15, 0.35]
        },
        "intensities": {
          "method": "alternating",
          "magnitude": 1.0
        }
      },
      "measurement": {
        "n_sensors": 100,
        "noise_level": 0.001
      },
      "optimizer": {
        "L-BFGS-B": {"n_restarts": 5, "maxiter": 2000},
        "differential_evolution": {"maxiter": 200, "polish": true}
      },
      "seed": 42
    }
  }
}
```

### YAML Configuration

Solver parameters can also be configured via YAML:

```yaml
forward:
  method: analytical
  n_boundary_points: 100
  domain_type: disk

inverse:
  method: nonlinear
  n_sources: 4
  optimizer: differential_evolution
```

## Key Results

### Nonlinear Solvers (Recommended)

All nonlinear solvers achieve position errors < 1e-5 for well-separated sources:

| Domain | L-BFGS-B | Diff. Evolution |
|--------|----------|-----------------|
| Disk | ~1e-7 | ~1e-8 |
| Ellipse | ~1e-6 | ~1e-7 |
| Star | ~1e-5 | ~1e-6 |
| Square | ~1e-5 | ~1e-6 |
| Brain | ~1e-5 | ~1e-6 |

### Linear Solvers (For Source Detection)

Linear solvers (L1/L2/TV regularization) have fundamental limitations due to high mutual coherence (~0.99) in the discretized Green's matrix. Use for:
- Initial source detection (number and approximate locations)
- Not recommended for precise localization

## File Structure

```
inverse_source_project/
├── src/
│   ├── analytical_solver.py   # Closed-form Green's function (disk)
│   ├── conformal_solver.py    # Conformal mapping (ellipse, star)
│   ├── fem_solver.py          # Finite Element Method (any domain)
│   ├── comparison.py          # Solver comparison framework
│   ├── cli.py                 # Command-line interface
│   ├── test_config.py         # JSON preset configuration
│   ├── test_configurations.json  # Test presets
│   ├── calibration.py         # Parameter calibration
│   └── mesh.py                # Mesh generation utilities
├── Config/
│   ├── config.json            # Default configuration
│   └── config_template.json   # Configuration template
├── docs/
│   └── main.pdf               # Mathematical documentation
└── tests/
    └── test_forward_solver.py # Unit tests
```

## Mathematical Background

The forward problem solves the Poisson equation with Neumann boundary conditions:

```
-Δu = Σ qₖ δ(x - ξₖ)   in Ω
∂u/∂n = 0              on ∂Ω
∫_∂Ω u ds = 0          (normalization)
```

**Compatibility condition**: Sources must satisfy Σ qₖ = 0.

For the unit disk, the Neumann Green's function is:

```
G(x, ξ) = -1/(2π) [ln|x - ξ| + ln|x - ξ*| - ln|ξ|]
```

where ξ* = ξ/|ξ|² is the Kelvin reflection.

See `docs/main.pdf` for complete mathematical derivations.

## Citation

```bibtex
@software{inverse_source,
  title = {Inverse Source Localization from Boundary Measurements},
  year = {2026},
  url = {https://github.com/Shaerdan/inverse_source_project}
}
```

## License

MIT License
