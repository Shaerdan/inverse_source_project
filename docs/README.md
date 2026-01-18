# Inverse Source Localization from Boundary Measurements

A Python package for recovering point source locations and intensities from boundary measurements using the Poisson equation with Neumann boundary conditions.

## Overview

This package implements multiple approaches to the inverse source problem:

| Approach | Method | Domains Supported |
|----------|--------|-------------------|
| **Analytical** | Linear & Nonlinear | Unit disk only |
| **BEM Numerical** | Linear & Nonlinear | Unit disk only |
| **Conformal** | Linear & Nonlinear | Ellipse, star-shaped, any simply-connected |
| **FEM** | Linear & Nonlinear | Disk, ellipse, polygon (auto mesh generation) |

## Domain Support

### Unit Disk (Analytical & BEM)
Best performance, uses closed-form Green's function:
```python
from analytical_solver import AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver

forward = AnalyticalForwardSolver(n_boundary_points=100)
u = forward.solve(sources)
```

### Ellipse (Conformal or FEM)
```python
# Option 1: Conformal mapping (recommended)
from conformal_solver import EllipseMap, ConformalForwardSolver, ConformalNonlinearInverseSolver

ellipse = EllipseMap(a=2.0, b=1.0)  # Semi-axes
forward = ConformalForwardSolver(ellipse, n_boundary=100)
u = forward.solve(sources)

inverse = ConformalNonlinearInverseSolver(ellipse, n_sources=4, n_boundary=100)
inverse.set_measured_data(u)
result = inverse.solve()

# Option 2: FEM with auto-generated mesh
from mesh import create_ellipse_mesh, get_ellipse_source_grid
nodes, elements, boundary_idx, interior_idx = create_ellipse_mesh(a=2.0, b=1.0, resolution=0.1)
source_grid = get_ellipse_source_grid(a=2.0, b=1.0, resolution=0.15)
```

### Star-Shaped Domains (Conformal)
Any domain defined by r = R(θ):
```python
from conformal_solver import StarShapedMap, ConformalForwardSolver

# 5-petal flower
def flower_radius(theta):
    return 1.0 + 0.3 * np.cos(5 * theta)

flower = StarShapedMap(flower_radius, n_terms=32)
forward = ConformalForwardSolver(flower, n_boundary=100)
```

### Polygons (FEM)
```python
from mesh import create_polygon_mesh, get_polygon_source_grid

# Unit square
vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
nodes, elements, boundary_idx, interior_idx = create_polygon_mesh(vertices, resolution=0.1)
source_grid = get_polygon_source_grid(vertices, resolution=0.15)

# L-shaped domain
L_vertices = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
nodes, elements, boundary_idx, interior_idx = create_polygon_mesh(L_vertices, resolution=0.1)
```

### Domain Summary

| Domain | Auto Mesh | Auto Boundary | Forward | Inverse |
|--------|-----------|---------------|---------|---------|
| Unit disk | ✅ | ✅ | Analytical/BEM/FEM | ✅ Linear + Nonlinear |
| Ellipse | ✅ | ✅ | Conformal/FEM | ✅ Linear + Nonlinear |
| Star-shaped | — | ✅ | Conformal | ✅ Linear + Nonlinear |
| Polygon | ✅ | ✅ | FEM | ✅ Linear + Nonlinear |
| Custom mesh | Import | Import | FEM | ✅ Linear + Nonlinear |

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

### Dependencies

- **Core**: numpy, scipy, matplotlib, scikit-learn
- **Mesh**: gmsh, meshio (optional but recommended)
- **FEM**: fenics-dolfinx (optional, for FEM solver)

## Quick Start

```python
from analytical_solver import AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver

# Define true sources: list of ((x, y), intensity)
sources_true = [
    ((-0.3, 0.4), 1.0),
    ((0.5, 0.3), 1.0),
    ((-0.4, -0.4), -1.0),
    ((0.3, -0.5), -1.0),
]

# Generate synthetic boundary data
forward = AnalyticalForwardSolver(n_boundary_points=100)
u_measured = forward.solve(sources_true)

# Recover sources
inverse = AnalyticalNonlinearInverseSolver(n_sources=4, n_boundary=100)
inverse.set_measured_data(u_measured)
result = inverse.solve(method='differential_evolution', seed=42)

for src in result.sources:
    print(f"Position: ({src.x:.3f}, {src.y:.3f}), Intensity: {src.intensity:.3f}")
```

## Experiments

### 1. Basic Solver Comparison

Compare all solver types with default parameters:

```bash
cd src/
python -c "
from comparison import run_full_comparison
results = run_full_comparison(n_sources=4, noise_level=0.0, seed=42)
"
```

### 2. Linear Solver Regularization Comparison (L1 vs L2 vs TV)

```python
from analytical_solver import AnalyticalForwardSolver
from comparison import run_bem_linear, plot_comparison

sources = [
    ((-0.3, 0.4), 1.0), ((0.5, 0.3), 1.0),
    ((-0.4, -0.4), -1.0), ((0.3, -0.5), -1.0),
]

forward = AnalyticalForwardSolver(n_boundary_points=100)
u = forward.solve(sources)

results = []
for method in ['l1', 'l2', 'tv']:
    result = run_bem_linear(u, sources, alpha=1e-4, method=method)
    results.append(result)
    print(f"{method.upper()}: Pos RMSE={result.position_rmse:.3f}, "
          f"Int RMSE={result.intensity_rmse:.3f}, "
          f"{len(result.peaks)} peaks, {len(result.clusters)} clusters")

# Visualize
plot_comparison(results, sources, save_path='linear_comparison.png',
                show_peaks=True, show_clusters=True)
```

### 3. Nonlinear Solver Optimizer Comparison

```python
from comparison import run_bem_nonlinear

optimizers = ['differential_evolution', 'L-BFGS-B', 'basin_hopping']
for opt in optimizers:
    result = run_bem_nonlinear(u, sources, n_sources=4, 
                                optimizer=opt, n_restarts=5, seed=42)
    print(f"{opt}: Pos RMSE={result.position_rmse:.3f}")
```

### 4. FEM vs Analytical Comparison

```python
from comparison import run_bem_nonlinear, run_fem_nonlinear

result_analytical = run_bem_nonlinear(u, sources, n_sources=4, 
                                       optimizer='differential_evolution')
result_fem = run_fem_nonlinear(u, sources, n_sources=4,
                                optimizer='differential_evolution')

print(f"Analytical: Pos RMSE={result_analytical.position_rmse:.3f}")
print(f"FEM:        Pos RMSE={result_fem.position_rmse:.3f}")
```

### 5. Peak Detection vs DBSCAN Clustering Analysis

For linear solvers, the recovered intensity field is spread across the domain.
We provide two methods to identify source locations:

- **Peak Detection**: Finds local maxima/minima - better for smooth fields (L2, TV)
- **DBSCAN Clustering**: Groups connected regions - better for sparse fields (L1)

```python
from comparison import find_intensity_peaks, find_intensity_clusters

# After running linear solver...
peaks = find_intensity_peaks(
    grid_positions, grid_intensities,
    neighbor_radius=0.12,
    integration_radius=0.20,
    intensity_threshold_ratio=0.15,  # Only consider |q| > 15% of max
    min_peak_separation=0.25         # Min distance between peaks
)

clusters = find_intensity_clusters(
    grid_positions, grid_intensities,
    eps=0.18,                        # DBSCAN neighborhood radius
    min_samples=2,                   # Min points per cluster
    intensity_threshold_ratio=0.15
)

print(f"Peaks found: {len(peaks)}")
print(f"Clusters found: {len(clusters)}")
```

### 6. Regularization Parameter Study

```python
from parameter_study import run_alpha_study

alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
results = run_alpha_study(u, sources, alphas, method='l1')

for alpha, result in zip(alphas, results):
    print(f"α={alpha:.0e}: Residual={result.boundary_residual:.4f}, "
          f"Peaks={len(result.peaks)}")
```

### 7. Noise Sensitivity Analysis

```python
import numpy as np

noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1]
for noise in noise_levels:
    u_noisy = u + noise * np.random.randn(len(u)) * np.std(u)
    result = run_bem_nonlinear(u_noisy, sources, n_sources=4)
    print(f"Noise {noise*100:.0f}%: Pos RMSE={result.position_rmse:.3f}")
```

### 8. Full Comparison Script (CLI)

Run a comprehensive comparison of **all** available solvers for any domain:

```bash
# Unit disk - all methods (Analytical, FEM, both Linear and Nonlinear)
python -m inverse_source.cli compare --domain disk

# Ellipse - Conformal + FEM methods
python -m inverse_source.cli compare --domain ellipse --ellipse-a 2.0 --ellipse-b 1.0

# Star-shaped domain (conformal only)
python -m inverse_source.cli compare --domain star

# Square domain (FEM only)
python -m inverse_source.cli compare --domain square

# L-shaped polygon (FEM only)
python -m inverse_source.cli compare --domain polygon

# Custom polygon vertices
python -m inverse_source.cli compare --domain polygon \
    --vertices '[[0,0],[3,0],[3,1],[1,1],[1,3],[0,3]]'

# Quick mode (skip slower differential_evolution)
python -m inverse_source.cli compare --domain ellipse --quick
```

**Methods run per domain:**

| Domain | Linear Methods | Nonlinear Methods |
|--------|---------------|-------------------|
| **disk** | Analytical (L1,L2,TV) + FEM (L1,L2,TV) | Analytical + FEM |
| **ellipse** | Conformal (L1,L2,TV) + FEM (L1,L2,TV) | Conformal + FEM |
| **star** | Conformal (L1,L2,TV) | Conformal |
| **polygon/square** | FEM (L1,L2,TV) | FEM |

This generates:
- Console output with metrics for all solver combinations
- Visualization comparing linear and nonlinear approaches  
- Results saved to `results/run_{domain}_{hash}/`
- Performance timing information

### 9. Programmatic Comparison

```python
from inverse_source import compare_all_solvers_general, create_domain_sources, plot_comparison

# Run comparison for ellipse
domain_params = {'a': 2.0, 'b': 1.0}
sources = create_domain_sources('ellipse', domain_params)
results = compare_all_solvers_general(
    'ellipse', 
    domain_params=domain_params,
    sources_true=sources,
    noise_level=0.001,
    alpha=1e-4,
    quick=False,  # Include differential_evolution
    seed=42
)

# Print results
for r in results:
    print(f"{r.solver_name}: Pos RMSE={r.position_rmse:.4f}")

# Visualize
plot_comparison(results, sources, domain_type='ellipse', domain_params=domain_params)
```

## Key Findings

### Linear Solvers

| Regularization | Peaks | Clusters | Best For |
|---------------|-------|----------|----------|
| L1 | Many | Few, separated | Sparse reconstruction |
| L2 | Medium | Large, merged | Smooth reconstruction |
| TV | Medium | Large, merged | Edge-preserving |

**Important**: Linear solvers have inherent limitations due to the ill-posed nature of the inverse problem:
- Condition number of Green's matrix: ~10^15
- Total intensity is conserved but spread across the domain
- Best used for source **detection** rather than **quantification**

### Nonlinear Solvers

| Optimizer | Global Search | Speed | Recommended Use |
|-----------|--------------|-------|-----------------|
| `differential_evolution` | Yes | Slow | Best accuracy |
| `L-BFGS-B` | No | Fast | Good initial guess |
| `basin_hopping` | Yes | Medium | Balance |

### Solver Choice Guide

1. **Unknown number of sources** → Linear solver to detect approximate locations
2. **Known number of sources** → Nonlinear solver for precise recovery
3. **Real-time requirements** → Analytical with L-BFGS-B
4. **Complex geometry** → FEM or Conformal mapping
5. **Highest accuracy** → Nonlinear with differential_evolution

## File Structure

```
inverse_source/
├── analytical_solver.py   # Closed-form Green's function solvers
├── bem_solver.py          # Boundary element method (numerical integration)
├── fem_solver.py          # Finite element method (FEniCSx)
├── conformal_solver.py    # Conformal mapping for general domains
├── comparison.py          # Solver comparison and visualization
├── parameter_study.py     # Regularization parameter analysis
├── mesh.py                # Mesh generation utilities
├── regularization.py      # L1, L2, TV regularization
├── config.py              # JSON/YAML configuration system
├── cli.py                 # Command-line interface
└── utils.py               # Plotting and I/O utilities
```

## Configuration

Solvers can be configured via JSON:

```json
{
  "method": "analytical",
  "n_boundary": 100,
  "n_sources": 4,
  "optimizer": "differential_evolution",
  "regularization": {
    "type": "l1",
    "alpha": 1e-4
  }
}
```

Or YAML:

```yaml
method: analytical
n_boundary: 100
n_sources: 4
optimizer: differential_evolution
regularization:
  type: l1
  alpha: 0.0001
```

## Mathematical Background

The forward problem solves the Poisson equation with Neumann boundary conditions:

```
-Δu = Σ qₖ δ(x - ξₖ)   in Ω (unit disk)
∂u/∂n = 0              on ∂Ω
```

For the unit disk, the Green's function is derived using the method of images:

```
G(x, ξ) = -1/(2π) [ln|x - ξ| + ln|x - ξ*| - ln|ξ|]
```

where ξ* = ξ/|ξ|² is the Kelvin transform (image point).

See `docs/main.tex` for complete mathematical derivations with proper citations.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{inverse_source,
  title = {Inverse Source Localization from Boundary Measurements},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/inverse_source_project}
}
```

## License

MIT License - see LICENSE file for details.

## References

Key references are provided in `docs/references.bib`, including:

- Isakov (2006): Inverse Problems for Partial Differential Equations
- Ammari & Kang (2004): Reconstruction of Small Inhomogeneities
- El Badia & Ha-Duong (2000): Inverse Source Problem in Potential Analysis
- Jackson (1999): Classical Electrodynamics (method of images)

See the LaTeX documentation for complete citations.
