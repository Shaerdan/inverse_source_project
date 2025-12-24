# Inverse Source Localization Project

A Python package for solving inverse source problems in PDEs using FEniCSx.

## Problem Description

**Forward Problem**: Given source locations and intensities, solve the Poisson equation:
```
-Δu = f    in Ω (unit disk)
∂u/∂n = 0  on ∂Ω (zero Neumann BC)
```
where `f = Σ qᵢ δ(x - xᵢ)` is a sum of point sources.

**Inverse Problem** (Goal): Given boundary measurements `u|∂Ω`, recover the source locations `{xᵢ}` and intensities `{qᵢ}`.

## Project Structure

```
inverse_source_project/
├── environment.yml         # Conda environment file
├── requirements.txt        # pip requirements (limited)
├── README.md
├── src/
│   ├── __init__.py
│   ├── forward_solver.py   # Main forward problem solver
│   └── mesh_utils.py       # Mesh generation utilities
├── meshes/                 # Mesh files (.msh, .xdmf)
├── results/                # Output plots and data
└── notebooks/              # Jupyter notebooks for experiments
```

## Installation

### Option 1: Conda (Recommended)

This is the most reliable method for FEniCSx:

```bash
# Clone/download the project
cd inverse_source_project

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate inverse_source

# Verify installation
python -c "import dolfinx; print(f'FEniCSx version: {dolfinx.__version__}')"
```

### Option 2: Using Mamba (Faster)

```bash
# Install mamba if not available
conda install -c conda-forge mamba

# Create environment with mamba (faster than conda)
mamba env create -f environment.yml
conda activate inverse_source
```

### PyCharm Configuration

1. **Create Project**: Open PyCharm → File → Open → Select `inverse_source_project`

2. **Configure Interpreter**:
   - File → Settings → Project → Python Interpreter
   - Click gear icon → Add → Conda Environment
   - Select "Existing environment"
   - Choose: `~/anaconda3/envs/inverse_source/bin/python` (or similar path)

3. **Mark Source Root**:
   - Right-click `src/` folder → Mark Directory as → Sources Root

4. **Run Configuration**:
   - Run → Edit Configurations → Add → Python
   - Script: `src/forward_solver.py`
   - Working directory: Project root

## Quick Start

```python
from src.forward_solver import (
    create_disk_mesh, 
    solve_poisson_zero_neumann,
    get_boundary_values,
    plot_solution
)

# Define sources: ((x, y), intensity)
sources = [
    ((-0.3, 0.4), 1.0),    # Positive source
    ((0.5, 0.2), 1.0),     # Positive source
    ((-0.5, -0.3), -1.0),  # Sink
    ((0.3, -0.5), -1.0),   # Sink
]

# Create mesh
mesh, _, _ = create_disk_mesh(radius=1.0, resolution=0.05)

# Solve forward problem
u = solve_poisson_zero_neumann(mesh, sources)

# Get boundary measurements (for inverse problem)
angles, boundary_values = get_boundary_values(u)

# Visualize
plot_solution(u, sources)
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `fenics-dolfinx` | FEM solver (modern FEniCS) |
| `gmsh` | Mesh generation |
| `petsc4py` | Linear algebra backend |
| `mpi4py` | Parallel computing |
| `pyvista` | 3D visualization |
| `numpy`, `scipy` | Scientific computing |
| `matplotlib` | Plotting |

## API Reference

### Forward Solver

```python
solve_poisson_zero_neumann(mesh, sources, polynomial_degree=1)
```
Solves the Poisson equation with point sources and zero Neumann BC.

**Parameters:**
- `mesh`: dolfinx.mesh.Mesh - The computational domain
- `sources`: list of ((x, y), intensity) - Source locations and strengths
- `polynomial_degree`: int - FE polynomial order (default: 1)

**Returns:**
- `u`: dolfinx.fem.Function - The solution function

### Mesh Utilities

```python
create_disk_mesh(radius=1.0, resolution=0.05)
load_mesh_from_file(filepath)
save_mesh_to_file(mesh, filepath)
```

## Notes on the Physics

### Compatibility Condition
For pure Neumann problems, the total source intensity must be zero:
```
∫_Ω f dx = 0  ⟹  Σ qᵢ = 0
```
The solver automatically enforces mean-zero solution for uniqueness.

### Why Point Sources?
Point sources (delta functions) model localized phenomena like:
- Heat sources/sinks
- Pollutant releases
- Seismic events

### Inverse Problem Strategy (Future Work)
Common approaches include:
1. **Optimization-based**: Minimize `||u_measured - u_computed||²`
2. **MUSIC algorithm**: Spectral methods for source detection
3. **Bayesian inference**: Probabilistic source reconstruction
4. **Neural networks**: Deep learning approaches

## Troubleshooting

### "No module named 'dolfinx'"
FEniCSx isn't installed. Use conda installation method.

### MPI Errors
Ensure `mpi4py` is installed via conda (not pip):
```bash
conda install -c conda-forge mpi4py mpich
```

### Mesh Loading Errors
Check Gmsh version compatibility. Use Gmsh format 2.2 or 4.1.

## License

MIT License

## References

1. FEniCSx Documentation: https://docs.fenicsproject.org/
2. Gmsh Documentation: https://gmsh.info/doc/texinfo/gmsh.html
3. Inverse Problems in PDEs (textbooks by Isakov, Kirsch, etc.)
