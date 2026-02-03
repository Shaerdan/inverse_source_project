"""
Inverse Source Localization Package
====================================

A toolkit for inverse source localization of acoustic point sources in 2D domains.

Solver Types
------------
1. **Analytical Solver** (Unit Disk)
   - Exact closed-form Neumann Green's function
   - Fastest for point sources on unit disk

2. **Conformal Solver** (General Domains) 
   - Maps general domains to unit disk via MFS conformal mapping
   - Supports: disk, ellipse, star, polygon, brain, custom domains
   - Recommended for most use cases

3. **FEM Solver** (Finite Element Method)
   - Mesh-based discretization via FEniCSx
   - Works for any domain geometry
   - Requires FEniCSx installation

4. **IPOPT Solver** (Nonlinear Optimization)
   - Professional nonlinear optimizer matching MATLAB fmincon
   - Requires cyipopt installation

Problem Formulations
--------------------
1. **Nonlinear (Continuous)**: Optimize source positions and intensities
   - Use for: source localization with unknown positions
   - Recommended approach

2. **Linear (Grid-based)**: Sources on fixed grid, solve for intensities
   - Use for: intensity field estimation
   - Note: fundamentally ill-posed for sparse recovery

Quick Start
-----------
>>> from inverse_source import ConformalForwardSolver, ConformalNonlinearInverseSolver
>>> from inverse_source import create_conformal_map
>>> 
>>> # Create conformal map for ellipse domain
>>> cmap = create_conformal_map('ellipse', a=1.5, b=0.8)
>>> 
>>> # Define true sources (intensities must sum to zero)
>>> sources = [
...     ((0.3, 0.2), 1.0),
...     ((-0.2, -0.3), -1.0),
... ]
>>> 
>>> # Forward solve
>>> forward = ConformalForwardSolver(cmap, n_sensors=100)
>>> u_measured = forward.solve(sources)
>>> 
>>> # Inverse solve
>>> inverse = ConformalNonlinearInverseSolver(cmap, n_sources=2, n_sensors=100)
>>> result = inverse.solve(u_measured)

For Disk Domain (Faster)
------------------------
>>> from inverse_source import AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver
>>> 
>>> forward = AnalyticalForwardSolver(n_boundary_points=100)
>>> u_measured = forward.solve(sources)
>>> 
>>> inverse = AnalyticalNonlinearInverseSolver(n_sources=2, n_sensors=100)
>>> result = inverse.solve(u_measured)

Theoretical Bound
-----------------
The maximum number of recoverable sources is bounded by:

    N_max = (2/3) * log(σ_Four) / log(ρ_min)

where:
- σ_Four = σ_noise / √M is the Fourier-domain noise level
- ρ_min is the minimum conformal radius of the sources
- M is the number of sensors

See THEORY_FRAMEWORK.md for details.

Authors
-------
Serdan (https://github.com/Shaerdan/inverse_source_project)

License
-------
MIT License
"""

__version__ = "0.8.0"
__author__ = "Serdan"

# =============================================================================
# CONFORMAL SOLVER (General Domains - Recommended)
# =============================================================================
from .conformal_solver import (
    # Conformal maps
    MFSConformalMap,
    ConformalMap,
    DiskMap,
    EllipseMap,
    RectangleMap,
    PolygonMap,
    NumericalConformalMap,
    create_conformal_map,
    # Forward solver
    ConformalForwardSolver,
    # Inverse solvers
    ConformalNonlinearInverseSolver,
    ConformalLinearInverseSolver,
    # Utilities
    solve_forward_conformal,
)

# =============================================================================
# ANALYTICAL SOLVER (Unit Disk - Fastest)
# =============================================================================
from .analytical_solver import (
    # Forward
    AnalyticalForwardSolver,
    # Inverse - Nonlinear (recommended)
    AnalyticalNonlinearInverseSolver,
    # Inverse - Linear (grid-based, for comparison)
    AnalyticalLinearInverseSolver,
    # Green's function
    greens_function_disk_neumann,
    greens_function_disk_neumann_gradient,
    # Utilities
    generate_synthetic_data,
    Source,
    InverseResult,
)

# =============================================================================
# FEM SOLVER (Requires FEniCSx)
# =============================================================================
try:
    from .fem_solver import (
        FEMForwardSolver,
        FEMLinearInverseSolver,
        FEMNonlinearInverseSolver,
    )
    _FEM_AVAILABLE = True
except ImportError:
    _FEM_AVAILABLE = False

# =============================================================================
# IPOPT SOLVER (Requires cyipopt)
# =============================================================================
try:
    from .ipopt_solver import (
        IPOPTNonlinearInverseSolver,
        check_cyipopt_available,
        get_ipopt_version,
    )
    _IPOPT_AVAILABLE = True
except ImportError:
    _IPOPT_AVAILABLE = False

# =============================================================================
# DOMAINS
# =============================================================================
from .domains import (
    get_domain,
    DiskDomain,
    EllipseDomain,
    RectangleDomain,
    PolygonDomain,
    BrainDomain,
)

# =============================================================================
# MESH (Requires gmsh)
# =============================================================================
try:
    from .mesh import (
        create_disk_mesh,
        create_ellipse_mesh,
        create_rectangle_mesh,
        create_polygon_mesh,
        create_brain_mesh,
        get_brain_boundary,
    )
    _MESH_AVAILABLE = True
except ImportError:
    _MESH_AVAILABLE = False

# =============================================================================
# UTILITIES
# =============================================================================
from .utils import (
    compute_rmse,
    match_sources,
)

from .regularization import (
    solve_tikhonov,
    solve_l1,
    solve_tv,
)

# =============================================================================
# COMPARISON & CALIBRATION
# =============================================================================
from .comparison import (
    create_domain_sources,
    get_sensor_locations,
    run_comparison,
)

from .calibration import (
    create_well_separated_sources,
    create_clustered_sources,
)

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def check_dependencies():
    """Check which optional dependencies are available."""
    deps = {
        'numpy': True,
        'scipy': True,
        'matplotlib': True,
        'FEniCSx (fem_solver)': _FEM_AVAILABLE,
        'cyipopt (ipopt_solver)': _IPOPT_AVAILABLE,
        'gmsh (mesh)': _MESH_AVAILABLE,
    }
    
    print("Inverse Source Package - Dependency Check")
    print("=" * 45)
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")
    print()
    
    if not _FEM_AVAILABLE:
        print("Note: FEM solver requires FEniCSx.")
        print("      Install via: conda install -c conda-forge fenics-dolfinx")
    if not _IPOPT_AVAILABLE:
        print("Note: IPOPT solver requires cyipopt.")
        print("      Install via: conda install -c conda-forge cyipopt")
    
    return deps


# =============================================================================
# PUBLIC API
# =============================================================================
__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # Conformal solver (recommended)
    'MFSConformalMap',
    'ConformalMap', 
    'DiskMap',
    'EllipseMap',
    'RectangleMap',
    'PolygonMap',
    'NumericalConformalMap',
    'create_conformal_map',
    'ConformalForwardSolver',
    'ConformalNonlinearInverseSolver',
    'ConformalLinearInverseSolver',
    'solve_forward_conformal',
    
    # Analytical solver (disk only)
    'AnalyticalForwardSolver',
    'AnalyticalNonlinearInverseSolver',
    'AnalyticalLinearInverseSolver',
    'greens_function_disk_neumann',
    'greens_function_disk_neumann_gradient',
    'generate_synthetic_data',
    'Source',
    'InverseResult',
    
    # Domains
    'get_domain',
    'DiskDomain',
    'EllipseDomain',
    'RectangleDomain',
    'PolygonDomain',
    'BrainDomain',
    
    # Utilities
    'compute_rmse',
    'match_sources',
    'solve_tikhonov',
    'solve_l1',
    'solve_tv',
    
    # Comparison
    'create_domain_sources',
    'get_sensor_locations',
    'run_comparison',
    'create_well_separated_sources',
    'create_clustered_sources',
    
    # Check
    'check_dependencies',
]
