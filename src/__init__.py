"""
Inverse Source Localization Package
====================================

A comprehensive toolkit for inverse source localization in 2D domains.

Solver Types
------------
1. **Analytical Solver** (Unit Disk Only)
   - Uses exact closed-form Neumann Green's function
   - Fastest for point sources on unit disk
   - No discretization error in forward problem
   
2. **BEM Solver** (Boundary Element Method)
   - Numerical integration for boundary integrals
   - Handles distributed sources
   - Validates analytical solver

3. **FEM Solver** (Finite Element Method)
   - Mesh-based discretization
   - Works for any domain geometry
   - Handles general source distributions

4. **Conformal Solver** (General Domains)
   - Maps general domains to unit disk
   - Uses analytical solution after mapping
   - Supports ellipse, star-shaped domains

Problem Formulations
--------------------
1. **Linear (Distributional)**: Sources on fixed grid, solve for intensities
2. **Nonlinear (Continuous)**: Optimize source positions and intensities

Quick Start
-----------
>>> from inverse_source import AnalyticalForwardSolver, AnalyticalLinearInverseSolver
>>> import numpy as np
>>> 
>>> # Define sources (must sum to zero for Neumann BC)
>>> sources = [
...     ((-0.3, 0.4), 1.0),
...     ((0.5, -0.3), -1.0),
... ]
>>> 
>>> # Forward solve
>>> forward = AnalyticalForwardSolver(n_boundary_points=100)
>>> u_boundary = forward.solve(sources)
>>> 
>>> # Linear inverse with L1 regularization
>>> inverse = AnalyticalLinearInverseSolver(n_boundary=100)
>>> inverse.build_greens_matrix()
>>> q_recovered = inverse.solve_l1(u_boundary, alpha=1e-4)

Authors
-------
Serdan (https://github.com/Shaerdan/inverse_source_project)

License
-------
MIT License
"""

__version__ = "2.0.0"
__author__ = "Serdan"

# =============================================================================
# ANALYTICAL SOLVER (Exact solution for unit disk - formerly mislabeled as "BEM")
# =============================================================================
from .analytical_solver import (
    # Forward
    AnalyticalForwardSolver,
    # Inverse - Linear (grid-based)
    AnalyticalLinearInverseSolver,
    # Inverse - Nonlinear (continuous positions)
    AnalyticalNonlinearInverseSolver,
    # Green's function
    greens_function_disk_neumann,
    greens_function_disk_neumann_gradient,
    # Utilities
    generate_synthetic_data,
    Source,
    InverseResult,
    # Backward-compatible aliases (deprecated)
    BEMForwardSolver,  # -> AnalyticalForwardSolver
    BEMLinearInverseSolver,  # -> AnalyticalLinearInverseSolver
    BEMNonlinearInverseSolver,  # -> AnalyticalNonlinearInverseSolver
)

# =============================================================================
# TRUE BEM SOLVER (Numerical boundary integration)
# =============================================================================
from .bem_solver import (
    BEMForwardSolver as BEMNumericalForwardSolver,
    BEMLinearInverseSolver as BEMNumericalLinearInverseSolver,
    BEMDiscretization,
    BEMResult,
    fundamental_solution_2d,
    fundamental_solution_gradient_2d,
    validate_against_analytical,
    compare_bem_analytical,
)

# =============================================================================
# MESH GENERATION (shared by FEM and linear solvers)
# =============================================================================
from .mesh import (
    create_disk_mesh,
    get_source_grid,
)

# =============================================================================
# FEM SOLVER (Finite Element Method)
# =============================================================================
from .fem_solver import (
    # Forward
    FEMForwardSolver,
    # Inverse - Linear (grid-based)
    FEMLinearInverseSolver,
    # Inverse - Nonlinear (continuous positions)
    FEMNonlinearInverseSolver,
    # Low-level solve
    solve_poisson,
    generate_synthetic_data_fem,
)

# =============================================================================
# CONFORMAL MAPPING SOLVER (General domains)
# =============================================================================
from .conformal_solver import (
    # Maps
    ConformalMap,
    DiskMap,
    EllipseMap,
    StarShapedMap,
    # Solvers
    ConformalForwardSolver,
    ConformalLinearInverseSolver,
    ConformalNonlinearInverseSolver,
    # Convenience
    create_ellipse_solver,
    create_star_solver,
    # Backward-compatible aliases
    ConformalBEMSolver,  # -> ConformalForwardSolver
    ConformalNonlinearInverse,  # -> ConformalNonlinearInverseSolver
    ConformalLinearInverse,  # -> ConformalLinearInverseSolver
)

# =============================================================================
# REGULARIZATION METHODS
# =============================================================================
from .regularization import (
    solve_l1,
    solve_l2,
    solve_tv_chambolle_pock,
    solve_tv_admm,
    solve_regularized,
    build_gradient_operator,
    RegularizationResult,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
from .config import (
    Config,
    ForwardConfig,
    InverseConfig,
    GridConfig,
    TVConfig,
    VisualizationConfig,
    get_config,
    get_template,
    TEMPLATES,
)

# =============================================================================
# UTILITIES
# =============================================================================
from .utils import (
    plot_sources,
    plot_boundary_data,
    plot_recovery_comparison,
    compute_source_error,
    l_curve_analysis,
    plot_l_curve,
    create_test_sources,
)

# =============================================================================
# PARAMETER STUDY
# =============================================================================
from .parameter_study import (
    parameter_sweep,
    find_l_curve_corner,
    compare_methods,
    plot_l_curve_comparison,
    plot_method_comparison,
    noise_study,
    plot_noise_study,
    SweepResult,
)

# =============================================================================
# COMPARISON (all solver types)
# =============================================================================
from .comparison import (
    compare_all_solvers,
    print_comparison_table,
    plot_comparison,
    ComparisonResult,
)

# =============================================================================
# MODULE IMPORTS (for inverse_source.module style access)
# =============================================================================
from . import mesh
from . import analytical_solver
from . import bem_solver
from . import conformal_solver
from . import fem_solver
from . import regularization
from . import parameter_study
from . import config
from . import utils
from . import comparison

# =============================================================================
# PUBLIC API
# =============================================================================
__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # === ANALYTICAL SOLVER (recommended for unit disk) ===
    'AnalyticalForwardSolver',
    'AnalyticalLinearInverseSolver', 
    'AnalyticalNonlinearInverseSolver',
    'greens_function_disk_neumann',
    'greens_function_disk_neumann_gradient',
    'generate_synthetic_data',
    
    # === TRUE BEM SOLVER ===
    'BEMNumericalForwardSolver',
    'BEMNumericalLinearInverseSolver',
    'BEMDiscretization',
    'BEMResult',
    'fundamental_solution_2d',
    'validate_against_analytical',
    'compare_bem_analytical',
    
    # === MESH ===
    'create_disk_mesh',
    'get_source_grid',
    
    # === FEM SOLVER ===
    'FEMForwardSolver',
    'FEMLinearInverseSolver',
    'FEMNonlinearInverseSolver',
    'solve_poisson',
    'generate_synthetic_data_fem',
    
    # === CONFORMAL SOLVER ===
    'ConformalMap',
    'DiskMap',
    'EllipseMap', 
    'StarShapedMap',
    'ConformalForwardSolver',
    'ConformalLinearInverseSolver',
    'ConformalNonlinearInverseSolver',
    'create_ellipse_solver',
    'create_star_solver',
    
    # === DATA CLASSES ===
    'Source',
    'InverseResult',
    
    # === REGULARIZATION ===
    'solve_l1',
    'solve_l2',
    'solve_tv_chambolle_pock',
    'solve_tv_admm',
    'solve_regularized',
    'build_gradient_operator',
    'RegularizationResult',
    
    # === CONFIG ===
    'Config',
    'get_config',
    'get_template',
    'TEMPLATES',
    
    # === UTILS ===
    'plot_sources',
    'plot_boundary_data',
    'plot_recovery_comparison',
    'compute_source_error',
    'l_curve_analysis',
    'plot_l_curve',
    'create_test_sources',
    
    # === PARAMETER STUDY ===
    'parameter_sweep',
    'find_l_curve_corner',
    'compare_methods',
    'SweepResult',
    
    # === COMPARISON ===
    'compare_all_solvers',
    'print_comparison_table',
    'plot_comparison',
    'ComparisonResult',
    
    # === MODULES ===
    'mesh',
    'analytical_solver',
    'bem_solver',
    'conformal_solver',
    'fem_solver',
    'regularization',
    'parameter_study',
    'config',
    'utils',
    'comparison',
    
    # === BACKWARD-COMPATIBLE ALIASES (deprecated) ===
    'BEMForwardSolver',  # Use AnalyticalForwardSolver
    'BEMLinearInverseSolver',  # Use AnalyticalLinearInverseSolver
    'BEMNonlinearInverseSolver',  # Use AnalyticalNonlinearInverseSolver
    'ConformalBEMSolver',  # Use ConformalForwardSolver
    'ConformalNonlinearInverse',  # Use ConformalNonlinearInverseSolver
    'ConformalLinearInverse',  # Use ConformalLinearInverseSolver
]
