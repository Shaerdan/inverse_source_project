"""
Inverse Source Localization Package
====================================

A comprehensive toolkit for inverse source localization in 2D domains
using Boundary Element Methods (BEM) and Finite Element Methods (FEM).

Formulations
------------
1. **Linear (Distributional)**: Sources fixed to grid, solve for intensities
   - BEMLinearInverseSolver
   - FEMLinearInverseSolver
   
2. **Nonlinear (Continuous)**: Optimize source positions and intensities
   - BEMNonlinearInverseSolver  
   - FEMNonlinearInverseSolver

Quick Start
-----------
>>> from inverse_source import bem_solver
>>> 
>>> # Create sources
>>> sources = [
...     ((-0.3, 0.4), 1.0),
...     ((0.5, -0.3), -1.0),
... ]
>>> 
>>> # Forward solve
>>> forward = bem_solver.BEMForwardSolver(n_boundary_points=100)
>>> u_boundary = forward.solve(sources)
>>> 
>>> # Inverse solve (nonlinear - continuous positions)
>>> inverse = bem_solver.BEMNonlinearInverseSolver(n_sources=2, n_boundary=100)
>>> inverse.set_measured_data(u_boundary)
>>> result = inverse.solve()
>>> 
>>> # Or linear inverse (grid-based)
>>> linear = bem_solver.BEMLinearInverseSolver(n_boundary=100)
>>> q = linear.solve_l1(u_boundary, alpha=1e-4)

Authors
-------
Serdan (https://github.com/Shaerdan/inverse_source_project)

License
-------
MIT License
"""

__version__ = "1.0.0"
__author__ = "Serdan"

# =============================================================================
# BEM SOLVERS (Analytical Green's function - mesh-free)
# =============================================================================
from .bem_solver import (
    # Forward
    BEMForwardSolver,
    # Inverse - Linear (grid-based)
    BEMLinearInverseSolver,
    # Inverse - Nonlinear (continuous positions)
    BEMNonlinearInverseSolver,
    # Utilities
    greens_function_disk_neumann,
    generate_synthetic_data,
    Source,
    InverseResult,
)

# =============================================================================
# MESH GENERATION (shared by FEM and BEM linear)
# =============================================================================
from .mesh import (
    create_disk_mesh,
    get_source_grid,
)

# =============================================================================
# FEM SOLVERS (Finite Element - mesh-based)
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
# CONFORMAL BEM (General domains via conformal mapping)
# =============================================================================
from .conformal_bem import (
    ConformalBEMSolver,
    ConformalNonlinearInverse,
    ConformalLinearInverse,
    ConformalMap,
    DiskMap,
    EllipseMap,
    StarShapedMap,
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
# MODULE IMPORTS (for inverse_source.bem_solver style access)
# =============================================================================
from . import mesh
from . import bem_solver
from . import conformal_bem
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
    
    # BEM
    'BEMForwardSolver',
    'BEMLinearInverseSolver', 
    'BEMNonlinearInverseSolver',
    'greens_function_disk_neumann',
    'generate_synthetic_data',
    
    # Mesh
    'create_disk_mesh',
    'get_source_grid',
    
    # FEM
    'FEMForwardSolver',
    'FEMLinearInverseSolver',
    'FEMNonlinearInverseSolver',
    'solve_poisson',
    'generate_synthetic_data_fem',
    
    # Conformal
    'ConformalBEMSolver',
    'ConformalNonlinearInverse',
    'ConformalLinearInverse',
    'ConformalMap',
    'DiskMap',
    'EllipseMap',
    'StarShapedMap',
    
    # Data classes
    'Source',
    'InverseResult',
    
    # Regularization
    'solve_l1',
    'solve_l2',
    'solve_tv_chambolle_pock',
    'solve_tv_admm',
    'solve_regularized',
    'build_gradient_operator',
    'RegularizationResult',
    
    # Config
    'Config',
    'get_config',
    'get_template',
    'TEMPLATES',
    
    # Utils
    'plot_sources',
    'plot_boundary_data',
    'plot_recovery_comparison',
    'compute_source_error',
    'l_curve_analysis',
    'plot_l_curve',
    'create_test_sources',
    
    # Parameter study
    'parameter_sweep',
    'find_l_curve_corner',
    'compare_methods',
    'SweepResult',
    
    # Comparison
    'compare_all_solvers',
    'print_comparison_table',
    'plot_comparison',
    'ComparisonResult',
    
    # Modules
    'mesh',
    'bem_solver',
    'conformal_bem',
    'fem_solver',
    'regularization',
    'parameter_study',
    'config',
    'utils',
    'comparison',
]
