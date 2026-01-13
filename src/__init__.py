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

__version__ = "7.30.0"  # Fixed source generation: r in [0.5, 0.85] for better conditioning, well-separated test sources
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
    BEMNonlinearInverseSolver as BEMNumericalNonlinearInverseSolver,
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
    create_ellipse_mesh,
    get_ellipse_source_grid,
    create_polygon_mesh,
    get_polygon_source_grid,
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
    # Base class
    ConformalMap,
    # Specific maps
    DiskMap,
    EllipseMap,
    RectangleMap,
    PolygonMap,
    NumericalConformalMap,
    # Solvers
    ConformalForwardSolver,
    ConformalLinearInverseSolver,
    ConformalNonlinearInverseSolver,
    # Factory functions
    create_conformal_map,
    solve_forward_conformal,
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
# PARAMETER SELECTION (L-curve, discrepancy principle)
# =============================================================================
from .parameter_selection import (
    estimate_alpha,
    find_lcurve_corner,
    find_discrepancy_alpha,
    parameter_sweep as param_sweep_lcurve,  # Renamed to avoid conflict
    run_full_comparison,
    plot_parameter_sweep,
    plot_solutions_comparison,
    ParameterSweepResult,
    # New proper metrics (not threshold-dependent)
    localization_score,
    sparsity_ratio,
    intensity_weighted_centroid,
)

# =============================================================================
# COMPARISON (all solver types)
# =============================================================================
from .comparison import (
    compare_all_solvers,
    compare_all_solvers_general,
    run_domain_comparison,
    create_domain_sources,
    print_comparison_table,
    plot_comparison,
    ComparisonResult,
    # Individual solver runners
    run_bem_linear,
    run_bem_nonlinear,
    run_fem_linear,
    run_fem_nonlinear,
    run_conformal_linear,
    run_conformal_nonlinear,
    run_fem_polygon_linear,
    run_fem_polygon_nonlinear,
    run_fem_ellipse_linear,
    run_fem_ellipse_nonlinear,
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
from . import parameter_selection
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
    'create_ellipse_mesh',
    'get_ellipse_source_grid',
    'create_polygon_mesh',
    'get_polygon_source_grid',
    
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
    'compare_all_solvers_general',
    'run_domain_comparison',
    'create_domain_sources',
    'print_comparison_table',
    'plot_comparison',
    'ComparisonResult',
    'run_bem_linear',
    'run_bem_nonlinear',
    'run_fem_linear',
    'run_fem_nonlinear',
    'run_conformal_linear',
    'run_conformal_nonlinear',
    'run_fem_polygon_linear',
    'run_fem_polygon_nonlinear',
    'run_fem_ellipse_linear',
    'run_fem_ellipse_nonlinear',
    
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

# Mesh convergence study
from .mesh_convergence import (
    run_forward_mesh_convergence,
    run_inverse_source_grid_convergence,
    run_full_convergence_study,
    ConvergenceStudy,
    ConvergenceResult
)


# Calibration module
from .calibration import (
    calibrate_domain,
    calibrate_all_domains,
    load_calibration_config,
    save_calibration_config,
    get_domain_params,
    plot_calibration_results,
    DomainCalibration,
    CalibrationConfig
)

# Reference solution generation
from .reference_solution import (
    get_reference_solution,
    generate_measurement_data,
    verify_fem_convergence
)

# Experiment tracking
from .experiment_tracker import (
    ExperimentTracker,
    CalibrationTracker,
    ExperimentDatabase,
    ExperimentConfig,
    ExperimentMetrics,
    get_timestamp,
    get_short_hash,
    list_experiments,
    get_experiment_details
)

# Mesh saving functions
from .mesh import (
    save_mesh_npz,
    load_mesh_npz,
    save_mesh_msh,
    save_source_grid_npz,
    load_source_grid_npz,
    save_source_grid_msh,
    save_meshes
)
