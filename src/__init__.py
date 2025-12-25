"""
Inverse Source Localization Package
====================================

A comprehensive toolkit for inverse source localization in 2D domains
using Boundary Element Methods (BEM) and Finite Element Methods (FEM).

Key Features:
- BEM with analytical Green's function for unit disk
- Conformal BEM for general simply connected domains (ellipse, star-shaped)
- Multiple regularization methods (L1, L2, TV)
- Chambolle-Pock and ADMM algorithms for TV regularization
- JSON-based configuration system
- Parameter sweep and L-curve analysis tools
- Command-line interface

Quick Start:
-----------
>>> from inverse_source_package import bem_solver
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
>>> # Inverse solve (nonlinear)
>>> inverse = bem_solver.BEMNonlinearInverseSolver(n_sources=2, n_boundary=100)
>>> inverse.set_measured_data(u_boundary)
>>> sources_recovered, result = inverse.solve()

Modules:
-------
- bem_solver: BEM forward and inverse solvers for unit disk
- conformal_bem: BEM for general domains via conformal mapping
- fem_solver: FEM-based solvers (fallback for complex domains)
- regularization: L1, L2, TV regularization with various algorithms
- parameter_study: Parameter sweeps and L-curve analysis
- config: JSON configuration management
- utils: Plotting and analysis utilities
- cli: Command-line interface

Authors:
-------
Serdan (https://github.com/Shaerdan/inverse_source_project)

License:
-------
MIT License
"""

__version__ = "1.0.0"
__author__ = "Serdan"

# Core solvers
from .bem_solver import (
    BEMForwardSolver,
    BEMLinearInverseSolver,
    BEMNonlinearInverseSolver,
    greens_function_disk_neumann,
)

from .conformal_bem import (
    ConformalBEMSolver,
    ConformalNonlinearInverse,
    ConformalLinearInverse,
    ConformalMap,
    DiskMap,
    EllipseMap,
    StarShapedMap,
    greens_function_disk,
)

# Regularization methods
from .regularization import (
    solve_l1,
    solve_l2,
    solve_tv_chambolle_pock,
    solve_tv_admm,
    solve_regularized,
    build_gradient_operator,
    RegularizationResult,
)

# Configuration
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

# Utilities
from .utils import (
    plot_sources,
    plot_boundary_data,
    plot_recovery_comparison,
    compute_source_error,
    l_curve_analysis,
    plot_l_curve,
    create_test_sources,
)

# Parameter study
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

# Convenience imports
from . import bem_solver
from . import conformal_bem
from . import fem_solver
from . import regularization
from . import parameter_study
from . import config
from . import utils

__all__ = [
    '__version__',
    '__author__',
    'BEMForwardSolver',
    'BEMLinearInverseSolver', 
    'BEMNonlinearInverseSolver',
    'greens_function_disk_neumann',
    'ConformalBEMSolver',
    'ConformalNonlinearInverse',
    'ConformalLinearInverse',
    'ConformalMap',
    'DiskMap',
    'EllipseMap',
    'StarShapedMap',
    'greens_function_disk',
    'solve_l1',
    'solve_l2',
    'solve_tv_chambolle_pock',
    'solve_tv_admm',
    'solve_regularized',
    'build_gradient_operator',
    'RegularizationResult',
    'Config',
    'get_config',
    'get_template',
    'TEMPLATES',
    'plot_sources',
    'plot_boundary_data',
    'plot_recovery_comparison',
    'compute_source_error',
    'l_curve_analysis',
    'plot_l_curve',
    'create_test_sources',
    'parameter_sweep',
    'find_l_curve_corner',
    'compare_methods',
    'SweepResult',
    'bem_solver',
    'conformal_bem',
    'fem_solver',
    'regularization',
    'parameter_study',
    'config',
    'utils',
]
