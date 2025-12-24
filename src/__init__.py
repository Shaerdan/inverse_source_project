"""
Inverse Source Localization Package
====================================
A package for solving inverse source problems using FEniCSx.

Modules:
- forward_solver: Solve Poisson equation with point sources
- mesh_utils: Mesh generation and manipulation utilities
"""

from .forward_solver import (
    solve_poisson_zero_neumann,
    create_disk_mesh,
    get_boundary_values,
    plot_solution,
    plot_boundary_solution,
)

from .mesh_utils import (
    load_mesh_from_file,
    save_mesh_to_file,
    get_mesh_info,
    plot_mesh,
)

__version__ = "0.1.0"
__author__ = "Your Name"
