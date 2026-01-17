"""
Inverse Source Visualization Toolbox
====================================

Comprehensive visualization tools for inverse source localization.

This toolbox provides diagnostic visualizations for:
- Forward problem solutions
- Inverse problem source recovery
- Optimization convergence and trajectories
- Barrier function analysis
- Conformal mapping
- Green's function analysis
- Animated optimization

Quick Start
-----------
    from visualization import diagnostic_dashboard, plot_convergence

    # After running solver...
    fig = diagnostic_dashboard(
        sources_true, sources_recovered,
        u_measured, u_recovered, theta,
        domain_boundary, history
    )
    fig.savefig('diagnostic.png')

Modules
-------
forward_viz
    Forward problem visualizations (boundary values, interior solution)
inverse_viz
    Inverse problem visualizations (source recovery, boundary fit)
optimization_viz
    Optimization diagnostics (convergence, trajectories)
barrier_viz
    Barrier function visualizations
conformal_viz
    Conformal mapping visualizations
greens_viz
    Green's function visualizations
animation
    Animated visualizations
dashboard
    Combined diagnostic dashboards
config
    Styling configuration
utils
    Utility functions
"""

# Configuration and styling
from .config import (
    COLORS,
    MARKERS,
    FIGSIZE,
    COLORMAPS,
    LINESTYLES,
    get_source_color,
    get_source_marker,
    get_trajectory_color,
    apply_style,
    reset_style,
    viz_style,
)

# Utility functions
from .utils import (
    get_domain_boundary,
    domain_mask,
    point_in_domain,
    get_bounding_box,
    sources_to_arrays,
    match_sources,
    compute_source_errors,
    add_domain_boundary,
    add_source_markers,
    add_sensor_markers,
    add_colorbar,
    set_domain_axes,
    add_matching_arrows,
    format_axes_pi,
    create_figure,
)

# Forward problem visualizations
from .forward_viz import (
    plot_boundary_values,
    plot_interior_solution,
    plot_source_configuration,
    plot_boundary_comparison,
    plot_forward_diagnostics,
)

# Inverse problem visualizations
from .inverse_viz import (
    plot_boundary_fit,
    plot_source_recovery,
    plot_linear_solution,
    plot_linear_methods_comparison,
    plot_inverse_residual_map,
    plot_source_intensity_bar,
)

# Optimization visualizations
from .optimization_viz import (
    plot_convergence,
    plot_multistart_convergence,
    plot_source_trajectory,
    plot_parameter_evolution,
    plot_optimization_summary,
    plot_gradient_norms,
)

# Barrier function visualizations
from .barrier_viz import (
    plot_barrier_landscape,
    plot_objective_slice,
    plot_barrier_radial_profile,
    plot_constraint_violation,
    plot_barrier_diagnostics,
)

# Conformal mapping visualizations
from .conformal_viz import (
    plot_domain_correspondence,
    plot_mapping_jacobian,
    plot_boundary_correspondence,
    plot_conformal_distortion,
    plot_source_mapping,
)

# Green's function visualizations
from .greens_viz import (
    plot_greens_function,
    plot_greens_boundary,
    plot_greens_matrix,
    plot_greens_function_comparison,
)

# Animations
from .animation import (
    animate_optimization,
    animate_source_movement,
    animate_boundary_fit,
    create_optimization_video,
)

# Dashboards
from .dashboard import (
    diagnostic_dashboard,
    solver_comparison_dashboard,
    domain_comparison_dashboard,
    quick_diagnostic,
)

# Apply default style on import
apply_style()

# Version
__version__ = '1.0.0'

# All public exports
__all__ = [
    # Config
    'COLORS',
    'MARKERS',
    'FIGSIZE',
    'COLORMAPS',
    'LINESTYLES',
    'get_source_color',
    'get_source_marker',
    'get_trajectory_color',
    'apply_style',
    'reset_style',
    'viz_style',

    # Utils
    'get_domain_boundary',
    'domain_mask',
    'point_in_domain',
    'get_bounding_box',
    'sources_to_arrays',
    'match_sources',
    'compute_source_errors',
    'add_domain_boundary',
    'add_source_markers',
    'add_sensor_markers',
    'add_colorbar',
    'set_domain_axes',
    'add_matching_arrows',
    'format_axes_pi',
    'create_figure',

    # Forward viz
    'plot_boundary_values',
    'plot_interior_solution',
    'plot_source_configuration',
    'plot_boundary_comparison',
    'plot_forward_diagnostics',

    # Inverse viz
    'plot_boundary_fit',
    'plot_source_recovery',
    'plot_linear_solution',
    'plot_linear_methods_comparison',
    'plot_inverse_residual_map',
    'plot_source_intensity_bar',

    # Optimization viz
    'plot_convergence',
    'plot_multistart_convergence',
    'plot_source_trajectory',
    'plot_parameter_evolution',
    'plot_optimization_summary',
    'plot_gradient_norms',

    # Barrier viz
    'plot_barrier_landscape',
    'plot_objective_slice',
    'plot_barrier_radial_profile',
    'plot_constraint_violation',
    'plot_barrier_diagnostics',

    # Conformal viz
    'plot_domain_correspondence',
    'plot_mapping_jacobian',
    'plot_boundary_correspondence',
    'plot_conformal_distortion',
    'plot_source_mapping',

    # Green's viz
    'plot_greens_function',
    'plot_greens_boundary',
    'plot_greens_matrix',
    'plot_greens_function_comparison',

    # Animation
    'animate_optimization',
    'animate_source_movement',
    'animate_boundary_fit',
    'create_optimization_video',

    # Dashboard
    'diagnostic_dashboard',
    'solver_comparison_dashboard',
    'domain_comparison_dashboard',
    'quick_diagnostic',
]
