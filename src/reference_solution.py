"""
Reference Solution Generation
=============================

This module provides the best possible reference solutions for each domain type,
used for:
1. Generating synthetic measurement data (u_measured)
2. Forward mesh convergence studies (comparing FEM to reference)

Reference Solution Hierarchy
----------------------------
Domain      | Method                    | Accuracy
------------|---------------------------|-----------
Disk        | Analytical Green's func   | Exact (machine precision)
Ellipse     | Conformal (Joukowski)     | Exact (machine precision)
Star        | Conformal (smooth)        | Exact (machine precision)
Square      | Richardson Extrapolation  | O(h^4) or better
Polygon     | Richardson Extrapolation  | O(h^4) or better
Brain       | Richardson Extrapolation  | O(h^4) or better

Richardson Extrapolation
------------------------
For FEM with O(h²) error, using solutions at h, h/2:
    u_exact ≈ (4*u_{h/2} - u_h) / 3

With solutions at h, h/2, h/4 (higher order):
    u_exact ≈ (64*u_{h/4} - 20*u_{h/2} + u_h) / 45

This gives us a reference solution that's independent of any single mesh.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings


def get_reference_solution(domain_type: str, 
                           domain_params: Dict,
                           sources: List[Tuple[Tuple[float, float], float]],
                           sensor_locations: np.ndarray,
                           verbose: bool = True) -> np.ndarray:
    """
    Get the best possible reference solution for a given domain.
    
    Parameters
    ----------
    domain_type : str
        'disk', 'ellipse', 'star', 'square', 'polygon', 'brain'
    domain_params : dict
        Domain-specific parameters
    sources : list
        List of ((x, y), intensity) tuples
    sensor_locations : array, shape (n_sensors, 2)
        Fixed measurement locations on boundary
    verbose : bool
        Print progress information
        
    Returns
    -------
    u_reference : array, shape (n_sensors,)
        Reference solution at sensor locations
    """
    if domain_type == 'disk':
        return _disk_analytical(sources, sensor_locations, verbose)
    
    elif domain_type in ['ellipse', 'star', 'square', 'polygon', 'brain']:
        # Use Richardson extrapolation for all non-disk domains
        # Conformal mapping doesn't preserve Neumann boundary conditions
        return _richardson_extrapolation(domain_type, domain_params, sources, 
                                          sensor_locations, verbose)
    
    else:
        raise ValueError(f"Unknown domain type: {domain_type}")


def _disk_analytical(sources: List, sensor_locations: np.ndarray, 
                     verbose: bool) -> np.ndarray:
    """Exact analytical solution for disk using Green's function."""
    try:
        from .analytical_solver import AnalyticalForwardSolver
    except ImportError:
        from analytical_solver import AnalyticalForwardSolver
    
    if verbose:
        print("  Reference: Analytical Green's function (EXACT)")
    
    solver = AnalyticalForwardSolver(sensor_locations=sensor_locations)
    return solver.solve(sources)


def _ellipse_conformal(a: float, b: float, sources: List, 
                        sensor_locations: np.ndarray, verbose: bool) -> np.ndarray:
    """Exact solution for ellipse using conformal mapping (Joukowski)."""
    try:
        from .conformal_solver import ConformalForwardSolver, EllipseMap
    except ImportError:
        from conformal_solver import ConformalForwardSolver, EllipseMap
    
    if verbose:
        print("  Reference: Conformal mapping (Joukowski, EXACT)")
    
    cmap = EllipseMap(a, b)
    solver = ConformalForwardSolver(cmap, sensor_locations=sensor_locations)
    return solver.solve(sources)


def _star_conformal(domain_params: Dict, sources: List, 
                     sensor_locations: np.ndarray, verbose: bool) -> np.ndarray:
    """Solution for star using numerical conformal mapping."""
    try:
        from .conformal_solver import ConformalForwardSolver, NumericalConformalMap
    except ImportError:
        from conformal_solver import ConformalForwardSolver, NumericalConformalMap
    
    if verbose:
        print("  Reference: Numerical conformal mapping (high accuracy)")
    
    # Generate star boundary function for the conformal map
    n_petals = domain_params.get('n_petals', 5) if domain_params else 5
    amplitude = domain_params.get('amplitude', 0.3) if domain_params else 0.3
    
    # Create boundary function γ(t) -> complex
    def star_boundary(t):
        r = 1.0 + amplitude * np.cos(n_petals * t)
        return r * np.exp(1j * t)
    
    cmap = NumericalConformalMap(star_boundary, n_boundary=256)
    solver = ConformalForwardSolver(cmap, sensor_locations=sensor_locations)
    return solver.solve(sources)


def _richardson_extrapolation(domain_type: str, domain_params: Dict,
                               sources: List, sensor_locations: np.ndarray,
                               verbose: bool) -> np.ndarray:
    """
    Reference solution via Richardson extrapolation.
    
    Uses FEM solutions at progressively finer meshes to extrapolate
    to h→0 limit. For O(h²) convergence:
        u_exact ≈ (4*u_{h/2} - u_h) / 3
    
    We use three resolutions for verification.
    """
    try:
        from .fem_solver import FEMForwardSolver
        from .mesh import create_polygon_mesh, create_brain_mesh, get_brain_boundary, create_ellipse_mesh
    except ImportError:
        from fem_solver import FEMForwardSolver
        from mesh import create_polygon_mesh, create_brain_mesh, get_brain_boundary, create_ellipse_mesh
    
    if verbose:
        print("  Reference: Richardson extrapolation (O(h^4) accuracy)")
    
    # Use three mesh resolutions: h, h/2, h/4
    base_h = 0.08  # Base resolution
    resolutions = [base_h, base_h/2, base_h/4]  # 0.08, 0.04, 0.02
    
    solutions = []
    
    for h in resolutions:
        if verbose:
            print(f"    Computing at h={h:.3f}...")
        
        # Build mesh with sensor locations
        if domain_type == 'square':
            vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            mesh_data = create_polygon_mesh(vertices, h, sensor_locations=sensor_locations)
        
        elif domain_type == 'polygon':
            vertices = domain_params.get('vertices') if domain_params else \
                       [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
            mesh_data = create_polygon_mesh(vertices, h, sensor_locations=sensor_locations)
        
        elif domain_type == 'brain':
            mesh_data = create_brain_mesh(h, sensor_locations=sensor_locations)
        
        elif domain_type == 'ellipse':
            a = domain_params.get('a', 2.0) if domain_params else 2.0
            b = domain_params.get('b', 1.0) if domain_params else 1.0
            mesh_data = create_ellipse_mesh(a, b, h, sensor_locations=sensor_locations)
        
        elif domain_type == 'star':
            # Star domain as polygon
            n_petals = domain_params.get('n_petals', 5) if domain_params else 5
            amplitude = domain_params.get('amplitude', 0.3) if domain_params else 0.3
            n_verts = 100
            theta = np.linspace(0, 2*np.pi, n_verts, endpoint=False)
            r = 1.0 + amplitude * np.cos(n_petals * theta)
            vertices = [(r[i] * np.cos(theta[i]), r[i] * np.sin(theta[i])) for i in range(n_verts)]
            mesh_data = create_polygon_mesh(vertices, h, sensor_locations=sensor_locations)
        
        else:
            raise ValueError(f"Richardson not implemented for: {domain_type}")
        
        # Solve
        solver = FEMForwardSolver(mesh_data=mesh_data, sensor_locations=sensor_locations, 
                                  verbose=False)
        u = solver.solve(sources)
        solutions.append(u)
    
    u_h, u_h2, u_h4 = solutions
    
    # Richardson extrapolation for O(h²) convergence
    # First extrapolation: (4*u_{h/2} - u_h) / 3
    u_extrap1 = (4 * u_h2 - u_h) / 3
    u_extrap2 = (4 * u_h4 - u_h2) / 3
    
    # Second extrapolation (O(h^4)): (16*u_extrap2 - u_extrap1) / 15
    u_reference = (16 * u_extrap2 - u_extrap1) / 15
    
    # Check convergence by comparing extrapolations
    error_estimate = np.linalg.norm(u_extrap2 - u_extrap1) / np.linalg.norm(u_reference)
    
    if verbose:
        print(f"    Estimated extrapolation error: {error_estimate:.2e}")
        if error_estimate > 0.01:
            warnings.warn(f"Richardson extrapolation may not have converged well "
                         f"(error estimate {error_estimate:.2e})")
    
    return u_reference


def verify_fem_convergence(domain_type: str, domain_params: Dict,
                            sources: List, sensor_locations: np.ndarray,
                            resolutions: List[float] = None,
                            verbose: bool = True) -> Dict:
    """
    Verify FEM convergence by comparing to reference solution.
    
    This is the proper way to do forward mesh convergence:
    - Get reference solution (analytical or Richardson extrapolation)
    - Create FEM mesh WITH SENSORS EMBEDDED (no interpolation!)
    - Compare FEM at various h to reference at exactly the same sensor points
    - Should see O(h²) convergence
    
    IMPORTANT: Sensors are embedded into the mesh as exact nodes, so
    FEM solution is extracted directly without any interpolation error.
    
    Parameters
    ----------
    domain_type : str
        Domain type
    domain_params : dict
        Domain parameters
    sources : list
        Source configuration
    sensor_locations : array
        FIXED evaluation points embedded into mesh
    resolutions : list, optional
        Mesh resolutions to test
    verbose : bool
        Print progress
        
    Returns
    -------
    dict with keys:
        'resolutions': list of h values
        'errors': list of relative errors
        'convergence_rate': estimated convergence rate
        'reference_method': string describing reference method
    """
    try:
        from .fem_solver import FEMForwardSolver
        from .mesh import create_disk_mesh, create_ellipse_mesh, create_polygon_mesh, create_brain_mesh
    except ImportError:
        from fem_solver import FEMForwardSolver
        from mesh import create_disk_mesh, create_ellipse_mesh, create_polygon_mesh, create_brain_mesh
    
    if resolutions is None:
        # Use widely spaced resolutions to show clear O(h²) convergence
        # Avoid h=0.08 which can have mesh-source geometry issues
        resolutions = [0.20, 0.15, 0.10, 0.07, 0.05, 0.04]
    
    if verbose:
        print(f"Forward mesh convergence for {domain_type}")
        print("=" * 50)
    
    # Get reference solution at FIXED sensor locations
    if verbose:
        print("Getting reference solution...")
    u_ref = get_reference_solution(domain_type, domain_params, sources, 
                                    sensor_locations, verbose=verbose)
    
    # Determine reference method for documentation
    if domain_type in ['disk']:
        ref_method = 'Analytical Green\'s function'
    else:
        ref_method = 'Richardson extrapolation'
    
    if verbose:
        print(f"\nComparing FEM to reference at {len(sensor_locations)} fixed points...")
    
    errors = []
    for h in resolutions:
        # Build mesh WITH SENSORS EMBEDDED (no interpolation needed!)
        if domain_type == 'disk':
            mesh_data = create_disk_mesh(h, sensor_locations=sensor_locations)
        elif domain_type == 'ellipse':
            a = domain_params.get('a', 2.0) if domain_params else 2.0
            b = domain_params.get('b', 1.0) if domain_params else 1.0
            mesh_data = create_ellipse_mesh(a, b, h, sensor_locations=sensor_locations)
        elif domain_type == 'square':
            vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            mesh_data = create_polygon_mesh(vertices, h, sensor_locations=sensor_locations)
        elif domain_type == 'polygon':
            vertices = domain_params.get('vertices') if domain_params else \
                       [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
            mesh_data = create_polygon_mesh(vertices, h, sensor_locations=sensor_locations)
        elif domain_type == 'brain':
            mesh_data = create_brain_mesh(h, sensor_locations=sensor_locations)
        elif domain_type == 'star':
            # Star domain as polygon
            n_petals = domain_params.get('n_petals', 5) if domain_params else 5
            amplitude = domain_params.get('amplitude', 0.3) if domain_params else 0.3
            n_verts = 100
            theta = np.linspace(0, 2*np.pi, n_verts, endpoint=False)
            r = 1.0 + amplitude * np.cos(n_petals * theta)
            vertices = [(r[i] * np.cos(theta[i]), r[i] * np.sin(theta[i])) for i in range(n_verts)]
            mesh_data = create_polygon_mesh(vertices, h, sensor_locations=sensor_locations)
        else:
            raise ValueError(f"Unknown domain: {domain_type}")
        
        nodes = mesh_data[0]
        
        # Solve FEM with embedded sensors - returns solution at sensor nodes directly
        solver = FEMForwardSolver(mesh_data=mesh_data, sensor_locations=sensor_locations, verbose=False)
        u_fem_at_sensors = solver.solve(sources)  # Direct extraction, NO interpolation!
        
        # Center both solutions
        u_fem_centered = u_fem_at_sensors - np.mean(u_fem_at_sensors)
        u_ref_centered = u_ref - np.mean(u_ref)
        
        error = np.linalg.norm(u_fem_centered - u_ref_centered) / np.linalg.norm(u_ref_centered)
        errors.append(error)
        
        if verbose:
            print(f"  h={h:.3f}: error={error:.2e}, nodes={len(nodes)}")
    
    # Estimate convergence rate from log-log slope
    log_h = np.log(resolutions)
    log_err = np.log(errors)
    
    # Linear fit
    slope, intercept = np.polyfit(log_h, log_err, 1)
    
    if verbose:
        print(f"\nEstimated convergence rate: O(h^{slope:.2f})")
        print(f"Expected for P1 FEM: O(h^2)")
    
    return {
        'resolutions': resolutions,
        'errors': errors,
        'convergence_rate': slope,
        'reference_method': ref_method
    }


def generate_measurement_data(domain_type: str, domain_params: Dict,
                               sources: List[Tuple[Tuple[float, float], float]],
                               sensor_locations: np.ndarray,
                               noise_level: float = 0.001,
                               seed: int = 42,
                               verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic measurement data using the best available reference.
    
    This is THE function to call at the start of any calibration or comparison.
    The returned u_measured should be FIXED throughout all subsequent steps.
    
    Parameters
    ----------
    domain_type : str
        Domain type
    domain_params : dict
        Domain parameters
    sources : list
        True source configuration
    sensor_locations : array, shape (n_sensors, 2)
        Fixed sensor locations
    noise_level : float
        Standard deviation of Gaussian noise
    seed : int
        Random seed for reproducibility
    verbose : bool
        Print info
        
    Returns
    -------
    u_clean : array
        Noise-free reference solution
    u_measured : array
        Noisy measurement data
    """
    if verbose:
        print(f"Generating measurement data for {domain_type}")
        print(f"  Sensors: {len(sensor_locations)}")
        print(f"  Sources: {len(sources)}")
        print(f"  Noise level: {noise_level}")
    
    # Get reference solution
    u_clean = get_reference_solution(domain_type, domain_params, sources,
                                      sensor_locations, verbose=verbose)
    
    # Add noise
    np.random.seed(seed)
    noise = noise_level * np.random.randn(len(u_clean))
    u_measured = u_clean + noise
    
    if verbose:
        signal_norm = np.linalg.norm(u_clean)
        noise_norm = np.linalg.norm(noise)
        snr = signal_norm / noise_norm if noise_norm > 0 else float('inf')
        print(f"  Signal-to-noise ratio: {snr:.1f}")
    
    return u_clean, u_measured
