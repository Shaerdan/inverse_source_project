"""
Calibration Module for Inverse Source Localization
===================================================

This module finds optimal parameters for all supported domains:
1. Optimal regularization parameter (alpha) via L-curve for each method
2. Optimal forward mesh resolution via convergence study
3. Optimal source grid resolution via convergence study

Results are saved to a JSON configuration file that can be loaded
for subsequent experiments.

Usage:
    python -m inverse_source.cli calibrate --output-dir results/calibration
    
    # Then run experiments using calibrated parameters:
    python -m inverse_source.cli compare --domain disk --use-calibration

Author: Claude (Anthropic)
Date: January 2026
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from time import time
from pathlib import Path

# Import sensor utilities
try:
    from .comparison import get_sensor_locations
    from .reference_solution import (get_reference_solution, generate_measurement_data,
                                      verify_fem_convergence)
except ImportError:
    from comparison import get_sensor_locations
    from reference_solution import (get_reference_solution, generate_measurement_data,
                                     verify_fem_convergence)


def _build_forward_solver(domain_type: str, domain_params: dict, 
                           resolution: float, sensor_locations: np.ndarray):
    """Build appropriate forward solver for domain with given sensors."""
    if domain_type == 'disk':
        try:
            from .analytical_solver import AnalyticalForwardSolver
        except ImportError:
            from analytical_solver import AnalyticalForwardSolver
        return AnalyticalForwardSolver(sensor_locations=sensor_locations)
    
    elif domain_type == 'ellipse':
        try:
            from .conformal_solver import ConformalForwardSolver
            from .comparison import get_conformal_map
        except ImportError:
            from conformal_solver import ConformalForwardSolver
            from comparison import get_conformal_map
        conformal_map = get_conformal_map(domain_type, domain_params)
        return ConformalForwardSolver(conformal_map, sensor_locations=sensor_locations)
    
    elif domain_type == 'star':
        try:
            from .conformal_solver import ConformalForwardSolver
            from .comparison import get_conformal_map
        except ImportError:
            from conformal_solver import ConformalForwardSolver
            from comparison import get_conformal_map
        conformal_map = get_conformal_map(domain_type, domain_params)
        return ConformalForwardSolver(conformal_map, sensor_locations=sensor_locations)
    
    elif domain_type in ['square', 'polygon']:
        try:
            from .fem_solver import FEMForwardSolver
            from .mesh import create_polygon_mesh
        except ImportError:
            from fem_solver import FEMForwardSolver
            from mesh import create_polygon_mesh
        
        if domain_type == 'square':
            vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        else:
            vertices = domain_params.get('vertices') if domain_params else [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        
        mesh_data = create_polygon_mesh(vertices, resolution, sensor_locations=sensor_locations)
        return FEMForwardSolver(mesh_data=mesh_data, sensor_locations=sensor_locations, verbose=False)
    
    elif domain_type == 'brain':
        try:
            from .fem_solver import FEMForwardSolver
            from .mesh import create_brain_mesh
        except ImportError:
            from fem_solver import FEMForwardSolver
            from mesh import create_brain_mesh
        
        mesh_data = create_brain_mesh(resolution, sensor_locations=sensor_locations)
        return FEMForwardSolver(mesh_data=mesh_data, sensor_locations=sensor_locations, verbose=False)
    
    return None


@dataclass
class DomainCalibration:
    """Calibration results for a single domain."""
    domain_type: str
    domain_params: dict
    
    # Optimal mesh resolutions
    forward_mesh_resolution: float
    source_grid_resolution: float
    
    # Optimal alpha for each regularization method
    alpha_l1: float
    alpha_l2: float
    alpha_tv: float
    
    # Convergence study results
    forward_convergence: dict  # {resolution: error}
    source_convergence: dict   # {resolution: localization_score}
    
    # L-curve analysis results
    lcurve_l1: dict  # {alpha: (residual, regularizer)}
    lcurve_l2: dict
    lcurve_tv: dict
    
    # Timing info
    calibration_time_seconds: float


@dataclass 
class CalibrationConfig:
    """Complete calibration configuration."""
    version: str
    calibration_date: str
    domains: Dict[str, DomainCalibration]
    
    # Global defaults (fallback)
    default_forward_resolution: float = 0.08
    default_source_resolution: float = 0.12
    default_alpha_l1: float = 1e-4
    default_alpha_l2: float = 1e-4
    default_alpha_tv: float = 1e-2


# Default domain configurations
DEFAULT_DOMAINS = {
    'disk': {},
    'ellipse': {'a': 2.0, 'b': 1.0},
    'star': {'n_petals': 5, 'amplitude': 0.3},
    'square': {},
    'polygon': {'vertices': [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]},
    'brain': {}
}


def calibrate_domain(domain_type: str, 
                     domain_params: dict = None,
                     noise_level: float = 0.001,
                     seed: int = 42,
                     verbose: bool = True) -> DomainCalibration:
    """
    Run full calibration for a single domain.
    
    Parameters
    ----------
    domain_type : str
        Domain type: 'disk', 'ellipse', 'star', 'square', 'polygon', 'brain'
    domain_params : dict
        Domain-specific parameters
    noise_level : float
        Noise level for synthetic data
    seed : int
        Random seed
    verbose : bool
        Print progress
        
    Returns
    -------
    calibration : DomainCalibration
        Complete calibration results
    """
    t_start = time()
    
    if domain_params is None:
        domain_params = DEFAULT_DOMAINS.get(domain_type, {})
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"CALIBRATING: {domain_type.upper()}")
        print(f"{'='*70}")
    
    # =========================================================================
    # PHASE 0: Setup - Define sensors and sources
    # =========================================================================
    n_sensors = 100
    sensor_locations = get_sensor_locations(domain_type, domain_params, n_sensors)
    
    try:
        from .comparison import create_domain_sources
    except ImportError:
        from comparison import create_domain_sources
    
    sources_true = create_domain_sources(domain_type, domain_params)
    
    if verbose:
        print(f"  Sensors: {n_sensors} fixed locations")
        print(f"  Sources: {len(sources_true)}")
    
    # =========================================================================
    # PHASE 1: Generate u_measured ONCE (this is FIXED for all calibration)
    # =========================================================================
    if verbose:
        print("\n[0/3] Generating reference measurement data...")
    
    u_clean, u_measured = generate_measurement_data(
        domain_type, domain_params, sources_true, sensor_locations,
        noise_level=noise_level, seed=seed, verbose=verbose
    )
    
    # =========================================================================
    # PHASE 2: Forward mesh convergence (verify FEM accuracy)
    # =========================================================================
    if verbose:
        print("\n[1/3] Forward mesh convergence study...")
    
    forward_conv_result = verify_fem_convergence(
        domain_type, domain_params, sources_true, sensor_locations,
        resolutions=[0.20, 0.15, 0.10, 0.08, 0.06, 0.05],
        verbose=verbose
    )
    
    forward_conv = dict(zip(forward_conv_result['resolutions'], 
                            forward_conv_result['errors']))
    
    # Find optimal: first resolution with error < 1%
    optimal_forward = 0.05  # Default to finest
    for h, err in zip(forward_conv_result['resolutions'], forward_conv_result['errors']):
        if err < 0.01:
            optimal_forward = h
            break
    
    if verbose:
        print(f"  -> Optimal forward resolution: {optimal_forward:.3f}")
    
    # =========================================================================
    # PHASE 3: Source grid convergence (uses FIXED u_measured)
    # =========================================================================
    if verbose:
        print("\n[2/3] Source grid convergence study...")
    
    source_conv, optimal_source = _run_source_convergence_fixed(
        domain_type, domain_params, sources_true,
        sensor_locations, u_measured,  # Pass fixed data
        forward_resolution=optimal_forward,
        verbose=verbose
    )
    
    # =========================================================================
    # PHASE 4: L-curve analysis for optimal alpha (uses FIXED u_measured)
    # =========================================================================
    if verbose:
        print("\n[3/3] L-curve analysis for optimal alpha...")
    
    lcurve_results, optimal_alphas = _run_lcurve_analysis_fixed(
        domain_type, domain_params, sources_true,
        sensor_locations, u_measured,  # Pass fixed data
        forward_resolution=optimal_forward,
        source_resolution=optimal_source,
        verbose=verbose
    )
    
    calibration_time = time() - t_start
    
    if verbose:
        print(f"\n{'-'*70}")
        print(f"CALIBRATION COMPLETE: {domain_type}")
        print(f"{'-'*70}")
        print(f"  Forward mesh resolution: {optimal_forward:.3f}")
        print(f"  Source grid resolution:  {optimal_source:.3f}")
        print(f"  Alpha L1: {optimal_alphas['l1']:.2e}")
        print(f"  Alpha L2: {optimal_alphas['l2']:.2e}")
        print(f"  Alpha TV: {optimal_alphas['tv']:.2e}")
        print(f"  Convergence rate: O(h^{forward_conv_result['convergence_rate']:.2f})")
        print(f"  Time: {calibration_time:.1f}s")
    
    return DomainCalibration(
        domain_type=domain_type,
        domain_params=domain_params,
        forward_mesh_resolution=optimal_forward,
        source_grid_resolution=optimal_source,
        alpha_l1=optimal_alphas['l1'],
        alpha_l2=optimal_alphas['l2'],
        alpha_tv=optimal_alphas['tv'],
        forward_convergence=forward_conv,
        source_convergence=source_conv,
        lcurve_l1=lcurve_results['l1'],
        lcurve_l2=lcurve_results['l2'],
        lcurve_tv=lcurve_results['tv'],
        calibration_time_seconds=calibration_time
    )


def _run_forward_convergence(domain_type: str, domain_params: dict,
                              sources_true: list, verbose: bool = True) -> Tuple[dict, float]:
    """Run forward mesh convergence study."""
    try:
        from .mesh_convergence import _solve_forward_at_resolution
    except ImportError:
        from mesh_convergence import _solve_forward_at_resolution
    
    resolutions = [0.20, 0.15, 0.10, 0.08, 0.06, 0.05]
    reference_resolution = 0.03
    
    # Compute reference solution
    ref_result = _solve_forward_at_resolution(
        domain_type, domain_params, sources_true, reference_resolution
    )
    u_ref = ref_result['boundary_values']
    
    from scipy.interpolate import interp1d
    
    convergence = {}
    for h in resolutions:
        res = _solve_forward_at_resolution(domain_type, domain_params, sources_true, h)
        
        # Interpolate for comparison
        theta_ref = np.linspace(0, 2*np.pi, len(u_ref), endpoint=False)
        theta_test = np.linspace(0, 2*np.pi, len(res['boundary_values']), endpoint=False)
        
        interp = interp1d(theta_ref, u_ref, kind='linear', fill_value='extrapolate')
        u_ref_interp = interp(theta_test)
        
        error = np.linalg.norm(res['boundary_values'] - u_ref_interp) / np.linalg.norm(u_ref_interp)
        convergence[h] = float(error)
        
        if verbose:
            print(f"    h={h:.2f}: error={error:.2e}")
    
    # Find optimal: first resolution with error < 1%
    optimal = resolutions[-1]  # Default to finest
    for h in resolutions:
        if convergence[h] < 0.01:
            optimal = h
            break
    
    return convergence, optimal


def _run_source_convergence(domain_type: str, domain_params: dict,
                             sources_true: list, forward_resolution: float,
                             noise_level: float, seed: int,
                             verbose: bool = True) -> Tuple[dict, float]:
    """Run proper source grid convergence study with L-curve alpha per resolution.
    
    For each grid resolution, we find the optimal alpha via L-curve, then
    measure localization. This gives a fair comparison across resolutions.
    
    CRITICAL: Uses FIXED sensor locations independent of mesh resolution.
    """
    try:
        from .fem_solver import FEMLinearInverseSolver, FEMForwardSolver
        from .parameter_selection import find_lcurve_corner
        from .mesh import (get_brain_boundary, create_polygon_mesh, create_ellipse_mesh, 
                          create_disk_mesh, get_disk_sensor_locations, get_ellipse_sensor_locations,
                          get_polygon_sensor_locations)
    except ImportError:
        from fem_solver import FEMLinearInverseSolver, FEMForwardSolver
        from parameter_selection import find_lcurve_corner
        from mesh import (get_brain_boundary, create_polygon_mesh, create_ellipse_mesh, 
                         create_disk_mesh, get_disk_sensor_locations, get_ellipse_sensor_locations,
                         get_polygon_sensor_locations)
    
    import cvxpy as cp
    
    # Wide range: coarse to fine
    resolutions = [0.30, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06]
    
    # Get true source positions for localization calculation
    true_positions = np.array([s[0] for s in sources_true])
    
    # CRITICAL: Define FIXED sensor locations (independent of mesh)
    n_sensors = 100
    sensor_locations = get_sensor_locations(domain_type, domain_params, n_sensors)
    
    if verbose:
        print(f"    Using {n_sensors} fixed sensor locations")
    
    # Generate synthetic data ONCE with fixed sensors
    np.random.seed(seed)
    forward = _build_forward_solver(domain_type, domain_params, forward_resolution, sensor_locations)
    if forward is None:
        return {}, resolutions[0]
    
    u_clean = forward.solve(sources_true)
    u_measured = u_clean + noise_level * np.random.randn(len(u_clean))
    u_centered = u_measured - np.mean(u_measured)
    
    convergence = {}
    best_loc = 0
    optimal = resolutions[0]
    
    for h in resolutions:
        try:
            t_start = time()
            
            # Build inverse solver with SAME sensor locations
            solver = _build_solver_for_domain(domain_type, domain_params, 
                                               forward_resolution, h, sensor_locations)
            if solver is None:
                if verbose:
                    print(f"    h={h:.2f}: SKIPPED (solver build failed)")
                continue
            
            # Build Green's matrix (uses same sensors as data)
            solver.build_greens_matrix(verbose=False)
            G = solver.G
            interior_points = solver.interior_points
            n = G.shape[1]
            
            # Verify dimensions match (sanity check)
            if G.shape[0] != len(u_centered):
                if verbose:
                    print(f"    h={h:.2f}: DIMENSION MISMATCH G={G.shape}, u={len(u_centered)}")
                continue
            
            # L-curve to find optimal alpha for THIS resolution
            alphas = np.logspace(-6, -1, 12)
            residuals = []
            regularizers = []
            solutions = []
            
            for alpha in alphas:
                q_var = cp.Variable(n)
                constraints = [cp.sum(q_var) == 0]
                objective = cp.Minimize(
                    0.5 * cp.sum_squares(G @ q_var - u_centered) + 
                    0.5 * alpha * cp.sum_squares(q_var)
                )
                prob = cp.Problem(objective, constraints)
                try:
                    prob.solve(solver=cp.ECOS, verbose=False)
                    q = q_var.value if q_var.value is not None else np.zeros(n)
                except:
                    q = np.zeros(n)
                
                res = np.linalg.norm(G @ q - u_centered)
                reg = np.linalg.norm(q)
                residuals.append(res)
                regularizers.append(reg)
                solutions.append(q)
            
            # Find L-curve corner
            idx = find_lcurve_corner(np.array(residuals), np.array(regularizers))
            optimal_alpha = alphas[idx]
            q_optimal = solutions[idx]
            
            # Calculate localization score with optimal solution
            loc_score = _compute_localization(q_optimal, interior_points, true_positions)
            
            t_elapsed = time() - t_start
            
            convergence[h] = {
                'localization': float(loc_score),
                'optimal_alpha': float(optimal_alpha),
                'n_sources': n,
                'time': float(t_elapsed)
            }
            
            if verbose:
                print(f"    h={h:.2f}: loc={loc_score:.4f}, α*={optimal_alpha:.2e}, "
                      f"n={n}, time={t_elapsed:.2f}s")
            
            if loc_score > best_loc:
                best_loc = loc_score
                optimal = h
                
        except Exception as e:
            if verbose:
                print(f"    h={h:.2f}: FAILED - {e}")
            convergence[h] = {
                'localization': 0.0,
                'optimal_alpha': 1e-4,
                'n_sources': 0,
                'time': 0.0
            }
    
    if verbose:
        print(f"    -> Best resolution: h={optimal:.2f} (localization={best_loc:.4f})")
    
    return convergence, optimal


def _run_source_convergence_fixed(domain_type: str, domain_params: dict,
                                   sources_true: list, sensor_locations: np.ndarray,
                                   u_measured: np.ndarray, forward_resolution: float,
                                   verbose: bool = True) -> Tuple[dict, float]:
    """Run source grid convergence with PRE-COMPUTED u_measured.
    
    This is the correct version: u_measured is generated ONCE at the start
    and passed to all calibration steps. No regeneration!
    
    Parameters
    ----------
    domain_type : str
        Domain type
    domain_params : dict
        Domain parameters
    sources_true : list
        True source configuration
    sensor_locations : array
        FIXED sensor locations
    u_measured : array
        PRE-COMPUTED measurement data (FIXED, do not regenerate!)
    forward_resolution : float
        Forward mesh resolution
    verbose : bool
        Print progress
    """
    try:
        from .fem_solver import FEMLinearInverseSolver
    except ImportError:
        from fem_solver import FEMLinearInverseSolver
    
    import cvxpy as cp
    
    # Wide range: coarse to fine
    resolutions = [0.30, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06]
    
    # Get true source positions for localization calculation
    true_positions = np.array([s[0] for s in sources_true])
    
    u_centered = u_measured - np.mean(u_measured)
    
    convergence = {}
    best_loc = 0
    optimal = resolutions[0]
    
    for h in resolutions:
        try:
            t_start = time()
            
            # Build inverse solver with SAME sensor locations as data
            solver = _build_solver_for_domain(domain_type, domain_params, 
                                               forward_resolution, h, sensor_locations)
            if solver is None:
                if verbose:
                    print(f"    h={h:.2f}: SKIPPED (solver build failed)")
                continue
            
            # Build Green's matrix
            solver.build_greens_matrix(verbose=False)
            G = solver.G
            interior_points = solver.interior_points
            n = G.shape[1]
            
            # Verify dimensions match
            if G.shape[0] != len(u_centered):
                if verbose:
                    print(f"    h={h:.2f}: DIMENSION MISMATCH G={G.shape}, u={len(u_centered)}")
                continue
            
            # L-curve to find optimal alpha for THIS resolution
            alphas = np.logspace(-6, -1, 12)
            residuals = []
            regularizers = []
            
            for alpha in alphas:
                try:
                    q = cp.Variable(n)
                    objective = cp.Minimize(cp.sum_squares(G @ q - u_centered) + 
                                           alpha * cp.norm1(q))
                    constraints = [cp.sum(q) == 0]
                    problem = cp.Problem(objective, constraints)
                    problem.solve(solver=cp.SCS, verbose=False)
                    
                    if q.value is not None:
                        residuals.append(np.linalg.norm(G @ q.value - u_centered))
                        regularizers.append(np.sum(np.abs(q.value)))
                    else:
                        residuals.append(np.inf)
                        regularizers.append(np.inf)
                except:
                    residuals.append(np.inf)
                    regularizers.append(np.inf)
            
            # Find L-curve corner
            valid = [i for i in range(len(alphas)) if residuals[i] < np.inf]
            if len(valid) < 3:
                optimal_alpha = 1e-4
            else:
                # Simple curvature-based corner detection
                log_res = np.log10([residuals[i] for i in valid])
                log_reg = np.log10([regularizers[i] + 1e-12 for i in valid])
                
                # Finite differences for curvature
                if len(valid) >= 3:
                    curvature = []
                    for i in range(1, len(valid)-1):
                        d1 = (log_res[i] - log_res[i-1], log_reg[i] - log_reg[i-1])
                        d2 = (log_res[i+1] - log_res[i], log_reg[i+1] - log_reg[i])
                        cross = d1[0]*d2[1] - d1[1]*d2[0]
                        curvature.append(cross)
                    corner_idx = valid[1 + np.argmax(curvature)]
                else:
                    corner_idx = valid[len(valid)//2]
                optimal_alpha = alphas[corner_idx]
            
            # Solve with optimal alpha
            q = cp.Variable(n)
            objective = cp.Minimize(cp.sum_squares(G @ q - u_centered) + 
                                   optimal_alpha * cp.norm1(q))
            constraints = [cp.sum(q) == 0]
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.SCS, verbose=False)
            
            if q.value is None:
                if verbose:
                    print(f"    h={h:.2f}: SOLVE FAILED")
                continue
            
            q_sol = q.value
            
            # Compute localization
            loc = _compute_localization(q_sol, interior_points, true_positions)
            
            elapsed = time() - t_start
            
            convergence[h] = {
                'localization': loc,
                'optimal_alpha': optimal_alpha,
                'n_sources': n,
                'time': elapsed
            }
            
            if verbose:
                print(f"    h={h:.2f}: loc={loc:.4f}, α={optimal_alpha:.1e}, "
                      f"sources={n}, time={elapsed:.1f}s")
            
            if loc > best_loc:
                best_loc = loc
                optimal = h
                
        except Exception as e:
            if verbose:
                print(f"    h={h:.2f}: FAILED ({e})")
            convergence[h] = {
                'localization': 0.0,
                'optimal_alpha': 1e-4,
                'n_sources': 0,
                'time': 0.0
            }
    
    if verbose:
        print(f"    -> Best resolution: h={optimal:.2f} (localization={best_loc:.4f})")
    
    return convergence, optimal


def _build_solver_for_domain(domain_type: str, domain_params: dict,
                              forward_resolution: float, source_resolution: float,
                              sensor_locations: np.ndarray = None):
    """Build FEM solver for the given domain with fixed sensor locations."""
    try:
        from .fem_solver import FEMLinearInverseSolver
        from .mesh import get_brain_boundary, create_disk_mesh, create_polygon_mesh, create_ellipse_mesh
    except ImportError:
        from fem_solver import FEMLinearInverseSolver
        from mesh import get_brain_boundary, create_disk_mesh, create_polygon_mesh, create_ellipse_mesh
    
    try:
        if domain_type == 'disk':
            return FEMLinearInverseSolver(
                forward_resolution=forward_resolution,
                source_resolution=source_resolution,
                sensor_locations=sensor_locations,
                verbose=False
            )
        elif domain_type == 'ellipse':
            a = domain_params.get('a', 2.0) if domain_params else 2.0
            b = domain_params.get('b', 1.0) if domain_params else 1.0
            # Create mesh with sensors
            mesh_data = create_ellipse_mesh(a, b, forward_resolution, sensor_locations=sensor_locations)
            return FEMLinearInverseSolver(
                forward_resolution=forward_resolution,
                source_resolution=source_resolution,
                sensor_locations=sensor_locations,
                mesh_data=mesh_data,
                verbose=False
            )
        else:
            # Square, polygon, star, brain - all use polygon representation
            if domain_type == 'square':
                vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            elif domain_type == 'polygon':
                vertices = domain_params.get('vertices', [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]) if domain_params else [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
            elif domain_type == 'star':
                n_petals = domain_params.get('n_petals', 5) if domain_params else 5
                amplitude = domain_params.get('amplitude', 0.3) if domain_params else 0.3
                n_v = 100
                theta_v = np.linspace(0, 2*np.pi, n_v, endpoint=False)
                r_v = 1.0 + amplitude * np.cos(n_petals * theta_v)
                vertices = [(r_v[i] * np.cos(theta_v[i]), r_v[i] * np.sin(theta_v[i])) for i in range(n_v)]
            else:  # brain
                boundary = get_brain_boundary(n_points=100)
                vertices = [tuple(p) for p in boundary]
            
            # Create mesh with sensor locations
            mesh_data = create_polygon_mesh(vertices, forward_resolution, sensor_locations=sensor_locations)
            return FEMLinearInverseSolver(
                forward_resolution=forward_resolution,
                source_resolution=source_resolution,
                sensor_locations=sensor_locations,
                mesh_data=mesh_data,
                verbose=False
            )
    except Exception as e:
        print(f"    Warning: Could not build solver for {domain_type}: {e}")
        return None


def _compute_localization(q: np.ndarray, interior_points: np.ndarray, 
                          true_positions: np.ndarray, sigma: float = 0.15) -> float:
    """Compute localization score - how much intensity is near true sources."""
    if q is None or len(q) == 0:
        return 0.0
    
    q_abs = np.abs(q)
    total_intensity = np.sum(q_abs)
    if total_intensity < 1e-12:
        return 0.0
    
    # For each interior point, find distance to nearest true source
    weights = np.zeros(len(interior_points))
    for i, pt in enumerate(interior_points):
        dists = np.linalg.norm(true_positions - pt, axis=1)
        min_dist = np.min(dists)
        weights[i] = np.exp(-min_dist**2 / (2 * sigma**2))
    
    # Weighted sum
    loc_score = np.sum(q_abs * weights) / total_intensity
    return float(loc_score)


def _run_lcurve_analysis(domain_type: str, domain_params: dict,
                          sources_true: list, forward_resolution: float,
                          source_resolution: float, noise_level: float,
                          seed: int, verbose: bool = True) -> Tuple[dict, dict]:
    """Run L-curve analysis to find optimal alpha for each method.
    
    CRITICAL: Uses the inverse solver's mesh for generating synthetic data
    to ensure boundary points correspond exactly.
    """
    try:
        from .fem_solver import FEMLinearInverseSolver, FEMForwardSolver
        from .parameter_selection import find_lcurve_corner, build_gradient_operator
        from .mesh import get_brain_boundary
    except ImportError:
        from fem_solver import FEMLinearInverseSolver, FEMForwardSolver
        from parameter_selection import find_lcurve_corner, build_gradient_operator
        from mesh import get_brain_boundary
    
    import cvxpy as cp
    
    # Build inverse solver FIRST
    if domain_type == 'disk':
        solver = FEMLinearInverseSolver(
            forward_resolution=forward_resolution,
            source_resolution=source_resolution,
            verbose=False
        )
    elif domain_type == 'ellipse':
        a = domain_params.get('a', 2.0)
        b = domain_params.get('b', 1.0)
        solver = FEMLinearInverseSolver.from_ellipse(
            a, b,
            forward_resolution=forward_resolution,
            source_resolution=source_resolution,
            verbose=False
        )
    else:
        # Square, polygon, star, brain - all use polygon representation
        if domain_type == 'square':
            vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        elif domain_type == 'polygon':
            vertices = domain_params.get('vertices', [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
        elif domain_type == 'star':
            n_petals = domain_params.get('n_petals', 5)
            amplitude = domain_params.get('amplitude', 0.3)
            n_v = 100
            theta_v = np.linspace(0, 2*np.pi, n_v, endpoint=False)
            r_v = 1.0 + amplitude * np.cos(n_petals * theta_v)
            vertices = [(r_v[i] * np.cos(theta_v[i]), r_v[i] * np.sin(theta_v[i])) for i in range(n_v)]
        else:  # brain
            boundary = get_brain_boundary(n_points=100)
            vertices = [tuple(p) for p in boundary]
        
        solver = FEMLinearInverseSolver.from_polygon(
            vertices,
            forward_resolution=forward_resolution,
            source_resolution=source_resolution,
            verbose=False
        )
    
    # CRITICAL FIX: Generate synthetic data using the SAME mesh as the inverse solver
    mesh_data = (solver.nodes, solver.elements, solver.boundary_indices, 
                np.setdiff1d(np.arange(len(solver.nodes)), solver.boundary_indices))
    forward_solver = FEMForwardSolver(resolution=forward_resolution, verbose=False, 
                                      mesh_data=mesh_data)
    
    np.random.seed(seed)
    u_clean = forward_solver.solve(sources_true)
    u_measured = u_clean + noise_level * np.random.randn(len(u_clean))
    
    solver.build_greens_matrix(verbose=False)
    G = solver.G
    interior_points = solver.interior_points
    
    # Build gradient operator for TV
    D = build_gradient_operator(interior_points)
    u_centered = u_measured - np.mean(u_measured)
    
    alphas = np.logspace(-6, -1, 15)
    
    lcurve_results = {'l1': {}, 'l2': {}, 'tv': {}}
    optimal_alphas = {}
    
    for method in ['l1', 'l2', 'tv']:
        if verbose:
            print(f"    L-curve for {method.upper()}...")
        
        residuals = []
        regularizers = []
        
        for alpha in alphas:
            n = G.shape[1]
            q_var = cp.Variable(n)
            constraints = [cp.sum(q_var) == 0]
            
            if method == 'l1':
                objective = cp.Minimize(
                    0.5 * cp.sum_squares(G @ q_var - u_centered) + 
                    alpha * cp.norm1(q_var)
                )
            elif method == 'l2':
                objective = cp.Minimize(
                    0.5 * cp.sum_squares(G @ q_var - u_centered) + 
                    0.5 * alpha * cp.sum_squares(q_var)
                )
            else:  # tv
                objective = cp.Minimize(
                    0.5 * cp.sum_squares(G @ q_var - u_centered) + 
                    alpha * cp.norm1(D @ q_var)
                )
            
            prob = cp.Problem(objective, constraints)
            try:
                prob.solve(solver=cp.ECOS, verbose=False)
                q = q_var.value if q_var.value is not None else np.zeros(n)
            except:
                try:
                    prob.solve(verbose=False)
                    q = q_var.value if q_var.value is not None else np.zeros(n)
                except:
                    q = np.zeros(n)
            
            res = np.linalg.norm(G @ q - u_centered)
            if method == 'l1':
                reg = np.sum(np.abs(q))
            elif method == 'l2':
                reg = np.linalg.norm(q)
            else:
                reg = np.sum(np.abs(D @ q))
            
            residuals.append(res)
            regularizers.append(reg)
            lcurve_results[method][float(alpha)] = [float(res), float(reg)]
        
        # Find corner
        idx = find_lcurve_corner(np.array(residuals), np.array(regularizers))
        optimal_alphas[method] = float(alphas[idx])
        
        if verbose:
            print(f"      Optimal alpha: {optimal_alphas[method]:.2e}")
    
    return lcurve_results, optimal_alphas


def _run_lcurve_analysis_fixed(domain_type: str, domain_params: dict,
                                sources_true: list, sensor_locations: np.ndarray,
                                u_measured: np.ndarray, forward_resolution: float,
                                source_resolution: float,
                                verbose: bool = True) -> Tuple[dict, dict]:
    """Run L-curve analysis with PRE-COMPUTED u_measured.
    
    This is the correct version: u_measured is generated ONCE at the start
    and passed to all calibration steps.
    
    Parameters
    ----------
    domain_type : str
        Domain type
    domain_params : dict
        Domain parameters
    sources_true : list
        True source configuration
    sensor_locations : array
        FIXED sensor locations
    u_measured : array
        PRE-COMPUTED measurement data (FIXED!)
    forward_resolution : float
        Forward mesh resolution
    source_resolution : float
        Source grid resolution
    verbose : bool
        Print progress
    """
    try:
        from .fem_solver import FEMLinearInverseSolver
        from .parameter_selection import build_gradient_operator
    except ImportError:
        from fem_solver import FEMLinearInverseSolver
        from parameter_selection import build_gradient_operator
    
    import cvxpy as cp
    
    # Build inverse solver with fixed sensor locations
    solver = _build_solver_for_domain(domain_type, domain_params,
                                       forward_resolution, source_resolution,
                                       sensor_locations)
    
    if solver is None:
        if verbose:
            print("    ERROR: Could not build solver")
        return {}, {'l1': 1e-4, 'l2': 1e-4, 'tv': 1e-2}
    
    solver.build_greens_matrix(verbose=False)
    G = solver.G
    interior_points = solver.interior_points
    n = G.shape[1]
    
    # Verify dimensions
    if G.shape[0] != len(u_measured):
        if verbose:
            print(f"    ERROR: Dimension mismatch G={G.shape}, u={len(u_measured)}")
        return {}, {'l1': 1e-4, 'l2': 1e-4, 'tv': 1e-2}
    
    # Build gradient operator for TV
    D = build_gradient_operator(interior_points)
    u_centered = u_measured - np.mean(u_measured)
    
    alphas = np.logspace(-6, -1, 15)
    
    lcurve_results = {'l1': {}, 'l2': {}, 'tv': {}}
    optimal_alphas = {}
    
    methods = [
        ('l1', lambda q, alpha: alpha * cp.norm1(q)),
        ('l2', lambda q, alpha: alpha * cp.sum_squares(q)),
        ('tv', lambda q, alpha: alpha * cp.norm1(D @ q))
    ]
    
    for method, regularizer in methods:
        if verbose:
            print(f"    L-curve for {method.upper()}...")
        
        residuals = []
        regularizers = []
        
        for alpha in alphas:
            try:
                q = cp.Variable(n)
                objective = cp.Minimize(cp.sum_squares(G @ q - u_centered) + 
                                       regularizer(q, alpha))
                constraints = [cp.sum(q) == 0]
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.SCS, verbose=False)
                
                if q.value is not None:
                    res = np.linalg.norm(G @ q.value - u_centered)
                    if method == 'l1':
                        reg = np.sum(np.abs(q.value))
                    elif method == 'l2':
                        reg = np.sum(q.value**2)
                    else:  # tv
                        reg = np.sum(np.abs(D @ q.value))
                    residuals.append(res)
                    regularizers.append(reg)
                else:
                    residuals.append(np.inf)
                    regularizers.append(np.inf)
            except:
                residuals.append(np.inf)
                regularizers.append(np.inf)
        
        # Store L-curve data
        for i, alpha in enumerate(alphas):
            if residuals[i] < np.inf:
                lcurve_results[method][float(alpha)] = {
                    'residual': float(residuals[i]),
                    'regularizer': float(regularizers[i])
                }
        
        # Find corner
        valid = [i for i in range(len(alphas)) if residuals[i] < np.inf]
        if len(valid) < 3:
            optimal_alphas[method] = 1e-4 if method != 'tv' else 1e-2
        else:
            log_res = np.log10([residuals[i] for i in valid])
            log_reg = np.log10([regularizers[i] + 1e-12 for i in valid])
            
            if len(valid) >= 3:
                curvature = []
                for i in range(1, len(valid)-1):
                    d1 = (log_res[i] - log_res[i-1], log_reg[i] - log_reg[i-1])
                    d2 = (log_res[i+1] - log_res[i], log_reg[i+1] - log_reg[i])
                    cross = d1[0]*d2[1] - d1[1]*d2[0]
                    curvature.append(cross)
                corner_idx = valid[1 + np.argmax(curvature)]
            else:
                corner_idx = valid[len(valid)//2]
            optimal_alphas[method] = alphas[corner_idx]
        
        if verbose:
            print(f"      Optimal alpha: {optimal_alphas[method]:.2e}")
    
    return lcurve_results, optimal_alphas


def calibrate_all_domains(domains: List[str] = None,
                          output_dir: str = 'results/calibration',
                          noise_level: float = 0.001,
                          seed: int = 42,
                          verbose: bool = True) -> CalibrationConfig:
    """
    Run calibration for all specified domains.
    
    Output structure (Option A - stable path + archive):
        output_dir/
        ├── calibration_config.json          # Always the current/active config
        ├── experiments.db                   # Database with all runs
        └── archive/
            └── calibration_2026-01-11_T14h57min_bbfa367d/
                ├── calibration_config.json  # Archived copy
                ├── disk_calibration.json
                └── ...
    
    Parameters
    ----------
    domains : list
        Domain types to calibrate. Default: all supported domains.
    output_dir : str
        Directory to save results
    noise_level : float
        Noise level for synthetic data
    seed : int
        Random seed
    verbose : bool
        Print progress
        
    Returns
    -------
    config : CalibrationConfig
        Complete calibration configuration
    """
    from datetime import datetime
    
    if domains is None:
        domains = list(DEFAULT_DOMAINS.keys())
    
    # Create identifiers
    timestamp = datetime.now().strftime("%Y-%m-%d_T%Hh%Mmin")
    import secrets
    hash_id = secrets.token_hex(4)
    calibration_id = f"calibration_{timestamp}_{hash_id}"
    
    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    archive_dir = os.path.join(output_dir, 'archive', calibration_id)
    os.makedirs(archive_dir, exist_ok=True)
    
    # Paths
    stable_config_path = os.path.join(output_dir, 'calibration_config.json')
    archive_config_path = os.path.join(archive_dir, 'calibration_config.json')
    db_path = os.path.join(output_dir, 'experiments.db')
    
    if verbose:
        print("="*70)
        print("INVERSE SOURCE LOCALIZATION - PARAMETER CALIBRATION")
        print("="*70)
        print(f"Calibration ID: {calibration_id}")
        print(f"Domains: {domains}")
        print(f"Noise level: {noise_level}")
        print(f"Archive: {archive_dir}")
        print(f"Stable config: {stable_config_path}")
    
    domain_calibrations = {}
    
    # Try to initialize database
    db = None
    try:
        from .experiment_tracker import ExperimentDatabase
        db = ExperimentDatabase(db_path)
    except Exception as e:
        if verbose:
            print(f"Note: Could not initialize experiment database: {e}")
    
    for domain_type in domains:
        domain_params = DEFAULT_DOMAINS.get(domain_type, {})
        
        try:
            calibration = calibrate_domain(
                domain_type=domain_type,
                domain_params=domain_params,
                noise_level=noise_level,
                seed=seed,
                verbose=verbose
            )
            domain_calibrations[domain_type] = calibration
            
            # Save to archive directory
            _save_domain_calibration(calibration, archive_dir)
            
            # Save to database
            if db:
                try:
                    db.insert_calibration({
                        'calibration_id': f"{calibration_id}_{domain_type}",
                        'timestamp': timestamp,
                        'domain_type': domain_type,
                        'domain_params': calibration.domain_params,
                        'forward_mesh_resolution': calibration.forward_mesh_resolution,
                        'source_grid_resolution': calibration.source_grid_resolution,
                        'alpha_l1': calibration.alpha_l1,
                        'alpha_l2': calibration.alpha_l2,
                        'alpha_tv': calibration.alpha_tv,
                        'forward_convergence': calibration.forward_convergence,
                        'source_convergence': calibration.source_convergence,
                        'lcurve_l1': calibration.lcurve_l1,
                        'lcurve_l2': calibration.lcurve_l2,
                        'lcurve_tv': calibration.lcurve_tv,
                        'calibration_time_seconds': calibration.calibration_time_seconds,
                        'config_path': stable_config_path
                    })
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Could not save to database: {e}")
            
        except Exception as e:
            print(f"\nWARNING: Calibration failed for {domain_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Close database
    if db:
        db.close()
    
    # Create final config
    config = CalibrationConfig(
        version="7.23",
        calibration_date=datetime.now().isoformat(),
        domains={k: asdict(v) for k, v in domain_calibrations.items()}
    )
    
    # Save to BOTH locations: stable path and archive
    save_calibration_config(config, stable_config_path)
    save_calibration_config(config, archive_config_path)
    
    if verbose:
        print("\n" + "="*70)
        print("CALIBRATION COMPLETE")
        print("="*70)
        print(f"Stable config: {stable_config_path}")
        print(f"Archive: {archive_config_path}")
        print(f"Database: {db_path}")
        _print_summary(config)
    
    return config


def _save_domain_calibration(calibration: DomainCalibration, output_dir: str):
    """Save individual domain calibration."""
    path = os.path.join(output_dir, f'{calibration.domain_type}_calibration.json')
    with open(path, 'w') as f:
        json.dump(asdict(calibration), f, indent=2)


def save_calibration_config(config: CalibrationConfig, path: str):
    """Save calibration config to JSON file."""
    # Convert to serializable dict
    data = {
        'version': config.version,
        'calibration_date': config.calibration_date,
        'default_forward_resolution': config.default_forward_resolution,
        'default_source_resolution': config.default_source_resolution,
        'default_alpha_l1': config.default_alpha_l1,
        'default_alpha_l2': config.default_alpha_l2,
        'default_alpha_tv': config.default_alpha_tv,
        'domains': config.domains
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved calibration config to: {path}")


def load_calibration_config(path: str) -> dict:
    """Load calibration config from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def get_domain_params(config: dict, domain_type: str) -> dict:
    """
    Get calibrated parameters for a specific domain.
    
    Parameters
    ----------
    config : dict
        Loaded calibration config
    domain_type : str
        Domain type
        
    Returns
    -------
    params : dict
        Dictionary with keys:
        - forward_mesh_resolution
        - source_grid_resolution
        - alpha_l1, alpha_l2, alpha_tv
    """
    if domain_type in config.get('domains', {}):
        d = config['domains'][domain_type]
        return {
            'forward_mesh_resolution': d['forward_mesh_resolution'],
            'source_grid_resolution': d['source_grid_resolution'],
            'alpha_l1': d['alpha_l1'],
            'alpha_l2': d['alpha_l2'],
            'alpha_tv': d['alpha_tv'],
            'domain_params': d.get('domain_params', {})
        }
    else:
        # Return defaults
        return {
            'forward_mesh_resolution': config.get('default_forward_resolution', 0.08),
            'source_grid_resolution': config.get('default_source_resolution', 0.12),
            'alpha_l1': config.get('default_alpha_l1', 1e-4),
            'alpha_l2': config.get('default_alpha_l2', 1e-4),
            'alpha_tv': config.get('default_alpha_tv', 1e-2),
            'domain_params': DEFAULT_DOMAINS.get(domain_type, {})
        }


def _print_summary(config: CalibrationConfig):
    """Print calibration summary table."""
    print("\nCalibration Summary:")
    print("-" * 90)
    print(f"{'Domain':<10} {'Fwd Mesh':<10} {'Src Grid':<10} {'α L1':<12} {'α L2':<12} {'α TV':<12} {'Time':<8}")
    print("-" * 90)
    
    for domain_type, cal in config.domains.items():
        if isinstance(cal, dict):
            print(f"{domain_type:<10} {cal['forward_mesh_resolution']:<10.3f} "
                  f"{cal['source_grid_resolution']:<10.3f} "
                  f"{cal['alpha_l1']:<12.2e} {cal['alpha_l2']:<12.2e} "
                  f"{cal['alpha_tv']:<12.2e} {cal['calibration_time_seconds']:<8.1f}s")


def plot_calibration_results(config_path: str, output_dir: str = None):
    """
    Generate plots for calibration results (non-blocking).
    
    Parameters
    ----------
    config_path : str
        Path to calibration config JSON
    output_dir : str
        Output directory for plots (default: same as config)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    config = load_calibration_config(config_path)
    
    if output_dir is None:
        output_dir = os.path.dirname(config_path)
    
    for domain_type, cal in config.get('domains', {}).items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: Forward convergence
        ax = axes[0]
        fwd = cal['forward_convergence']
        h_vals = sorted([float(k) for k in fwd.keys()])
        errors = [fwd[str(h) if str(h) in fwd else h] for h in h_vals]
        ax.loglog(h_vals, errors, 'bo-', linewidth=2, markersize=8)
        ax.axhline(0.01, color='r', linestyle='--', label='1% threshold')
        ax.axvline(cal['forward_mesh_resolution'], color='g', linestyle=':', 
                   label=f"Optimal: {cal['forward_mesh_resolution']}")
        ax.set_xlabel('Mesh resolution h')
        ax.set_ylabel('Relative error')
        ax.set_title('Forward Mesh Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        # Plot 2: Source grid convergence (localization vs resolution with optimal alpha)
        ax = axes[1]
        src = cal['source_convergence']
        h_vals = sorted([float(k) for k in src.keys()])
        
        # Handle both old format (float) and new format (dict)
        if h_vals and isinstance(src.get(str(h_vals[0]), src.get(h_vals[0])), dict):
            # New format with localization and optimal_alpha
            locs = []
            for h in h_vals:
                entry = src.get(str(h), src.get(h, {}))
                locs.append(entry.get('localization', 0))
            
            ax.plot(h_vals, locs, 'gs-', linewidth=2, markersize=8)
            ax.axvline(cal['source_grid_resolution'], color='r', linestyle=':', 
                       linewidth=2, label=f"Best: {cal['source_grid_resolution']}")
            ax.set_xlabel('Source grid resolution h')
            ax.set_ylabel('Localization score (with optimal α)')
            ax.set_title('Source Grid Convergence\n(L-curve α per resolution)')
            ax.invert_xaxis()
        else:
            # Old format (localization scores directly)
            locs = [src.get(str(h), src.get(h, 0)) for h in h_vals]
            ax.plot(h_vals, locs, 'gs-', linewidth=2, markersize=8)
            ax.axvline(cal['source_grid_resolution'], color='g', linestyle=':', 
                       label=f"Optimal: {cal['source_grid_resolution']}")
            ax.set_xlabel('Source grid resolution h')
            ax.set_ylabel('Localization score')
            ax.set_title('Source Grid Convergence')
            ax.invert_xaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: L-curves
        ax = axes[2]
        colors = {'l1': 'blue', 'l2': 'green', 'tv': 'red'}
        for method in ['l1', 'l2', 'tv']:
            lc = cal.get(f'lcurve_{method}', {})
            if not lc:
                continue
            
            # Handle both string and numeric keys
            try:
                # Try to convert all keys to float for sorting
                key_val_pairs = []
                for k, v in lc.items():
                    try:
                        key_float = float(k)
                        key_val_pairs.append((key_float, v))
                    except (ValueError, TypeError):
                        continue
                
                if not key_val_pairs:
                    continue
                    
                key_val_pairs.sort(key=lambda x: x[0])
                alphas = [kv[0] for kv in key_val_pairs]
                
                # Handle both list [res, reg] and dict {'residual': ..., 'regularizer': ...} formats
                residuals = []
                regularizers = []
                for kv in key_val_pairs:
                    v = kv[1]
                    if isinstance(v, dict):
                        residuals.append(v.get('residual', v.get(0, 0)))
                        regularizers.append(v.get('regularizer', v.get(1, 0)))
                    elif isinstance(v, (list, tuple)):
                        residuals.append(v[0])
                        regularizers.append(v[1])
                    else:
                        continue
                
                if not residuals:
                    continue
                
                ax.loglog(residuals, regularizers, 'o-', color=colors[method], 
                         label=f"{method.upper()}: α={cal[f'alpha_{method}']:.1e}")
            except Exception as e:
                print(f"Warning: Could not plot L-curve for {method}: {e}")
                continue
                
        ax.set_xlabel('Residual ||Gq - u||')
        ax.set_ylabel('Regularizer')
        ax.set_title('L-Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{domain_type.upper()} Calibration Results', fontsize=14)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'{domain_type}_calibration_plots.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Calibrate inverse source parameters')
    parser.add_argument('--domains', type=str, nargs='+', default=None,
                       help='Domains to calibrate (default: all)')
    parser.add_argument('--output-dir', type=str, default='results/calibration',
                       help='Output directory')
    parser.add_argument('--noise', type=float, default=0.001,
                       help='Noise level')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots after calibration')
    
    args = parser.parse_args()
    
    config = calibrate_all_domains(
        domains=args.domains,
        output_dir=args.output_dir,
        noise_level=args.noise,
        seed=args.seed,
        verbose=True
    )
    
    if args.plot:
        config_path = os.path.join(args.output_dir, 'calibration_config.json')
        plot_calibration_results(config_path)
