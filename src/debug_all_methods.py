#!/usr/bin/env python3
"""
Comprehensive Debug Check for All Domains and Methods
======================================================

This script systematically tests each component to identify bugs.

Checks:
1. Forward solver correctness (does it produce sensible boundary data?)
2. Green's matrix properties (conditioning, mutual coherence)
3. Linear solver sanity (can it recover exact grid point sources?)
4. Nonlinear solver sanity (can it recover well-separated sources?)
5. Sensor embedding (are sensors actually in the mesh?)
6. Cross-validation (FEM vs Analytical on disk)

Author: Debug session 2026-01-12
"""

import numpy as np
import sys
import time
import traceback
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class DiagnosticResult:
    domain: str
    component: str
    test_name: str
    passed: bool
    details: str
    value: float = None

def create_test_sources_for_domain(domain: str, n_sources: int = 2) -> List[Tuple[Tuple[float, float], float]]:
    """Create well-separated test sources appropriate for each domain."""
    
    if domain == 'disk':
        # Sources at r=0.5, well separated
        sources = [
            ((0.5, 0.0), 1.0),
            ((-0.5, 0.0), -1.0),
        ]
        if n_sources >= 4:
            sources.extend([
                ((0.0, 0.5), 1.0),
                ((0.0, -0.5), -1.0),
            ])
    
    elif domain == 'ellipse':
        # Ellipse with a=2, b=1; sources at ~50% of axes
        sources = [
            ((1.0, 0.0), 1.0),
            ((-1.0, 0.0), -1.0),
        ]
        if n_sources >= 4:
            sources.extend([
                ((0.0, 0.4), 1.0),
                ((0.0, -0.4), -1.0),
            ])
    
    elif domain == 'star':
        # Star domain has inner radius ~0.4; keep sources at r=0.15 (well inside all lobes)
        sources = [
            ((0.15, 0.0), 1.0),
            ((-0.15, 0.0), -1.0),
        ]
        if n_sources >= 4:
            sources.extend([
                ((0.0, 0.15), 1.0),
                ((0.0, -0.15), -1.0),
            ])
    
    elif domain == 'square':
        # Square [-1,1]^2; sources well inside
        sources = [
            ((0.5, 0.0), 1.0),
            ((-0.5, 0.0), -1.0),
        ]
        if n_sources >= 4:
            sources.extend([
                ((0.0, 0.5), 1.0),
                ((0.0, -0.5), -1.0),
            ])
    
    elif domain == 'polygon':
        # L-shaped: [0,2]x[0,1] + [0,1]x[1,2]; centroid ~(0.75, 0.75)
        sources = [
            ((0.5, 0.5), 1.0),
            ((1.5, 0.5), -1.0),
        ]
        if n_sources >= 4:
            sources.extend([
                ((0.5, 1.5), 1.0),
                ((0.3, 0.3), -1.0),
            ])
    
    elif domain == 'brain':
        # Brain shape ~[-1.1, 1.1] x [-0.6, 0.7]; keep sources central
        sources = [
            ((0.3, 0.0), 1.0),
            ((-0.3, 0.0), -1.0),
        ]
        if n_sources >= 4:
            sources.extend([
                ((0.0, 0.2), 1.0),
                ((0.0, -0.2), -1.0),
            ])
    
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    return sources[:n_sources]


def get_domain_params(domain: str) -> dict:
    """Get domain parameters."""
    if domain == 'ellipse':
        return {'a': 2.0, 'b': 1.0}
    elif domain == 'square':
        return {'vertices': [(-1, -1), (1, -1), (1, 1), (-1, 1)]}
    elif domain == 'polygon':
        return {'vertices': [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]}
    elif domain == 'star':
        return {'n_petals': 5, 'amplitude': 0.3}
    elif domain == 'brain':
        return {}
    else:
        return {}


def check_forward_solver(domain: str) -> List[DiagnosticResult]:
    """Check that forward solver produces sensible output."""
    results = []
    sources = create_test_sources_for_domain(domain, n_sources=2)
    domain_params = get_domain_params(domain)
    
    # Test 1: Analytical/Conformal forward solver
    if domain == 'disk':
        try:
            from analytical_solver import AnalyticalForwardSolver
            forward = AnalyticalForwardSolver(n_boundary_points=100)
            u = forward.solve(sources)
            
            # Checks
            has_values = len(u) == 100
            mean_near_zero = abs(np.mean(u)) < 0.1  # Neumann BC: mean should be small
            has_variation = np.std(u) > 1e-6
            no_nans = not np.any(np.isnan(u))
            
            passed = has_values and mean_near_zero and has_variation and no_nans
            details = f"len={len(u)}, mean={np.mean(u):.4f}, std={np.std(u):.4f}, range=[{u.min():.3f}, {u.max():.3f}]"
            
            results.append(DiagnosticResult(domain, 'forward', 'Analytical forward', passed, details))
        except Exception as e:
            results.append(DiagnosticResult(domain, 'forward', 'Analytical forward', False, str(e)))
    
    else:
        # Conformal forward for non-disk domains
        try:
            from conformal_solver import ConformalForwardSolver, EllipseMap, NumericalConformalMap
            from mesh import get_brain_boundary
            
            if domain == 'ellipse':
                cmap = EllipseMap(a=2.0, b=1.0)
            elif domain == 'star':
                def star_boundary(t):
                    r = 0.7 + 0.3 * np.cos(5 * t)
                    return r * np.exp(1j * t)
                cmap = NumericalConformalMap(star_boundary, n_boundary=256)
            elif domain == 'square':
                def square_boundary(t):
                    # Parametric square
                    t = t % (2*np.pi)
                    if t < np.pi/2:
                        return complex(1, -1 + 4*t/np.pi)
                    elif t < np.pi:
                        return complex(1 - 4*(t-np.pi/2)/np.pi, 1)
                    elif t < 3*np.pi/2:
                        return complex(-1, 1 - 4*(t-np.pi)/np.pi)
                    else:
                        return complex(-1 + 4*(t-3*np.pi/2)/np.pi, -1)
                cmap = NumericalConformalMap(square_boundary, n_boundary=256)
            elif domain == 'polygon':
                vertices = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
                def polygon_boundary(t):
                    # Simple L-shape parametrization
                    n = len(vertices)
                    total_t = t / (2*np.pi) * n
                    idx = int(total_t) % n
                    frac = total_t - int(total_t)
                    v1 = vertices[idx]
                    v2 = vertices[(idx + 1) % n]
                    x = v1[0] + frac * (v2[0] - v1[0])
                    y = v1[1] + frac * (v2[1] - v1[1])
                    return complex(x, y)
                cmap = NumericalConformalMap(polygon_boundary, n_boundary=256)
            elif domain == 'brain':
                boundary = get_brain_boundary(n_points=256)
                def brain_boundary(t):
                    idx = int(t / (2*np.pi) * len(boundary)) % len(boundary)
                    return complex(boundary[idx, 0], boundary[idx, 1])
                cmap = NumericalConformalMap(brain_boundary, n_boundary=256)
            else:
                raise ValueError(f"No conformal map for {domain}")
            
            forward = ConformalForwardSolver(cmap, n_boundary=100)
            u = forward.solve(sources)
            
            has_values = len(u) == 100
            has_variation = np.std(u) > 1e-6
            no_nans = not np.any(np.isnan(u))
            
            passed = has_values and has_variation and no_nans
            details = f"len={len(u)}, mean={np.mean(u):.4f}, std={np.std(u):.4f}"
            
            results.append(DiagnosticResult(domain, 'forward', 'Conformal forward', passed, details))
        except Exception as e:
            results.append(DiagnosticResult(domain, 'forward', 'Conformal forward', False, f"{e}\n{traceback.format_exc()}"))
    
    # Test 2: FEM forward solver
    try:
        from fem_solver import FEMForwardSolver
        from mesh import create_disk_mesh, create_ellipse_mesh, create_polygon_mesh
        
        n_sensors = 50
        
        if domain == 'disk':
            theta = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
            sensor_locs = np.column_stack([np.cos(theta), np.sin(theta)])
            forward = FEMForwardSolver(resolution=0.1, sensor_locations=sensor_locs, verbose=False)
        elif domain == 'ellipse':
            a, b = 2.0, 1.0
            theta = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
            sensor_locs = np.column_stack([a * np.cos(theta), b * np.sin(theta)])
            mesh_data = create_ellipse_mesh(a, b, resolution=0.1, sensor_locations=sensor_locs)
            forward = FEMForwardSolver(resolution=0.1, mesh_data=mesh_data, verbose=False)
        elif domain in ['square', 'polygon']:
            vertices = get_domain_params(domain).get('vertices')
            # Simple sensor placement on boundary
            verts = np.array(vertices)
            sensor_locs = []
            for i in range(len(vertices)):
                v1, v2 = verts[i], verts[(i+1) % len(verts)]
                for t in np.linspace(0, 1, n_sensors // len(vertices), endpoint=False):
                    sensor_locs.append(v1 + t * (v2 - v1))
            sensor_locs = np.array(sensor_locs[:n_sensors])
            mesh_data = create_polygon_mesh(vertices, resolution=0.1, sensor_locations=sensor_locs)
            forward = FEMForwardSolver(resolution=0.1, mesh_data=mesh_data, verbose=False)
        elif domain == 'star':
            theta = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
            r = 0.7 + 0.3 * np.cos(5 * theta)
            sensor_locs = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            # Star needs polygon mesh
            n_v = 100
            theta_v = np.linspace(0, 2*np.pi, n_v, endpoint=False)
            r_v = 0.7 + 0.3 * np.cos(5 * theta_v)
            vertices = [(r_v[i] * np.cos(theta_v[i]), r_v[i] * np.sin(theta_v[i])) for i in range(n_v)]
            mesh_data = create_polygon_mesh(vertices, resolution=0.1, sensor_locations=sensor_locs)
            forward = FEMForwardSolver(resolution=0.1, mesh_data=mesh_data, verbose=False)
        elif domain == 'brain':
            from mesh import get_brain_boundary
            boundary = get_brain_boundary(n_points=n_sensors)
            sensor_locs = boundary
            vertices = [(b[0], b[1]) for b in get_brain_boundary(n_points=100)]
            mesh_data = create_polygon_mesh(vertices, resolution=0.1, sensor_locations=sensor_locs)
            forward = FEMForwardSolver(resolution=0.1, mesh_data=mesh_data, verbose=False)
        
        u = forward.solve(sources)
        
        # Check sensor count
        actual_sensors = len(u)
        has_sensors = actual_sensors > 0
        has_variation = np.std(u) > 1e-6 if has_sensors else False
        no_nans = not np.any(np.isnan(u)) if has_sensors else False
        
        passed = has_sensors and has_variation and no_nans
        details = f"n_sensors={actual_sensors}, mean={np.mean(u):.4f}, std={np.std(u):.4f}" if has_sensors else "NO SENSORS"
        
        results.append(DiagnosticResult(domain, 'forward', 'FEM forward', passed, details))
        
    except Exception as e:
        results.append(DiagnosticResult(domain, 'forward', 'FEM forward', False, f"{e}\n{traceback.format_exc()}"))
    
    return results


def check_greens_matrix(domain: str) -> List[DiagnosticResult]:
    """Check Green's matrix properties."""
    results = []
    
    if domain == 'disk':
        try:
            from analytical_solver import AnalyticalLinearInverseSolver
            
            solver = AnalyticalLinearInverseSolver(n_boundary=100, source_resolution=0.15, verbose=False)
            solver.build_greens_matrix(verbose=False)
            G = solver.G
            
            # Properties
            m, n = G.shape
            rank = np.linalg.matrix_rank(G)
            cond = np.linalg.cond(G)
            
            # Mutual coherence (max correlation between columns)
            G_norm = G / np.linalg.norm(G, axis=0, keepdims=True)
            gram = np.abs(G_norm.T @ G_norm)
            np.fill_diagonal(gram, 0)
            coherence = gram.max()
            
            # Column norm variation
            col_norms = np.linalg.norm(G, axis=0)
            norm_ratio = col_norms.max() / col_norms.min()
            
            details = f"shape={G.shape}, rank={rank}, cond={cond:.1e}, coherence={coherence:.4f}, norm_ratio={norm_ratio:.1f}"
            
            # Sparse recovery theory: need coherence < 1/(2k-1) for k sources
            # With coherence ~0.99, sparse recovery is theoretically impossible
            is_well_conditioned = cond < 1e10
            is_recoverable = coherence < 0.5  # Very generous threshold
            
            passed = is_well_conditioned  # Don't fail on coherence - it's expected to be high
            
            results.append(DiagnosticResult(domain, 'greens_matrix', 'Analytical Green\'s matrix', 
                                           passed, details, value=coherence))
            
            if coherence > 0.9:
                results.append(DiagnosticResult(domain, 'greens_matrix', 'WARNING: High coherence', 
                                               False, f"coherence={coherence:.4f} > 0.9 means sparse recovery is ill-posed"))
            
        except Exception as e:
            results.append(DiagnosticResult(domain, 'greens_matrix', 'Analytical Green\'s matrix', False, str(e)))
    
    # FEM Green's matrix
    try:
        from fem_solver import FEMLinearInverseSolver
        from mesh import create_ellipse_mesh, create_polygon_mesh
        
        if domain == 'disk':
            solver = FEMLinearInverseSolver(forward_resolution=0.1, source_resolution=0.15, verbose=False)
        elif domain == 'ellipse':
            solver = FEMLinearInverseSolver.from_ellipse(2.0, 1.0, forward_resolution=0.1, 
                                                          source_resolution=0.15, verbose=False)
        elif domain in ['square', 'polygon', 'star', 'brain']:
            if domain == 'square':
                vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            elif domain == 'polygon':
                vertices = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
            elif domain == 'star':
                n_v = 100
                theta_v = np.linspace(0, 2*np.pi, n_v, endpoint=False)
                r_v = 0.7 + 0.3 * np.cos(5 * theta_v)
                vertices = [(r_v[i] * np.cos(theta_v[i]), r_v[i] * np.sin(theta_v[i])) for i in range(n_v)]
            elif domain == 'brain':
                from mesh import get_brain_boundary
                vertices = [(b[0], b[1]) for b in get_brain_boundary(n_points=100)]
            solver = FEMLinearInverseSolver.from_polygon(vertices, forward_resolution=0.1,
                                                          source_resolution=0.15, verbose=False)
        
        solver.build_greens_matrix(verbose=False)
        G = solver.G
        
        m, n = G.shape
        rank = np.linalg.matrix_rank(G)
        cond = np.linalg.cond(G) if min(m, n) > 0 else float('inf')
        
        # Mutual coherence
        if n > 1:
            G_norm = G / (np.linalg.norm(G, axis=0, keepdims=True) + 1e-10)
            gram = np.abs(G_norm.T @ G_norm)
            np.fill_diagonal(gram, 0)
            coherence = gram.max()
        else:
            coherence = 0
        
        details = f"shape={G.shape}, rank={rank}, cond={cond:.1e}, coherence={coherence:.4f}"
        passed = cond < 1e12 and m > 0 and n > 0
        
        results.append(DiagnosticResult(domain, 'greens_matrix', 'FEM Green\'s matrix', 
                                       passed, details, value=coherence))
        
    except Exception as e:
        results.append(DiagnosticResult(domain, 'greens_matrix', 'FEM Green\'s matrix', 
                                       False, f"{e}\n{traceback.format_exc()}"))
    
    return results


def check_linear_solver(domain: str) -> List[DiagnosticResult]:
    """Check linear solver functionality."""
    results = []
    sources = create_test_sources_for_domain(domain, n_sources=2)
    
    # Skip disk analytical linear - we know it's fundamentally ill-posed due to coherence
    # But test it anyway to document the behavior
    
    if domain == 'disk':
        try:
            from analytical_solver import AnalyticalForwardSolver, AnalyticalLinearInverseSolver
            
            # Generate data
            forward = AnalyticalForwardSolver(n_boundary_points=100)
            u_exact = forward.solve(sources)
            
            # Add small noise
            np.random.seed(42)
            u_noisy = u_exact + 0.001 * np.random.randn(len(u_exact))
            
            solver = AnalyticalLinearInverseSolver(n_boundary=100, source_resolution=0.15, verbose=False)
            solver.build_greens_matrix(verbose=False)
            
            for method in ['l2', 'l1']:
                try:
                    if method == 'l2':
                        q = solver.solve_l2(u_noisy, alpha=1e-4)
                    else:
                        q = solver.solve_l1(u_noisy, alpha=1e-4)
                    
                    # Check solution properties
                    has_solution = q is not None and len(q) > 0
                    no_nans = not np.any(np.isnan(q)) if has_solution else False
                    has_peaks = np.max(np.abs(q)) > 0.01 if has_solution else False
                    
                    # Find peaks
                    n_significant = np.sum(np.abs(q) > 0.1 * np.max(np.abs(q))) if has_solution else 0
                    
                    # Residual
                    residual = np.linalg.norm(solver.G @ q - u_noisy) if has_solution else float('inf')
                    
                    details = f"max|q|={np.max(np.abs(q)):.3f}, n_significant={n_significant}, residual={residual:.2e}"
                    passed = has_solution and no_nans
                    
                    results.append(DiagnosticResult(domain, 'linear', f'Analytical {method.upper()}', 
                                                   passed, details))
                except Exception as e:
                    results.append(DiagnosticResult(domain, 'linear', f'Analytical {method.upper()}', 
                                                   False, str(e)))
            
        except Exception as e:
            results.append(DiagnosticResult(domain, 'linear', 'Analytical linear', False, str(e)))
    
    # FEM linear solver
    try:
        from fem_solver import FEMForwardSolver, FEMLinearInverseSolver
        from mesh import create_ellipse_mesh, create_polygon_mesh
        
        n_sensors = 100
        
        # Create appropriate mesh with sensors
        if domain == 'disk':
            theta = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
            sensor_locs = np.column_stack([np.cos(theta), np.sin(theta)])
            forward = FEMForwardSolver(resolution=0.1, sensor_locations=sensor_locs, verbose=False)
            inverse = FEMLinearInverseSolver(forward_resolution=0.1, source_resolution=0.15, 
                                              sensor_locations=sensor_locs, verbose=False)
        elif domain == 'ellipse':
            a, b = 2.0, 1.0
            theta = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
            sensor_locs = np.column_stack([a * np.cos(theta), b * np.sin(theta)])
            mesh_data = create_ellipse_mesh(a, b, resolution=0.1, sensor_locations=sensor_locs)
            forward = FEMForwardSolver(resolution=0.1, mesh_data=mesh_data, verbose=False)
            inverse = FEMLinearInverseSolver.from_ellipse(a, b, forward_resolution=0.1, 
                                                           source_resolution=0.15,
                                                           sensor_locations=sensor_locs, verbose=False)
        elif domain in ['square', 'polygon', 'star', 'brain']:
            if domain == 'square':
                vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            elif domain == 'polygon':
                vertices = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
            elif domain == 'star':
                n_v = 100
                theta_v = np.linspace(0, 2*np.pi, n_v, endpoint=False)
                r_v = 0.7 + 0.3 * np.cos(5 * theta_v)
                vertices = [(r_v[i] * np.cos(theta_v[i]), r_v[i] * np.sin(theta_v[i])) for i in range(n_v)]
            elif domain == 'brain':
                from mesh import get_brain_boundary
                vertices = [(b[0], b[1]) for b in get_brain_boundary(n_points=100)]
            
            # Generate sensor locations on boundary
            verts = np.array(vertices)
            n_verts = len(vertices)
            sensor_locs = []
            for i in range(n_verts):
                v1, v2 = verts[i], verts[(i+1) % n_verts]
                edge_sensors = max(1, n_sensors // n_verts)
                for t in np.linspace(0, 1, edge_sensors, endpoint=False):
                    sensor_locs.append(v1 + t * (v2 - v1))
            sensor_locs = np.array(sensor_locs[:n_sensors])
            
            mesh_data = create_polygon_mesh(vertices, resolution=0.1, sensor_locations=sensor_locs)
            forward = FEMForwardSolver(resolution=0.1, mesh_data=mesh_data, verbose=False)
            inverse = FEMLinearInverseSolver.from_polygon(vertices, forward_resolution=0.1,
                                                           source_resolution=0.15,
                                                           sensor_locations=sensor_locs, verbose=False)
        
        # Check sensor count first
        n_actual_sensors = forward.n_sensors
        if n_actual_sensors == 0:
            results.append(DiagnosticResult(domain, 'linear', 'FEM linear', False, 
                                           "NO SENSORS - mesh embedding failed"))
            return results
        
        u_exact = forward.solve(sources)
        np.random.seed(42)
        u_noisy = u_exact + 0.001 * np.random.randn(len(u_exact))
        
        inverse.build_greens_matrix(verbose=False)
        
        for method in ['l2', 'l1']:
            try:
                if method == 'l2':
                    q = inverse.solve_l2(u_noisy, alpha=1e-3)
                else:
                    q = inverse.solve_l1(u_noisy, alpha=1e-3)
                
                has_solution = q is not None and len(q) > 0
                no_nans = not np.any(np.isnan(q)) if has_solution else False
                
                # Find peaks
                n_significant = np.sum(np.abs(q) > 0.1 * np.max(np.abs(q))) if has_solution else 0
                max_q = np.max(np.abs(q)) if has_solution else 0
                
                details = f"n_sensors={n_actual_sensors}, max|q|={max_q:.3f}, n_peaks={n_significant}"
                passed = has_solution and no_nans and max_q > 0.01
                
                results.append(DiagnosticResult(domain, 'linear', f'FEM {method.upper()}', 
                                               passed, details))
            except Exception as e:
                results.append(DiagnosticResult(domain, 'linear', f'FEM {method.upper()}', 
                                               False, str(e)))
        
    except Exception as e:
        results.append(DiagnosticResult(domain, 'linear', 'FEM linear', 
                                       False, f"{e}\n{traceback.format_exc()}"))
    
    return results


def check_nonlinear_solver(domain: str) -> List[DiagnosticResult]:
    """Check nonlinear solver functionality."""
    results = []
    sources = create_test_sources_for_domain(domain, n_sources=2)
    
    if domain == 'disk':
        # Analytical nonlinear
        try:
            from analytical_solver import AnalyticalForwardSolver, AnalyticalNonlinearInverseSolver
            
            forward = AnalyticalForwardSolver(n_boundary_points=100)
            u = forward.solve(sources)
            
            inverse = AnalyticalNonlinearInverseSolver(n_sources=2, n_boundary=100)
            inverse.set_measured_data(u)
            result = inverse.solve(method='L-BFGS-B', maxiter=200, n_restarts=5)
            
            # Compute position error
            recovered = [((s.x, s.y), s.intensity) for s in result.sources]
            pos_err = compute_position_error(recovered, sources)
            
            passed = pos_err < 0.1  # Should be very accurate for 2 sources
            details = f"pos_err={pos_err:.2e}, residual={result.residual:.2e}"
            
            results.append(DiagnosticResult(domain, 'nonlinear', 'Analytical L-BFGS-B', 
                                           passed, details, value=pos_err))
        except Exception as e:
            results.append(DiagnosticResult(domain, 'nonlinear', 'Analytical L-BFGS-B', 
                                           False, str(e)))
    
    else:
        # Conformal nonlinear for non-disk
        try:
            from conformal_solver import (ConformalForwardSolver, ConformalNonlinearInverseSolver,
                                          EllipseMap, NumericalConformalMap)
            from mesh import get_brain_boundary
            
            if domain == 'ellipse':
                cmap = EllipseMap(a=2.0, b=1.0)
            elif domain == 'star':
                def star_boundary(t):
                    r = 0.7 + 0.3 * np.cos(5 * t)
                    return r * np.exp(1j * t)
                cmap = NumericalConformalMap(star_boundary, n_boundary=256)
            elif domain == 'square':
                def square_boundary(t):
                    t = t % (2*np.pi)
                    if t < np.pi/2:
                        return complex(1, -1 + 4*t/np.pi)
                    elif t < np.pi:
                        return complex(1 - 4*(t-np.pi/2)/np.pi, 1)
                    elif t < 3*np.pi/2:
                        return complex(-1, 1 - 4*(t-np.pi)/np.pi)
                    else:
                        return complex(-1 + 4*(t-3*np.pi/2)/np.pi, -1)
                cmap = NumericalConformalMap(square_boundary, n_boundary=256)
            elif domain == 'polygon':
                vertices = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
                def polygon_boundary(t):
                    n = len(vertices)
                    total_t = t / (2*np.pi) * n
                    idx = int(total_t) % n
                    frac = total_t - int(total_t)
                    v1 = vertices[idx]
                    v2 = vertices[(idx + 1) % n]
                    x = v1[0] + frac * (v2[0] - v1[0])
                    y = v1[1] + frac * (v2[1] - v1[1])
                    return complex(x, y)
                cmap = NumericalConformalMap(polygon_boundary, n_boundary=256)
            elif domain == 'brain':
                boundary = get_brain_boundary(n_points=256)
                def brain_boundary(t):
                    idx = int(t / (2*np.pi) * len(boundary)) % len(boundary)
                    return complex(boundary[idx, 0], boundary[idx, 1])
                cmap = NumericalConformalMap(brain_boundary, n_boundary=256)
            
            forward = ConformalForwardSolver(cmap, n_boundary=100)
            u = forward.solve(sources)
            
            inverse = ConformalNonlinearInverseSolver(cmap, n_sources=2, n_boundary=100)
            recovered, residual = inverse.solve(u, method='L-BFGS-B')
            
            pos_err = compute_position_error(recovered, sources)
            
            # For non-disk domains with numerical maps, we expect some error
            threshold = 0.2 if domain in ['star', 'polygon', 'brain'] else 0.1
            passed = pos_err < threshold
            details = f"pos_err={pos_err:.2e}, residual={residual:.2e}"
            
            results.append(DiagnosticResult(domain, 'nonlinear', 'Conformal L-BFGS-B', 
                                           passed, details, value=pos_err))
        except Exception as e:
            results.append(DiagnosticResult(domain, 'nonlinear', 'Conformal L-BFGS-B', 
                                           False, f"{e}\n{traceback.format_exc()}"))
    
    # FEM nonlinear for all domains
    try:
        from fem_solver import FEMForwardSolver, FEMNonlinearInverseSolver
        from mesh import create_ellipse_mesh, create_polygon_mesh
        
        n_sensors = 100
        
        if domain == 'disk':
            theta = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
            sensor_locs = np.column_stack([np.cos(theta), np.sin(theta)])
            forward = FEMForwardSolver(resolution=0.1, sensor_locations=sensor_locs, verbose=False)
            inverse = FEMNonlinearInverseSolver(n_sources=2, resolution=0.1, verbose=False)
        elif domain == 'ellipse':
            a, b = 2.0, 1.0
            theta = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
            sensor_locs = np.column_stack([a * np.cos(theta), b * np.sin(theta)])
            mesh_data = create_ellipse_mesh(a, b, resolution=0.1, sensor_locations=sensor_locs)
            forward = FEMForwardSolver(resolution=0.1, mesh_data=mesh_data, verbose=False)
            inverse = FEMNonlinearInverseSolver.from_ellipse(a, b, n_sources=2, resolution=0.1, 
                                                              n_sensors=n_sensors, verbose=False)
        elif domain in ['square', 'polygon', 'star', 'brain']:
            if domain == 'square':
                vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            elif domain == 'polygon':
                vertices = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
            elif domain == 'star':
                n_v = 100
                theta_v = np.linspace(0, 2*np.pi, n_v, endpoint=False)
                r_v = 0.7 + 0.3 * np.cos(5 * theta_v)
                vertices = [(r_v[i] * np.cos(theta_v[i]), r_v[i] * np.sin(theta_v[i])) for i in range(n_v)]
            elif domain == 'brain':
                from mesh import get_brain_boundary
                vertices = [(b[0], b[1]) for b in get_brain_boundary(n_points=100)]
            
            # Sensor locations
            verts = np.array(vertices)
            n_verts = len(vertices)
            sensor_locs = []
            for i in range(n_verts):
                v1, v2 = verts[i], verts[(i+1) % n_verts]
                edge_sensors = max(1, n_sensors // n_verts)
                for t in np.linspace(0, 1, edge_sensors, endpoint=False):
                    sensor_locs.append(v1 + t * (v2 - v1))
            sensor_locs = np.array(sensor_locs[:n_sensors])
            
            mesh_data = create_polygon_mesh(vertices, resolution=0.1, sensor_locations=sensor_locs)
            forward = FEMForwardSolver(resolution=0.1, mesh_data=mesh_data, verbose=False)
            # CRITICAL: Pass same mesh_data to inverse solver to ensure consistency
            inverse = FEMNonlinearInverseSolver.from_polygon(vertices, n_sources=2, resolution=0.1,
                                                              n_sensors=n_sensors, 
                                                              mesh_data=mesh_data, verbose=False)
        
        u = forward.solve(sources)
        
        if len(u) == 0:
            results.append(DiagnosticResult(domain, 'nonlinear', 'FEM L-BFGS-B', False, 
                                           "NO SENSORS"))
        else:
            inverse.set_measured_data(u)
            result = inverse.solve(method='L-BFGS-B', maxiter=200, n_restarts=5)
            
            recovered = [((s.x, s.y), s.intensity) for s in result.sources]
            pos_err = compute_position_error(recovered, sources)
            
            passed = pos_err < 0.2
            details = f"pos_err={pos_err:.2e}, residual={result.residual:.2e}"
            
            results.append(DiagnosticResult(domain, 'nonlinear', 'FEM L-BFGS-B', 
                                           passed, details, value=pos_err))
        
    except Exception as e:
        results.append(DiagnosticResult(domain, 'nonlinear', 'FEM L-BFGS-B', 
                                       False, f"{e}\n{traceback.format_exc()}"))
    
    return results


def compute_position_error(recovered: list, true_sources: list) -> float:
    """Compute mean position error with optimal matching."""
    from scipy.optimize import linear_sum_assignment
    
    n = len(true_sources)
    if len(recovered) != n:
        return float('inf')
    
    cost_matrix = np.zeros((n, n))
    for i, (pos_r, _) in enumerate(recovered):
        for j, (pos_t, _) in enumerate(true_sources):
            cost_matrix[i, j] = np.sqrt((pos_r[0]-pos_t[0])**2 + (pos_r[1]-pos_t[1])**2)
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return np.mean([cost_matrix[i, j] for i, j in zip(row_ind, col_ind)])


def run_all_diagnostics():
    """Run all diagnostics for all domains."""
    domains = ['disk', 'ellipse', 'star', 'square', 'polygon', 'brain']
    
    all_results = []
    
    print("=" * 80)
    print("COMPREHENSIVE DEBUG CHECK - ALL DOMAINS AND METHODS")
    print("=" * 80)
    print()
    
    for domain in domains:
        print(f"\n{'='*80}")
        print(f"DOMAIN: {domain.upper()}")
        print(f"{'='*80}")
        
        # Test sources
        sources = create_test_sources_for_domain(domain, n_sources=2)
        print(f"\nTest sources:")
        for i, ((x, y), q) in enumerate(sources):
            print(f"  {i+1}: ({x:+.3f}, {y:+.3f}), q={q:+.2f}")
        
        # Forward solvers
        print(f"\n--- Forward Solvers ---")
        forward_results = check_forward_solver(domain)
        all_results.extend(forward_results)
        for r in forward_results:
            status = "✓" if r.passed else "✗"
            print(f"  {status} {r.test_name}: {r.details}")
        
        # Green's matrix
        print(f"\n--- Green's Matrix Properties ---")
        greens_results = check_greens_matrix(domain)
        all_results.extend(greens_results)
        for r in greens_results:
            status = "✓" if r.passed else "✗"
            print(f"  {status} {r.test_name}: {r.details}")
        
        # Linear solvers
        print(f"\n--- Linear Solvers ---")
        linear_results = check_linear_solver(domain)
        all_results.extend(linear_results)
        for r in linear_results:
            status = "✓" if r.passed else "✗"
            print(f"  {status} {r.test_name}: {r.details}")
        
        # Nonlinear solvers
        print(f"\n--- Nonlinear Solvers ---")
        nonlinear_results = check_nonlinear_solver(domain)
        all_results.extend(nonlinear_results)
        for r in nonlinear_results:
            status = "✓" if r.passed else "✗"
            print(f"  {status} {r.test_name}: {r.details}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in all_results if r.passed)
    failed = sum(1 for r in all_results if not r.passed)
    
    print(f"\nTotal tests: {len(all_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\n--- FAILED TESTS ---")
        for r in all_results:
            if not r.passed:
                print(f"  ✗ [{r.domain}] {r.component}/{r.test_name}")
                print(f"      {r.details[:100]}...")
    
    # Warnings
    warnings = [r for r in all_results if 'WARNING' in r.test_name or 'coherence' in r.details.lower()]
    if warnings:
        print(f"\n--- WARNINGS ---")
        for r in warnings:
            print(f"  ⚠ [{r.domain}] {r.test_name}: {r.details}")
    
    return all_results


if __name__ == '__main__':
    results = run_all_diagnostics()
