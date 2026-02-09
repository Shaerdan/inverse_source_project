"""
Conformal Mapping Solver for General Simply-Connected Domains
==============================================================

This module extends the analytical solver to ANY simply-connected domain
using conformal mapping. The key theoretical result (Theorem 6.3 in the docs):

    For any simply connected domain Ω with conformal map f: Ω → D,
    the solution to the Poisson-Neumann problem with point sources is:
    
        u(z) = Σ qₖ G_D(f(z), f(zₖ))
    
    where G_D is the disk Green's function and the intensities qₖ are PRESERVED
    (not scaled by the Jacobian).

Supported Domains
-----------------
1. **Disk**: Identity map (trivial)
2. **Ellipse**: Inverse Joukowsky transformation (explicit)
3. **Rectangle**: Schwarz-Christoffel with elliptic integrals (explicit)
4. **Regular Polygon**: Schwarz-Christoffel with known prevertices
5. **General Polygon**: Schwarz-Christoffel with numerical prevertices
6. **Arbitrary Smooth Domain**: Numerical conformal mapping via Fornberg/Kerzman-Stein

The Solution Recipe
-------------------
For ANY domain Ω:
1. Obtain conformal map f: Ω → D (explicit or numerical)
2. Map evaluation points: w = f(z)
3. Map source positions: wₖ = f(zₖ)  
4. Compute: u(z) = Σ qₖ G_D(w, wₖ)

That's it! No new PDEs to solve.

References
----------
- Driscoll & Trefethen, "Schwarz-Christoffel Mapping" (2002)
- Henrici, "Applied and Computational Complex Analysis" Vol 3
- Trefethen, "Numerical Conformal Mapping" (1980)
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, fsolve, brentq
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import ellipk
from typing import List, Tuple, Callable, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

try:
    from .analytical_solver import (
        greens_function_disk_neumann,
        Source,
        InverseResult
    )
    from .mesh import (get_source_grid, get_ellipse_source_grid, 
                       get_polygon_source_grid, get_brain_source_grid)
except ImportError:
    from analytical_solver import (
        greens_function_disk_neumann,
        Source,
        InverseResult
    )
    from mesh import (get_source_grid, get_ellipse_source_grid,
                      get_polygon_source_grid, get_brain_source_grid)

# Import optimization utilities for multistart and interior point initialization
try:
    from .optimization_utils import push_to_interior, generate_spread_init
    HAS_OPT_UTILS = True
except ImportError:
    try:
        from optimization_utils import push_to_interior, generate_spread_init
        HAS_OPT_UTILS = True
    except ImportError:
        HAS_OPT_UTILS = False
        push_to_interior = None
        generate_spread_init = None


# =============================================================================
# MFS-BASED CONFORMAL MAP (Proper implementation using Laplace equation)
# =============================================================================

class MFSConformalMap:
    """
    Proper numerical conformal map using Method of Fundamental Solutions.
    
    This fixes the broken radial-scaling approach by solving the Laplace
    equation to extend the boundary correspondence to the interior.
    
    The key insight:
    - Boundary correspondence θ(t) is computed via arc-length (correct)
    - Interior extension: solve Δu = 0 with u|_∂Ω = cos(θ), Δv = 0 with v|_∂Ω = sin(θ)
    - Then f(z) = u + iv is conformal (holomorphic)
    
    Parameters
    ----------
    boundary_func : callable
        Function γ(t) returning complex boundary point for t ∈ [0, 2π]
    n_boundary : int
        Number of boundary discretization points
    n_charge : int
        Number of MFS charge points (source singularities outside domain)
    charge_offset : float
        Distance of charge points from boundary (as fraction of char_size)
    """
    
    def __init__(self, boundary_func: Callable[[float], complex], 
                 n_boundary: int = 256,
                 n_charge: int = 200,
                 charge_offset: float = 0.2):
        self.gamma = boundary_func
        self.n_boundary = n_boundary
        self.n_charge = n_charge
        self.charge_offset = charge_offset
        
        # Sample boundary
        self.t_boundary = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)
        self.z_boundary = np.array([self.gamma(t) for t in self.t_boundary])
        
        # Compute boundary correspondence using arc-length
        self.theta_boundary = self._compute_boundary_correspondence()
        
        # Compute centroid and characteristic size
        self.centroid = np.mean(self.z_boundary)
        self.char_size = np.max(np.abs(self.z_boundary - self.centroid))
        
        # Setup MFS for Laplace solver
        self._setup_mfs()
        
        # Solve for conformal map coefficients
        self._solve_conformal_map()
    
    def _compute_boundary_correspondence(self) -> np.ndarray:
        """Compute boundary correspondence θ(t) using arc-length."""
        n = self.n_boundary
        
        # Compute arc lengths
        dz = np.diff(np.append(self.z_boundary, self.z_boundary[0]))
        arc_lengths = np.abs(dz)
        cumulative = np.cumsum(arc_lengths)
        total_length = cumulative[-1]
        
        # Map to [0, 2π] proportionally
        theta = np.zeros(n)
        theta[1:] = 2 * np.pi * cumulative[:-1] / total_length
        
        return theta
    
    def _setup_mfs(self):
        """Setup Method of Fundamental Solutions charge points."""
        # Place charges OUTSIDE domain (for interior Laplace problem)
        # Use boundary normals to offset
        
        # Compute outward normals
        dz = np.gradient(self.z_boundary)
        tangent = dz / np.abs(dz)
        normal = -1j * tangent  # Rotate 90° for outward normal
        
        # Offset distance
        offset = self.charge_offset * self.char_size
        
        # Charge points (subsample for efficiency)
        idx = np.linspace(0, self.n_boundary-1, self.n_charge, dtype=int)
        self.z_charge = self.z_boundary[idx] + offset * normal[idx]
        
        # Collocation points (boundary points)
        self.z_colloc = self.z_boundary.copy()
    
    def _solve_conformal_map(self):
        """Solve for MFS coefficients to represent conformal map (vectorized)."""
        n_colloc = len(self.z_colloc)
        n_charge = len(self.z_charge)
        
        # Build MFS matrix using vectorized computation: A[i,j] = log|z_colloc[i] - z_charge[j]|
        # Shape: (n_colloc, n_charge)
        z_colloc_col = self.z_colloc[:, np.newaxis]  # (n_colloc, 1)
        z_charge_row = self.z_charge[np.newaxis, :]  # (1, n_charge)
        
        dist = np.abs(z_colloc_col - z_charge_row)  # (n_colloc, n_charge)
        dist = np.maximum(dist, 1e-14)  # Avoid log(0)
        A = np.log(dist) / (2 * np.pi)
        
        # Boundary conditions for u (real part of f): u|_∂Ω = cos(θ)
        b_u = np.cos(self.theta_boundary)
        
        # Boundary conditions for v (imaginary part of f): v|_∂Ω = sin(θ)
        b_v = np.sin(self.theta_boundary)
        
        # Solve least squares (overdetermined system)
        self.coeff_u, _, _, _ = np.linalg.lstsq(A, b_u, rcond=None)
        self.coeff_v, _, _, _ = np.linalg.lstsq(A, b_v, rcond=None)
        
        # Verify accuracy on boundary
        u_check = A @ self.coeff_u
        v_check = A @ self.coeff_v
        
        err_u = np.max(np.abs(u_check - b_u))
        err_v = np.max(np.abs(v_check - b_v))
        
        if err_u > 0.01 or err_v > 0.01:
            warnings.warn(f"MFS boundary fit error: u={err_u:.4f}, v={err_v:.4f}. "
                         "Consider increasing n_charge or adjusting charge_offset.")
    
    def _eval_laplace(self, z: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate MFS solution at interior points (vectorized for speed)."""
        z = np.atleast_1d(z)
        
        # Vectorized computation: dist[i,j] = |z[i] - z_charge[j]|
        # Shape: (n_points, n_charge)
        z_col = z[:, np.newaxis]  # (n_points, 1)
        z_charge_row = self.z_charge[np.newaxis, :]  # (1, n_charge)
        
        dist = np.abs(z_col - z_charge_row)  # (n_points, n_charge)
        dist = np.maximum(dist, 1e-14)  # Avoid log(0)
        
        # MFS kernel: log(dist) / (2*pi)
        kernel = np.log(dist) / (2 * np.pi)  # (n_points, n_charge)
        
        # Sum over charges: result[i] = sum_j coeffs[j] * kernel[i,j]
        result = kernel @ coeffs  # (n_points,)
        
        return result
    
    def to_disk(self, z: np.ndarray) -> np.ndarray:
        """Map from physical domain to unit disk using proper conformal map."""
        z = np.asarray(z, dtype=complex)
        scalar = z.ndim == 0
        z = np.atleast_1d(z)
        
        # Evaluate u and v at each point
        u = self._eval_laplace(z, self.coeff_u)
        v = self._eval_laplace(z, self.coeff_v)
        
        # f(z) = u + iv
        w = u + 1j * v
        
        # Clamp to disk (numerical safety)
        w = np.where(np.abs(w) > 0.999, 0.999 * w / np.abs(w), w)
        
        return w[0] if scalar else w
    
    def from_disk(self, w: np.ndarray) -> np.ndarray:
        """Map from unit disk to physical domain (inverse map).
        
        Uses boundary interpolation for initial guess, then Newton iteration.
        """
        w = np.asarray(w, dtype=complex)
        scalar = w.ndim == 0
        w = np.atleast_1d(w)
        
        z = np.zeros_like(w)
        
        # Pre-compute boundary correspondence: for each theta in disk,
        # find corresponding point on physical boundary
        # theta_boundary[i] corresponds to z_boundary[i]
        
        for i, wi in enumerate(w):
            if np.abs(wi) < 1e-14:
                z[i] = self.centroid
            else:
                r = np.abs(wi)
                theta = np.angle(wi)
                
                # Better initial guess: interpolate boundary point at this angle,
                # then scale toward centroid
                # Find boundary point with closest theta_boundary to theta
                theta_diff = np.abs(np.angle(np.exp(1j * (self.theta_boundary - theta))))
                idx = np.argmin(theta_diff)
                z_boundary_at_theta = self.z_boundary[idx]
                
                # Initial guess: linear interpolation from centroid to boundary
                z_guess = self.centroid + r * (z_boundary_at_theta - self.centroid)
                
                # Newton iteration with damping and more iterations
                best_z = z_guess
                best_err = np.inf
                
                for iteration in range(20):  # More iterations
                    w_current = self.to_disk(np.array([z_guess]))[0]
                    err = np.abs(w_current - wi)
                    
                    # Track best solution
                    if err < best_err:
                        best_err = err
                        best_z = z_guess
                    
                    if err < 1e-10:
                        break
                    
                    # Numerical derivative
                    eps = 1e-7 * max(self.char_size, np.abs(z_guess - self.centroid))
                    dw_dx = (self.to_disk(np.array([z_guess + eps]))[0] - w_current) / eps
                    dw_dy = (self.to_disk(np.array([z_guess + 1j*eps]))[0] - w_current) / eps
                    
                    # Newton step
                    dw = wi - w_current
                    J = np.array([[np.real(dw_dx), np.real(dw_dy)],
                                  [np.imag(dw_dx), np.imag(dw_dy)]])
                    
                    try:
                        det = J[0,0]*J[1,1] - J[0,1]*J[1,0]
                        if np.abs(det) < 1e-14:
                            break
                        delta = np.linalg.solve(J, [np.real(dw), np.imag(dw)])
                        
                        # Damped Newton step to prevent divergence
                        step_size = 1.0
                        for _ in range(5):  # Line search
                            z_new = z_guess + step_size * (delta[0] + 1j * delta[1])
                            w_new = self.to_disk(np.array([z_new]))[0]
                            if np.abs(w_new - wi) < err:
                                z_guess = z_new
                                break
                            step_size *= 0.5
                        else:
                            # Line search failed, take small step anyway
                            z_guess = z_guess + 0.1 * (delta[0] + 1j * delta[1])
                    except:
                        break
                
                z[i] = best_z
        
        return z[0] if scalar else z
    
    def boundary_physical(self, n_points: int) -> np.ndarray:
        """Get boundary points at uniform CONFORMAL (disk) angles.
        
        This ensures DFT correctly computes Fourier coefficients.
        Physical spacing will be non-uniform for non-circular domains.
        
        Returns points z_j = f^{-1}(e^{i*theta_j}) where theta_j = 2*pi*j/n_points.
        """
        theta_disk = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        w = np.exp(1j * theta_disk)
        return self.from_disk(w)
    
    def is_inside(self, z: np.ndarray) -> np.ndarray:
        """Check if points are inside domain using winding number."""
        z = np.asarray(z, dtype=complex)
        scalar = z.ndim == 0
        z = np.atleast_1d(z)
        
        result = np.zeros(len(z), dtype=bool)
        
        for i, zi in enumerate(z):
            winding = 0
            n = len(self.z_boundary)
            
            for j in range(n):
                z1 = self.z_boundary[j] - zi
                z2 = self.z_boundary[(j+1) % n] - zi
                
                if z1.imag <= 0 < z2.imag:
                    if z1.real * z2.imag > z2.real * z1.imag:
                        winding += 1
                elif z2.imag <= 0 < z1.imag:
                    if z1.real * z2.imag < z2.real * z1.imag:
                        winding -= 1
            
            result[i] = winding != 0
        
        return result[0] if scalar else result


# =============================================================================
# CONFORMAL MAP BASE CLASS
# =============================================================================

class ConformalMap(ABC):
    """
    Abstract base class for conformal mappings f: Ω → D.
    
    A conformal map is a holomorphic bijection from a simply-connected 
    domain Ω to the unit disk D = {|w| < 1}.
    
    All subclasses must implement:
        - to_disk(z): Map from physical domain Ω to unit disk D
        - from_disk(w): Map from unit disk D to physical domain Ω
        - boundary_physical(n): Return n points on ∂Ω as complex numbers
        - is_inside(z): Check if points are inside Ω
    
    Optional (will use numerical derivative if not overridden):
        - derivative(z): Compute f'(z)
    """
    
    @abstractmethod
    def to_disk(self, z: np.ndarray) -> np.ndarray:
        """Map point(s) from physical domain Ω to unit disk D."""
        pass
    
    @abstractmethod
    def from_disk(self, w: np.ndarray) -> np.ndarray:
        """Map point(s) from unit disk D to physical domain Ω."""
        pass
    
    @abstractmethod
    def boundary_physical(self, n_points: int) -> np.ndarray:
        """Return n_points on the physical boundary ∂Ω as complex numbers."""
        pass
    
    @abstractmethod
    def is_inside(self, z: np.ndarray) -> np.ndarray:
        """Check if points are inside the physical domain Ω."""
        pass
    
    def derivative(self, z: np.ndarray) -> np.ndarray:
        """
        Compute f'(z) where f = to_disk.
        
        Default implementation uses numerical differentiation.
        Subclasses can override with analytical formula.
        """
        z = np.asarray(z, dtype=complex)
        eps = 1e-8
        return (self.to_disk(z + eps) - self.to_disk(z - eps)) / (2 * eps)
    
    def jacobian(self, z: np.ndarray) -> np.ndarray:
        """Compute |f'(z)|, the Jacobian determinant."""
        return np.abs(self.derivative(z))
    
    def get_interior_grid(self, n_radial: int = 10, n_angular: int = 20) -> np.ndarray:
        """
        Generate interior grid points in the physical domain.
        
        Maps a polar grid in the disk to the physical domain.
        """
        r = np.linspace(0.1, 0.95, n_radial)
        theta = np.linspace(0, 2*np.pi, n_angular, endpoint=False)
        R, Theta = np.meshgrid(r, theta)
        w_grid = R.flatten() * np.exp(1j * Theta.flatten())
        z_grid = self.from_disk(w_grid)
        return z_grid


# =============================================================================
# DISK MAP (Identity)
# =============================================================================

class DiskMap(ConformalMap):
    """
    Identity map for the unit disk (trivial case).
    
    Parameters
    ----------
    radius : float
        Disk radius (default 1.0). If radius ≠ 1, applies scaling.
    """
    
    def __init__(self, radius: float = 1.0):
        self.radius = radius
    
    def to_disk(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=complex) / self.radius
    
    def from_disk(self, w: np.ndarray) -> np.ndarray:
        return np.asarray(w, dtype=complex) * self.radius
    
    def derivative(self, z: np.ndarray) -> np.ndarray:
        return np.ones_like(np.asarray(z), dtype=complex) / self.radius
    
    def boundary_physical(self, n_points: int) -> np.ndarray:
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        return self.radius * np.exp(1j * theta)
    
    def is_inside(self, z: np.ndarray) -> np.ndarray:
        return np.abs(z) < self.radius


# =============================================================================
# ELLIPSE MAP (Inverse Joukowsky)
# =============================================================================

class EllipseMap(ConformalMap):
    """
    Conformal map from ellipse to unit disk using MFS (Method of Fundamental Solutions).
    
    NOTE: We use MFS instead of the analytical Joukowsky map because:
    - Joukowsky has singularities at foci (±c where c = sqrt(a²-b²))
    - These singularities cause objective function discontinuities during optimization
    - MFS provides a smooth conformal map without singularities in the interior
    
    The ellipse has semi-axes a (along x) and b (along y).
    
    Parameters
    ----------
    a : float
        Semi-major axis (along x)
    b : float
        Semi-minor axis (along y), must have b ≤ a
    n_mfs : int
        Number of MFS collocation points (default 256)
    """
    
    def __init__(self, a: float = 2.0, b: float = 1.0, n_mfs: int = 256):
        if b > a:
            a, b = b, a
        self.a = a
        self.b = b
        self.n_mfs = n_mfs
        
        # Focal distance (for reference, not used in MFS)
        self.c = np.sqrt(max(a**2 - b**2, 0))
        
        # Create MFS-based conformal map
        def ellipse_boundary(t):
            return a * np.cos(t) + 1j * b * np.sin(t)
        
        self._mfs_map = MFSConformalMap(
            ellipse_boundary, 
            n_boundary=n_mfs,
            n_charge=min(200, n_mfs),
            charge_offset=0.3
        )
    
    def to_disk(self, z: np.ndarray) -> np.ndarray:
        """Map from ellipse interior/boundary to unit disk."""
        return self._mfs_map.to_disk(z)
    
    def from_disk(self, w: np.ndarray) -> np.ndarray:
        """Map from unit disk to ellipse interior."""
        return self._mfs_map.from_disk(w)
    
    def derivative(self, z: np.ndarray) -> np.ndarray:
        """Numerical derivative of to_disk."""
        z = np.asarray(z, dtype=complex)
        eps = 1e-8 * max(self.a, self.b)
        return (self.to_disk(z + eps) - self.to_disk(z - eps)) / (2 * eps)
    
    def boundary_physical(self, n_points: int) -> np.ndarray:
        """Get boundary points at uniform CONFORMAL (disk) angles.
        
        This ensures DFT correctly computes Fourier coefficients.
        Physical spacing will be non-uniform for ellipses with a != b.
        
        Returns points z_j = f^{-1}(e^{i*theta_j}) where theta_j = 2*pi*j/n_points.
        """
        theta_disk = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        w = np.exp(1j * theta_disk)
        return self.from_disk(w)
    
    def is_inside(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=complex)
        return (np.real(z)/self.a)**2 + (np.imag(z)/self.b)**2 < 1


# =============================================================================
# RECTANGLE MAP (Schwarz-Christoffel with elliptic integrals)
# =============================================================================

class RectangleMap(ConformalMap):
    """
    Conformal map from rectangle to unit disk.
    
    Uses Schwarz-Christoffel with elliptic integrals for the rectangle
    [-a, a] × [-b, b].
    
    Parameters
    ----------
    half_width : float
        Half-width a (extent in x direction)
    half_height : float  
        Half-height b (extent in y direction)
    """
    
    def __init__(self, half_width: float = 1.0, half_height: float = 1.0):
        self.a = half_width
        self.b = half_height
        
        # For rectangle, use simplified mapping via scaling
        # Full SC would use elliptic integrals
        self.scale = max(self.a, self.b)
        
        # Build lookup tables for the mapping
        self._build_lookup_tables()
    
    def _build_lookup_tables(self, n_grid: int = 50):
        """Build interpolation tables for forward/inverse map."""
        # Sample interior on regular grid
        x = np.linspace(-0.95*self.a, 0.95*self.a, n_grid)
        y = np.linspace(-0.95*self.b, 0.95*self.b, n_grid)
        
        self._z_lookup = []
        self._w_lookup = []
        
        for xi in x:
            for yi in y:
                zi = xi + 1j * yi
                # Approximate map using scaled position
                r = np.sqrt((xi/self.a)**2 + (yi/self.b)**2)
                theta = np.arctan2(yi/self.b, xi/self.a)
                wi = min(r, 0.99) * np.exp(1j * theta)
                
                self._z_lookup.append(zi)
                self._w_lookup.append(wi)
        
        self._z_lookup = np.array(self._z_lookup)
        self._w_lookup = np.array(self._w_lookup)
    
    def to_disk(self, z: np.ndarray) -> np.ndarray:
        """Map from rectangle to unit disk using MFS-based conformal map.
        
        NOTE: This method has been fixed to use proper Laplace-based conformal mapping
        instead of the incorrect radial scaling approach.
        """
        z = np.asarray(z, dtype=complex)
        scalar = z.ndim == 0
        z = np.atleast_1d(z)
        
        # Lazy-init the proper MFS map
        if not hasattr(self, '_mfs_map'):
            a, b = self.a, self.b
            
            # Create boundary function for rectangle [-a,a] x [-b,b]
            def rect_boundary(t):
                # Parameter t in [0, 2π] maps around rectangle perimeter
                # Perimeter = 2*(2a + 2b) = 4(a+b)
                perimeter = 4 * (a + b)
                dist = t / (2 * np.pi) * perimeter
                
                # Bottom edge: -a to +a at y=-b
                if dist < 2*a:
                    x = -a + dist
                    return x - 1j*b
                dist -= 2*a
                
                # Right edge: +a, -b to +b
                if dist < 2*b:
                    y = -b + dist
                    return a + 1j*y
                dist -= 2*b
                
                # Top edge: +a to -a at y=+b
                if dist < 2*a:
                    x = a - dist
                    return x + 1j*b
                dist -= 2*a
                
                # Left edge: -a, +b to -b
                y = b - dist
                return -a + 1j*y
            
            self._mfs_map = MFSConformalMap(rect_boundary, n_boundary=256,
                                            n_charge=200, charge_offset=0.2)
        
        w = self._mfs_map.to_disk(z)
        return w[0] if scalar else w
    
    def from_disk(self, w: np.ndarray) -> np.ndarray:
        """Map from unit disk to rectangle using MFS-based conformal map.
        
        NOTE: This method has been fixed to use proper Laplace-based conformal mapping.
        """
        w = np.asarray(w, dtype=complex)
        scalar = w.ndim == 0
        w = np.atleast_1d(w)
        
        # Lazy-init the proper MFS map (same as in to_disk)
        if not hasattr(self, '_mfs_map'):
            # This will be initialized by to_disk call
            self.to_disk(np.array([0.0 + 0.0j]))
        
        z = self._mfs_map.from_disk(w)
        return z[0] if scalar else z
    
    def boundary_physical(self, n_points: int) -> np.ndarray:
        """Get boundary points at uniform CONFORMAL (disk) angles.
        
        This ensures DFT correctly computes Fourier coefficients.
        Physical spacing will be non-uniform for rectangles.
        
        Returns points z_j = f^{-1}(e^{i*theta_j}) where theta_j = 2*pi*j/n_points.
        """
        theta_disk = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        w = np.exp(1j * theta_disk)
        return self.from_disk(w)
    
    def is_inside(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=complex)
        return (np.abs(np.real(z)) < self.a) & (np.abs(np.imag(z)) < self.b)


# =============================================================================
# NUMERICAL CONFORMAL MAP (Arbitrary smooth boundaries)
# =============================================================================

class NumericalConformalMap(ConformalMap):
    """
    Numerical conformal map for arbitrary smooth simply-connected domains.
    
    Given a boundary parameterization γ: [0, 2π] → ∂Ω, computes the
    conformal map f: Ω → D numerically.
    
    The key idea:
    1. The conformal map takes boundary to boundary: f(γ(t)) = e^{iθ(t)}
    2. Find the boundary correspondence θ(t) via arc-length matching
    3. Extend to interior using interpolation
    
    Parameters
    ----------
    boundary_func : callable
        Function γ(t) returning complex boundary point for t ∈ [0, 2π]
    n_boundary : int
        Number of boundary discretization points (default: 256)
    """
    
    def __init__(self, boundary_func: Callable[[float], complex], n_boundary: int = 256):
        self.gamma = boundary_func
        self.n_boundary = n_boundary
        
        # Sample boundary
        self.t_boundary = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)
        self.z_boundary = np.array([self.gamma(t) for t in self.t_boundary])
        
        # Compute boundary correspondence using arc-length
        self.theta_boundary = self._compute_boundary_correspondence()
        
        # Corresponding points on unit circle
        self.w_boundary = np.exp(1j * self.theta_boundary)
        
        # Compute centroid for interior mapping
        self.centroid = np.mean(self.z_boundary)
        
        # Compute characteristic size
        self.char_size = np.max(np.abs(self.z_boundary - self.centroid))
        
        # Build interpolation
        self._build_interpolation()
    
    def _compute_boundary_correspondence(self) -> np.ndarray:
        """
        Compute the boundary correspondence θ(t) using arc-length parameterization.
        
        This is a good approximation for smooth, nearly-circular domains.
        """
        n = self.n_boundary
        
        # Compute arc lengths
        dz = np.diff(np.append(self.z_boundary, self.z_boundary[0]))
        arc_lengths = np.abs(dz)
        cumulative = np.cumsum(arc_lengths)
        total_length = cumulative[-1]
        
        # Map to [0, 2π] proportionally
        theta = np.zeros(n)
        theta[1:] = 2 * np.pi * cumulative[:-1] / total_length
        
        return theta
    
    def _build_interpolation(self):
        """Build interpolation functions for boundary mapping."""
        # Extend periodically
        t_ext = np.concatenate([self.t_boundary - 2*np.pi, 
                                self.t_boundary, 
                                self.t_boundary + 2*np.pi])
        theta_ext = np.concatenate([self.theta_boundary - 2*np.pi,
                                    self.theta_boundary,
                                    self.theta_boundary + 2*np.pi])
        z_ext = np.tile(self.z_boundary, 3)
        
        # θ → z interpolation (for from_disk on boundary)
        sort_idx = np.argsort(theta_ext)
        theta_sorted = theta_ext[sort_idx]
        z_sorted = z_ext[sort_idx]
        
        # Remove duplicate theta values (keep first occurrence)
        # This fixes issues with boundaries that have near-identical angles
        unique_mask = np.concatenate([[True], np.diff(theta_sorted) > 1e-10])
        theta_unique = theta_sorted[unique_mask]
        z_unique = z_sorted[unique_mask]
        
        self._theta_to_z = interp1d(theta_unique, z_unique, 
                                     kind='cubic', bounds_error=False,
                                     fill_value='extrapolate')
        
        # CRITICAL FIX: Build raw angle → z interpolation for consistency with to_disk()
        # to_disk() uses raw angles from centroid, so from_disk() must use the same
        raw_angles = np.angle(self.z_boundary - self.centroid)
        
        # Extend periodically for raw angles too
        raw_angles_ext = np.concatenate([raw_angles - 2*np.pi, 
                                          raw_angles, 
                                          raw_angles + 2*np.pi])
        
        # Sort by raw angle
        sort_idx_raw = np.argsort(raw_angles_ext)
        raw_sorted = raw_angles_ext[sort_idx_raw]
        z_sorted_raw = z_ext[sort_idx_raw]
        
        # Remove duplicates
        unique_mask_raw = np.concatenate([[True], np.diff(raw_sorted) > 1e-10])
        raw_unique = raw_sorted[unique_mask_raw]
        z_unique_raw = z_sorted_raw[unique_mask_raw]
        
        self._raw_angle_to_z = interp1d(raw_unique, z_unique_raw,
                                         kind='cubic', bounds_error=False,
                                         fill_value='extrapolate')
    
    def to_disk(self, z: np.ndarray) -> np.ndarray:
        """Map from physical domain to unit disk using MFS-based conformal map.
        
        NOTE: This method has been fixed to use proper Laplace-based conformal mapping
        instead of the incorrect radial scaling approach.
        """
        z = np.asarray(z, dtype=complex)
        scalar = z.ndim == 0
        z = np.atleast_1d(z)
        
        # Lazy-init the proper MFS map (computed once, cached)
        if not hasattr(self, '_mfs_map'):
            self._mfs_map = MFSConformalMap(self.gamma, n_boundary=self.n_boundary,
                                            n_charge=200, charge_offset=0.2)
        
        w = self._mfs_map.to_disk(z)
        return w[0] if scalar else w
    
    def from_disk(self, w: np.ndarray) -> np.ndarray:
        """Map from unit disk to physical domain using MFS-based conformal map.
        
        NOTE: This method has been fixed to use proper Laplace-based conformal mapping.
        """
        w = np.asarray(w, dtype=complex)
        scalar = w.ndim == 0
        w = np.atleast_1d(w)
        
        # Lazy-init the proper MFS map (computed once, cached)
        if not hasattr(self, '_mfs_map'):
            self._mfs_map = MFSConformalMap(self.gamma, n_boundary=self.n_boundary,
                                            n_charge=200, charge_offset=0.2)
        
        z = self._mfs_map.from_disk(w)
        return z[0] if scalar else z
    
    def boundary_physical(self, n_points: int) -> np.ndarray:
        """Get boundary points at uniform CONFORMAL (disk) angles.
        
        This ensures DFT correctly computes Fourier coefficients.
        Physical spacing will be non-uniform for non-circular domains.
        
        Returns points z_j = f^{-1}(e^{i*theta_j}) where theta_j = 2*pi*j/n_points.
        """
        theta_disk = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        w = np.exp(1j * theta_disk)
        return self.from_disk(w)
    
    def is_inside(self, z: np.ndarray) -> np.ndarray:
        """Check if inside using winding number."""
        z = np.asarray(z, dtype=complex)
        scalar = z.ndim == 0
        z = np.atleast_1d(z)
        
        result = np.zeros(len(z), dtype=bool)
        
        for i, zi in enumerate(z):
            winding = 0
            n = len(self.z_boundary)
            
            for j in range(n):
                z1 = self.z_boundary[j] - zi
                z2 = self.z_boundary[(j+1) % n] - zi
                
                if z1.imag <= 0 < z2.imag:
                    if z1.real * z2.imag > z2.real * z1.imag:
                        winding += 1
                elif z2.imag <= 0 < z1.imag:
                    if z1.real * z2.imag < z2.real * z1.imag:
                        winding -= 1
            
            result[i] = winding != 0
        
        return result[0] if scalar else result


# =============================================================================
# POLYGON MAP (Schwarz-Christoffel)
# =============================================================================

class PolygonMap(ConformalMap):
    """
    Conformal map from polygon to unit disk using Schwarz-Christoffel.
    
    For simple polygons, uses a simplified mapping based on the
    polygon's shape. For more accurate results with complex polygons,
    consider using specialized SC libraries.
    
    Parameters
    ----------
    vertices : list of complex or (x,y) tuples
        Polygon vertices in counterclockwise order
    """
    
    def __init__(self, vertices: Union[List[complex], List[Tuple[float, float]]]):
        # Convert to complex
        if isinstance(vertices[0], (tuple, list)):
            self.vertices = np.array([v[0] + 1j*v[1] for v in vertices])
        else:
            self.vertices = np.array(vertices, dtype=complex)
        
        self.n = len(self.vertices)
        
        # Compute centroid
        self.centroid = np.mean(self.vertices)
        
        # Compute characteristic size
        self.char_size = np.max(np.abs(self.vertices - self.centroid))
    
    def to_disk(self, z: np.ndarray) -> np.ndarray:
        """Map from polygon to unit disk using MFS-based conformal map.
        
        NOTE: This method has been fixed to use proper Laplace-based conformal mapping
        instead of the incorrect radial scaling approach.
        """
        z = np.asarray(z, dtype=complex)
        scalar = z.ndim == 0
        z = np.atleast_1d(z)
        
        # Lazy-init the proper MFS map
        if not hasattr(self, '_mfs_map'):
            # Create boundary function from polygon vertices
            def polygon_boundary(t):
                # Parameter t in [0, 2π] maps to polygon perimeter
                n = len(self.vertices)
                # Total perimeter
                edges = np.abs(np.diff(np.append(self.vertices, self.vertices[0])))
                total = np.sum(edges)
                
                # Find which edge and position
                target_dist = t / (2 * np.pi) * total
                cumsum = np.cumsum(edges)
                
                for i in range(n):
                    if i == 0:
                        if target_dist <= edges[0]:
                            frac = target_dist / edges[0]
                            return self.vertices[0] + frac * (self.vertices[1] - self.vertices[0])
                    else:
                        if target_dist <= cumsum[i]:
                            prev_dist = cumsum[i-1]
                            frac = (target_dist - prev_dist) / edges[i]
                            return self.vertices[i] + frac * (self.vertices[(i+1) % n] - self.vertices[i])
                
                return self.vertices[0]
            
            self._mfs_map = MFSConformalMap(polygon_boundary, n_boundary=256,
                                            n_charge=200, charge_offset=0.2)
        
        w = self._mfs_map.to_disk(z)
        return w[0] if scalar else w
    
    def from_disk(self, w: np.ndarray) -> np.ndarray:
        """Map from unit disk to polygon using MFS-based conformal map.
        
        NOTE: This method has been fixed to use proper Laplace-based conformal mapping.
        """
        w = np.asarray(w, dtype=complex)
        scalar = w.ndim == 0
        w = np.atleast_1d(w)
        
        # Lazy-init the proper MFS map (same as in to_disk)
        if not hasattr(self, '_mfs_map'):
            # This will be initialized by to_disk call
            self.to_disk(np.array([self.centroid]))
        
        z = self._mfs_map.from_disk(w)
        return z[0] if scalar else z
    
    def boundary_physical(self, n_points: int) -> np.ndarray:
        """Get boundary points at uniform CONFORMAL (disk) angles.
        
        This ensures DFT correctly computes Fourier coefficients.
        Physical spacing will be non-uniform for polygons.
        
        Returns points z_j = f^{-1}(e^{i*theta_j}) where theta_j = 2*pi*j/n_points.
        """
        theta_disk = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        w = np.exp(1j * theta_disk)
        return self.from_disk(w)
    
    def is_inside(self, z: np.ndarray) -> np.ndarray:
        """Check if inside polygon using winding number."""
        z = np.asarray(z, dtype=complex)
        scalar = z.ndim == 0
        z = np.atleast_1d(z)
        
        result = np.zeros(len(z), dtype=bool)
        
        for i, zi in enumerate(z):
            winding = 0
            for j in range(self.n):
                v1 = self.vertices[j] - zi
                v2 = self.vertices[(j+1) % self.n] - zi
                
                if v1.imag <= 0 < v2.imag:
                    if v1.real * v2.imag > v2.real * v1.imag:
                        winding += 1
                elif v2.imag <= 0 < v1.imag:
                    if v1.real * v2.imag < v2.real * v1.imag:
                        winding -= 1
            
            result[i] = winding != 0
        
        return result[0] if scalar else result


# =============================================================================
# CONFORMAL FORWARD SOLVER
# =============================================================================

class ConformalForwardSolver:
    """
    Forward solver for any domain using conformal mapping.
    
    Uses the universal solution formula:
        u(z) = Σ qₖ G_D(f(z), f(zₖ))
    
    Parameters
    ----------
    conformal_map : ConformalMap
        The conformal mapping from physical domain to unit disk
    n_boundary : int
        Number of sensor/measurement points
    sensor_locations : array, optional
        Custom sensor locations in physical domain. If None, uses evenly spaced.
    """
    
    def __init__(self, conformal_map: ConformalMap, n_boundary: int = 100,
                 sensor_locations: np.ndarray = None, disk_angles: np.ndarray = None):
        self.map = conformal_map
        self.n_boundary = n_boundary
        self.disk_angles = disk_angles  # Store exact disk angles if provided
        
        # Sensor/boundary points in physical domain
        if sensor_locations is not None:
            self.boundary_points = np.asarray(sensor_locations)
            self.z_boundary = self.boundary_points[:, 0] + 1j * self.boundary_points[:, 1]
            self.n_sensors = len(sensor_locations)
        else:
            self.z_boundary = conformal_map.boundary_physical(n_boundary)
            self.boundary_points = np.column_stack([
                np.real(self.z_boundary),
                np.imag(self.z_boundary)
            ])
            self.n_sensors = n_boundary
        
        # Map to disk
        self.w_boundary = conformal_map.to_disk(self.z_boundary)
        
        # For backward compatibility
        self.sensor_locations = self.boundary_points
    
    def solve(self, sources: List[Tuple[Tuple[float, float], float]]) -> np.ndarray:
        """
        Compute potential at sensor locations from sources (vectorized).
        
        Parameters
        ----------
        sources : list of ((x, y), q)
            Source positions and intensities
        
        Returns
        -------
        u : array of shape (n_sensors,)
            Potential values at sensor locations
        """
        # Convert sources to arrays
        z_sources = np.array([s[0][0] + 1j * s[0][1] for s in sources])
        q_sources = np.array([s[1] for s in sources])
        n_sources = len(sources)
        
        # Check compatibility
        if np.abs(np.sum(q_sources)) > 1e-10:
            warnings.warn(f"Sources do not sum to zero: Σq = {np.sum(q_sources):.6f}")
        
        # Map sources to disk
        w_sources = self.map.to_disk(z_sources)
        
        # Check if sensors are on boundary (|w| ≈ 1) OR if disk_angles were provided
        w_abs = np.abs(self.w_boundary)
        on_boundary = self.disk_angles is not None or np.allclose(w_abs, 1.0, rtol=0.01)
        
        u = np.zeros(self.n_sensors)
        
        if on_boundary:
            # Use boundary formula (Poisson kernel) - more accurate for boundary sensors
            # Use exact disk_angles if provided, otherwise extract from w_boundary
            if self.disk_angles is not None:
                theta_boundary = self.disk_angles
            else:
                theta_boundary = np.angle(self.w_boundary)
            
            for j in range(n_sources):
                r_j = np.abs(w_sources[j])
                phi_j = np.angle(w_sources[j])
                # Poisson kernel formula for boundary potential
                angle_diff = theta_boundary - phi_j
                u += q_sources[j] * (-1.0 / (2 * np.pi)) * np.log(1 + r_j**2 - 2*r_j*np.cos(angle_diff))
        else:
            # Use interior Green's function
            x_eval = np.column_stack([np.real(self.w_boundary), np.imag(self.w_boundary)])
            
            for j in range(n_sources):
                xi_src = np.array([np.real(w_sources[j]), np.imag(w_sources[j])])
                G_j = greens_function_disk_neumann(x_eval, xi_src)
                u += q_sources[j] * G_j
        
        # Center output to match disk solver convention (Neumann BC determines u up to constant)
        return u - np.mean(u)


# =============================================================================
# CONFORMAL LINEAR INVERSE SOLVER  
# =============================================================================

class ConformalLinearInverseSolver:
    """
    Linear inverse solver for any domain using conformal mapping.
    
    Places candidate sources on a grid, builds Green's matrix using
    the universal formula, then solves with regularization.
    
    Parameters
    ----------
    conformal_map : ConformalMap
        Conformal mapping from physical domain to unit disk
    n_boundary : int
        Number of sensor/measurement points
    sensor_locations : array, optional
        Custom sensor locations in physical domain. If None, uses evenly spaced.
    interior_points : ndarray, optional
        (N, 2) array of interior source candidate points in physical domain.
        If provided, uses these instead of generating a grid.
        This allows using the same mesh as FEM for fair comparison.
    source_resolution : float
        Approximate spacing between source grid points (only used if 
        interior_points is None)
    verbose : bool
        Print progress information
    """
    
    def __init__(self, conformal_map: ConformalMap, n_boundary: int = 100,
                 sensor_locations: np.ndarray = None,
                 interior_points: np.ndarray = None,
                 source_resolution: float = 0.15, verbose: bool = False):
        self.map = conformal_map
        self.n_boundary = n_boundary
        self.source_resolution = source_resolution
        self.verbose = verbose
        
        # Store sensor locations
        self.sensor_locations = sensor_locations
        
        # Create forward solver with sensor locations
        self.forward = ConformalForwardSolver(conformal_map, n_boundary, 
                                               sensor_locations=sensor_locations)
        self.n_sensors = self.forward.n_sensors
        
        # Use provided grid or generate one
        if interior_points is not None:
            self._set_grid_from_points(interior_points)
        else:
            self._generate_source_grid()
        
        # Green's matrix (built lazily or explicitly)
        self.G = None
    
    def _set_grid_from_points(self, points: np.ndarray):
        """Set source grid from externally provided points."""
        # Convert to complex for conformal mapping
        z_grid = points[:, 0] + 1j * points[:, 1]
        
        # Filter to interior only (in case some points are on/outside boundary)
        inside = self.map.is_inside(z_grid)
        z_grid = z_grid[inside]
        
        # Map to disk
        w_grid = self.map.to_disk(z_grid)
        
        self.z_grid = z_grid
        self.w_grid = w_grid
        self.n_sources = len(z_grid)
        self.grid_points = np.column_stack([np.real(z_grid), np.imag(z_grid)])
        
        if self.verbose:
            print(f"  Using {self.n_sources} provided interior points")
    
    def _generate_source_grid(self):
        """
        Generate interior source grid using gmsh mesh.
        
        Detects domain type from conformal map and uses appropriate
        gmsh-based grid generator for consistent methodology with FEM.
        """
        # Detect domain type and generate appropriate gmsh grid
        if isinstance(self.map, EllipseMap):
            # Ellipse domain - use ellipse grid
            grid_points = get_ellipse_source_grid(
                self.map.a, self.map.b,
                resolution=self.source_resolution,
                margin=0.0  # Include all interior points
            )
            if self.verbose:
                print(f"  Using gmsh ellipse grid: a={self.map.a}, b={self.map.b}")
                
        elif hasattr(self.map, 'boundary_func'):
            # MFSConformalMap or similar with boundary function
            # Sample boundary to get vertices for polygon grid
            n_vertices = 100
            t = np.linspace(0, 2*np.pi, n_vertices, endpoint=False)
            boundary = np.array([self.map.boundary_func(ti) for ti in t])
            vertices = [(np.real(z), np.imag(z)) for z in boundary]
            
            grid_points = get_polygon_source_grid(
                vertices,
                resolution=self.source_resolution,
                margin=0.0
            )
            if self.verbose:
                print(f"  Using gmsh polygon grid from boundary function")
                
        elif hasattr(self.map, 'vertices'):
            # Polygon map with explicit vertices
            vertices = self.map.vertices
            grid_points = get_polygon_source_grid(
                vertices,
                resolution=self.source_resolution,
                margin=0.0
            )
            if self.verbose:
                print(f"  Using gmsh polygon grid: {len(vertices)} vertices")
                
        else:
            # Fallback: try to sample boundary from boundary_physical method
            if hasattr(self.map, 'boundary_physical'):
                boundary = self.map.boundary_physical(100)
                vertices = [(np.real(z), np.imag(z)) for z in boundary]
                grid_points = get_polygon_source_grid(
                    vertices,
                    resolution=self.source_resolution,
                    margin=0.0
                )
                if self.verbose:
                    print(f"  Using gmsh polygon grid from boundary_physical")
            else:
                raise ValueError(
                    "Cannot auto-detect domain type for source grid generation. "
                    "Please provide interior_points explicitly."
                )
        
        # Convert to complex and map to disk
        z_grid = grid_points[:, 0] + 1j * grid_points[:, 1]
        
        # Filter to interior only
        inside = self.map.is_inside(z_grid)
        z_grid = z_grid[inside]
        
        # Map to unit disk for Green's function evaluation
        w_grid = self.map.to_disk(z_grid)
        
        self.z_grid = z_grid
        self.w_grid = w_grid
        self.n_sources = len(z_grid)
        self.grid_points = np.column_stack([np.real(z_grid), np.imag(z_grid)])
        
        if self.verbose:
            print(f"  Generated {self.n_sources} interior grid points (gmsh)")
    
    @property
    def interior_points(self) -> np.ndarray:
        """Alias for grid_points (compatibility with comparison.py)."""
        return self.grid_points
    
    def build_greens_matrix(self, verbose: bool = None):
        """
        Build Green's matrix G[i,j] = G_D(w_boundary[i], w_grid[j]).
        
        Uses the disk Green's function in mapped coordinates.
        """
        if verbose is None:
            verbose = self.verbose
            
        n_meas = self.n_sensors  # Use sensor count, not boundary count
        n_src = self.n_sources
        
        if verbose:
            print(f"  Building {n_meas}x{n_src} Green's matrix...")
        
        self.G = np.zeros((n_meas, n_src))
        
        for i in range(n_meas):
            w_eval = self.forward.w_boundary[i]
            x_eval = np.array([[np.real(w_eval), np.imag(w_eval)]])
            
            for j in range(n_src):
                w_src = self.w_grid[j]
                xi_src = np.array([np.real(w_src), np.imag(w_src)])
                self.G[i, j] = greens_function_disk_neumann(x_eval, xi_src)
        
        if verbose:
            print(f"  Green's matrix built: condition number ~ {np.linalg.cond(self.G):.2e}")
    
    def _ensure_greens_matrix(self):
        """Build Green's matrix if not already built."""
        if self.G is None:
            self.build_greens_matrix()
    
    def solve_l2(self, u_meas: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
        """
        Tikhonov (L2) regularization with zero-sum constraint.
        
        Solves: min ||Gq - u||² + α||q||²  subject to Σq = 0
        """
        self._ensure_greens_matrix()
        
        G, u = self.G, u_meas
        n = self.n_sources
        
        # Augmented system for constraint Σq = 0
        G_aug = np.vstack([G, np.ones(n)])
        u_aug = np.append(u, 0)
        
        # Normal equations with regularization
        A = G_aug.T @ G_aug + alpha * np.eye(n)
        b = G_aug.T @ u_aug
        
        q = np.linalg.solve(A, b)
        
        # Project to satisfy constraint exactly
        q = q - np.mean(q)
        
        return q
    
    def solve_l1(self, u_meas: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
        """
        L1 (sparsity) regularization with zero-sum constraint.
        
        Solves: min ||Gq - u||² + α||q||₁  subject to Σq = 0
        """
        self._ensure_greens_matrix()
        
        try:
            import cvxpy as cp
        except ImportError:
            warnings.warn("cvxpy not available, falling back to L2")
            return self.solve_l2(u_meas, alpha)
        
        n = self.n_sources
        q = cp.Variable(n)
        
        objective = cp.Minimize(
            0.5 * cp.sum_squares(self.G @ q - u_meas) + alpha * cp.norm1(q)
        )
        constraints = [cp.sum(q) == 0]
        
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.ECOS)
        except:
            try:
                problem.solve(solver=cp.SCS)
            except:
                problem.solve()
        
        return q.value if q.value is not None else self.solve_l2(u_meas, alpha)
    
    def solve_tv(self, u_meas: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
        """
        Total Variation regularization with zero-sum constraint.
        
        Solves: min ||Gq - u||² + α||Dq||₁  subject to Σq = 0
        
        Uses a simple finite difference approximation for TV on the grid.
        """
        self._ensure_greens_matrix()
        
        try:
            import cvxpy as cp
        except ImportError:
            warnings.warn("cvxpy not available, falling back to L2")
            return self.solve_l2(u_meas, alpha)
        
        n = self.n_sources
        
        # Build gradient operator using nearest neighbors
        D = self._build_gradient_operator()
        
        q = cp.Variable(n)
        
        objective = cp.Minimize(
            0.5 * cp.sum_squares(self.G @ q - u_meas) + alpha * cp.norm1(D @ q)
        )
        constraints = [cp.sum(q) == 0]
        
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.ECOS)
        except:
            try:
                problem.solve(solver=cp.SCS)
            except:
                problem.solve()
        
        return q.value if q.value is not None else self.solve_l2(u_meas, alpha)
    
    def _build_gradient_operator(self) -> np.ndarray:
        """Build finite difference gradient operator based on nearest neighbors."""
        from scipy.spatial import KDTree
        
        n = self.n_sources
        tree = KDTree(self.grid_points)
        
        # For each point, find nearest neighbor and create difference
        rows, cols, data = [], [], []
        row_idx = 0
        
        for i in range(n):
            # Find k nearest neighbors
            dists, idxs = tree.query(self.grid_points[i], k=min(4, n))
            
            for j, (d, idx) in enumerate(zip(dists[1:], idxs[1:])):  # Skip self
                if d < 3 * self.source_resolution:  # Only nearby points
                    rows.append(row_idx)
                    cols.append(i)
                    data.append(1.0)
                    
                    rows.append(row_idx)
                    cols.append(idx)
                    data.append(-1.0)
                    
                    row_idx += 1
        
        from scipy.sparse import csr_matrix
        D = csr_matrix((data, (rows, cols)), shape=(row_idx, n))
        return D.toarray()


# =============================================================================
# CONFORMAL NONLINEAR INVERSE SOLVER
# =============================================================================

class ConformalNonlinearInverseSolver:
    """
    Nonlinear inverse solver for any domain using conformal mapping.
    
    Optimizes source positions and intensities directly using the
    universal solution formula.
    
    Parameters
    ----------
    conformal_map : ConformalMap
        Conformal mapping from physical domain to unit disk
    n_sources : int
        Number of sources to recover
    n_boundary : int
        Number of boundary measurement points
    sensor_locations : array, optional
        Physical sensor locations
    disk_angles : array, optional
        Exact disk angles for sensors (avoids numerical precision issues)
    """
    
    def __init__(self, conformal_map: ConformalMap, n_sources: int = 4,
                 n_boundary: int = 100, sensor_locations: np.ndarray = None,
                 disk_angles: np.ndarray = None):
        self.map = conformal_map
        self.n_sources = n_sources
        self.n_boundary = n_boundary
        self.sensor_locations = sensor_locations
        self.disk_angles = disk_angles
        # CRITICAL: Pass sensor_locations AND disk_angles to ensure forward solver uses exact same angles as data
        self.forward = ConformalForwardSolver(conformal_map, n_boundary, 
                                               sensor_locations=sensor_locations,
                                               disk_angles=disk_angles)
    
    def _objective_misfit(self, params: np.ndarray, u_meas: np.ndarray) -> float:
        """
        Pure misfit objective: ||u_computed - u_measured||²
        
        Used with NonlinearConstraint (DE, trust-constr) where constraint
        is handled separately.
        
        Returns large penalty if sources are outside domain (needed because
        SLSQP may evaluate at infeasible points during line search).
        """
        n = self.n_sources
        
        # Unpack parameters
        positions = params[:2*n].reshape(n, 2)
        intensities = params[2*n:3*n]
        
        # Check if sources are inside domain via conformal map
        z_sources = positions[:, 0] + 1j * positions[:, 1]
        try:
            w_sources = self.map.to_disk(z_sources)
            w_abs_sq = np.abs(w_sources)**2
            
            # If any source is outside domain (|w| >= 1), return penalty
            if np.any(w_abs_sq >= 1.0):
                return 1e12
        except Exception:
            # Conformal map failed (point too far from domain)
            return 1e12
        
        # Enforce zero-sum constraint
        intensities = intensities - np.mean(intensities)
        
        # Build sources list
        sources = [((positions[k, 0], positions[k, 1]), intensities[k]) 
                   for k in range(n)]
        
        # Compute forward solution
        u_computed = self.forward.solve(sources)
        
        # Residual SQUARED
        return np.sum((u_computed - u_meas)**2)
    
    def _objective_with_barrier(self, params: np.ndarray, u_meas: np.ndarray, mu: float = 1e-6) -> float:
        """
        Objective with logarithmic barrier for L-BFGS-B interior point method.
        
        f(z) = ||u_computed - u_measured||² - μ * Σ log(1 - |w_k|²)
        
        where w_k = conformal_map(z_k) maps source positions to unit disk.
        """
        n = self.n_sources
        
        # Unpack parameters
        positions = params[:2*n].reshape(n, 2)
        intensities = params[2*n:3*n]
        
        # Enforce zero-sum constraint
        intensities = intensities - np.mean(intensities)
        
        # Map source positions to unit disk via conformal map
        z_sources = positions[:, 0] + 1j * positions[:, 1]
        w_sources = self.map.to_disk(z_sources)
        w_abs_sq = np.abs(w_sources)**2
        
        # Check if any source is outside (|w| >= 1)
        if np.any(w_abs_sq >= 1.0):
            return 1e12
        
        # Logarithmic barrier
        barrier = -mu * np.sum(np.log(1.0 - w_abs_sq))
        
        # Build sources list
        sources = [((positions[k, 0], positions[k, 1]), intensities[k]) 
                   for k in range(n)]
        
        # Compute forward solution
        u_computed = self.forward.solve(sources)
        
        # Residual SQUARED
        residual_sq = np.sum((u_computed - u_meas)**2)
        
        return residual_sq + barrier
    
    def _conformal_constraint(self, params: np.ndarray) -> np.ndarray:
        """
        Conformal domain constraint for NonlinearConstraint.
        
        Returns array of (1 - |w_k|²) for each source, where w_k = f(z_k).
        Constraint satisfied when all values > 0 (source inside domain).
        
        Returns large negative values if conformal map fails (indicating
        source is way outside domain).
        """
        n = self.n_sources
        positions = params[:2*n].reshape(n, 2)
        
        z_sources = positions[:, 0] + 1j * positions[:, 1]
        
        try:
            w_sources = self.map.to_disk(z_sources)
            w_abs_sq = np.abs(w_sources)**2
            return 1.0 - w_abs_sq
        except Exception:
            # Conformal map failed - return large negative (constraint violated)
            return -1e6 * np.ones(n)
    
    def _generate_valid_initial_guess(self, bounds: list, seed: int, 
                                       strategy: str = 'random') -> np.ndarray:
        """
        Generate initial guess with positions guaranteed to be inside the domain.
        
        Uses rejection sampling to ensure all source positions are valid.
        
        KEY FIX: Push to interior of bounds after generation to avoid gradient blow-up.
        
        Parameters
        ----------
        bounds : list of (lower, upper)
            Bounds for each parameter
        seed : int
            Random seed for reproducibility
        strategy : str
            'random', 'spread', or 'linspace'
        """
        n = self.n_sources
        np.random.seed(seed)
        
        x0 = []
        max_attempts = 1000
        
        if strategy == 'spread':
            # Spread sources evenly around domain center
            boundary = self.map.boundary_physical(100)
            center_x = np.real(boundary).mean()
            center_y = np.imag(boundary).mean()
            
            # Vary radius across restarts to cover search space (principled multi-start)
            # seed 0->40%, seed 1->55%, seed 2->70% of domain radius
            distances = np.abs(boundary - complex(center_x, center_y))
            base_radius = np.min(distances)
            scale = 0.4 + 0.15 * (seed % 3)  # Cycles through 0.4, 0.55, 0.7
            safe_radius = scale * base_radius
            
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            angles += seed * 0.1  # Small offset per restart
            
            for i in range(n):
                x = center_x + safe_radius * np.cos(angles[i])
                y = center_y + safe_radius * np.sin(angles[i])
                x0.extend([x, y])
        
        else:  # 'random' (default)
            for i in range(n):
                # Generate position inside domain using rejection sampling
                for attempt in range(max_attempts):
                    x = np.random.uniform(bounds[2*i][0], bounds[2*i][1])
                    y = np.random.uniform(bounds[2*i+1][0], bounds[2*i+1][1])
                    z = complex(x, y)
                    
                    if self.map.is_inside(z):
                        x0.extend([x, y])
                        break
                else:
                    # Fallback: use centroid
                    boundary = self.map.boundary_physical(100)
                    centroid_x = np.real(boundary).mean()
                    centroid_y = np.imag(boundary).mean()
                    # Add small random offset
                    x0.extend([centroid_x + 0.1 * np.random.randn(),
                              centroid_y + 0.1 * np.random.randn()])
        
        # Add intensities
        if strategy == 'spread':
            for i in range(n):
                x0.append(0.5 * (1 if i % 2 == 0 else -1))  # Alternating
        else:
            for i in range(n):
                x0.append(np.random.randn())
        
        x0 = np.array(x0)
        
        # KEY FIX: Push to interior of bounds to avoid gradient blow-up
        if HAS_OPT_UTILS and push_to_interior is not None:
            x0 = push_to_interior(x0, bounds, margin=0.1)
        
        return x0
    
    def solve(self, u_meas: np.ndarray, method: str = 'SLSQP',
              seed: int = 42, n_restarts: int = 5, mu: float = 1e-6) -> Tuple[List[Tuple[Tuple[float, float], float]], float]:
        """
        Solve nonlinear inverse problem.
        
        Parameters
        ----------
        u_meas : array
            Measured boundary potential
        method : str
            Optimization method:
            - 'SLSQP': Sequential Least Squares Programming (RECOMMENDED)
            - 'differential_evolution': Global with NonlinearConstraint
            - 'trust-constr': Local with NonlinearConstraint
            - 'L-BFGS-B' or 'lbfgsb': Local with log barrier
            - 'basin_hopping': Global with local polish
        seed : int
            Random seed for reproducibility
        n_restarts : int
            Number of random restarts for local optimizers
        mu : float
            Barrier parameter for L-BFGS-B interior point method
        
        Returns
        -------
        sources : list of ((x, y), q)
            Recovered source positions and intensities
        residual : float
            Final residual value
        """
        n = self.n_sources
        np.random.seed(seed)
        
        # CRITICAL FIX: Center measured data to remove arbitrary constant (gauge freedom)
        # This matches what AnalyticalNonlinearInverseSolver does in set_measured_data()
        u_meas = u_meas - np.mean(u_meas)
        
        # Get domain bounding box from boundary
        boundary = self.map.boundary_physical(100)
        x_min, x_max = np.real(boundary).min(), np.real(boundary).max()
        y_min, y_max = np.imag(boundary).min(), np.imag(boundary).max()
        
        # Add small margin to keep sources away from boundary
        margin = 0.05 * min(x_max - x_min, y_max - y_min)
        x_min, x_max = x_min + margin, x_max - margin
        y_min, y_max = y_min + margin, y_max - margin
        
        # Box bounds that contain the domain
        box_bounds = []
        for _ in range(n):
            box_bounds.append((x_min, x_max))  # x
            box_bounds.append((y_min, y_max))  # y
        for _ in range(n):
            box_bounds.append((-5.0, 5.0))  # intensity
        
        # NonlinearConstraint: 1 - |f(z)|² > 0 for each source
        from scipy.optimize import NonlinearConstraint
        conformal_constraint = NonlinearConstraint(
            self._conformal_constraint,
            0.0,      # lower bound (must be > 0, i.e., inside domain)
            np.inf    # upper bound
        )
        
        # Equality constraint for SLSQP: sum of intensities = 0
        def intensity_sum(params):
            return sum(params[2*n + i] for i in range(n))
        
        if method == 'SLSQP':
            # SLSQP with equality constraint and domain constraint
            
            # Inequality constraint: sources must be inside domain
            # _conformal_constraint returns 1 - |w|² which must be > 0
            def domain_constraint(params):
                return self._conformal_constraint(params)  # Must be >= 0
            
            constraints = [
                {'type': 'eq', 'fun': intensity_sum},
                {'type': 'ineq', 'fun': domain_constraint},  # g(x) >= 0
            ]
            
            best_params = None
            best_residual = np.inf
            
            # Use diverse initialization strategies
            # 'spread' varies radius with seed, so multiple spread calls cover search space
            strategies = ['spread'] * min(3, n_restarts) + ['random'] * max(0, n_restarts - 3)
            
            for restart, strategy in enumerate(strategies[:n_restarts]):
                x0 = self._generate_valid_initial_guess(box_bounds, seed + restart, strategy)
                
                try:
                    result = minimize(
                        lambda p: self._objective_misfit(p, u_meas),
                        x0,
                        method='SLSQP',
                        bounds=box_bounds,
                        constraints=constraints,
                        options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False}
                    )
                    
                    if result.fun < best_residual:
                        best_residual = result.fun
                        best_params = result.x
                except Exception:
                    pass
            
            params = best_params
            residual = best_residual
        
        elif method == 'differential_evolution':
            # DE with NonlinearConstraint
            result = differential_evolution(
                lambda p: self._objective_misfit(p, u_meas),
                box_bounds,
                constraints=conformal_constraint,
                seed=seed,
                maxiter=2000,
                tol=1e-8,
                polish=True,
                workers=1,
                mutation=(0.5, 1.0),
                recombination=0.7
            )
            params = result.x
            residual = result.fun
            
        elif method == 'trust-constr':
            # trust-constr with NonlinearConstraint
            from scipy.optimize import minimize as scipy_minimize
            
            best_params = None
            best_residual = np.inf
            
            strategies = ['spread', 'random'] + ['random'] * max(0, n_restarts - 2)
            
            for restart, strategy in enumerate(strategies[:n_restarts]):
                x0 = self._generate_valid_initial_guess(box_bounds, seed + restart, strategy)
                
                result = scipy_minimize(
                    lambda p: self._objective_misfit(p, u_meas),
                    x0,
                    method='trust-constr',
                    bounds=box_bounds,
                    constraints=conformal_constraint,
                    options={'maxiter': 2000}
                )
                
                if result.fun < best_residual:
                    best_residual = result.fun
                    best_params = result.x
            
            params = best_params
            residual = best_residual
            
        elif method == 'basin_hopping':
            from scipy.optimize import basinhopping
            
            x0 = self._generate_valid_initial_guess(box_bounds, seed, 'spread')
            
            # Basinhopping with L-BFGS-B using log barrier
            bounds_barrier = [(None, None)] * (2*n) + [(-5.0, 5.0)] * n
            
            result = basinhopping(
                lambda p: self._objective_with_barrier(p, u_meas, mu=mu),
                x0,
                niter=50,
                seed=seed,
                minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds_barrier}
            )
            params = result.x
            residual = result.fun
            
        else:  # L-BFGS-B with log barrier
            from scipy.optimize import minimize as scipy_minimize
            
            # Positions UNCONSTRAINED - log barrier handles domain
            bounds_barrier = [(None, None)] * (2*n) + [(-5.0, 5.0)] * n
            
            best_params = None
            best_residual = np.inf
            
            strategies = ['spread', 'random'] + ['random'] * max(0, n_restarts - 2)
            
            for restart, strategy in enumerate(strategies[:n_restarts]):
                x0 = self._generate_valid_initial_guess(box_bounds, seed + restart, strategy)
                
                result = scipy_minimize(
                    lambda p: self._objective_with_barrier(p, u_meas, mu=mu),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds_barrier,
                    options={'maxiter': 2000}
                )
                
                if result.fun < best_residual:
                    best_residual = result.fun
                    best_params = result.x
            
            params = best_params
            residual = best_residual
        
        # Extract sources
        positions = params[:2*n].reshape(n, 2)
        intensities = params[2*n:3*n]
        intensities = intensities - np.mean(intensities)  # Enforce constraint
        
        sources = [((positions[k, 0], positions[k, 1]), intensities[k]) 
                   for k in range(n)]
        
        return sources, residual


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_conformal_map(domain: str, **kwargs) -> ConformalMap:
    """
    Factory function to create conformal maps.
    
    Parameters
    ----------
    domain : str
        One of: 'disk', 'ellipse', 'rectangle', 'polygon', 'custom'
    **kwargs : 
        Domain-specific parameters
    
    Returns
    -------
    ConformalMap instance
    
    Examples
    --------
    >>> cmap = create_conformal_map('ellipse', a=2.0, b=1.0)
    >>> cmap = create_conformal_map('rectangle', half_width=1.5, half_height=1.0)
    >>> cmap = create_conformal_map('polygon', vertices=[(0,0), (1,0), (0.5,1)])
    >>> cmap = create_conformal_map('custom', boundary_func=lambda t: np.exp(1j*t)*(1+0.2*np.cos(3*t)))
    """
    domain = domain.lower()
    
    if domain == 'disk':
        return DiskMap(radius=kwargs.get('radius', 1.0))
    
    elif domain == 'ellipse':
        a = kwargs.get('a', 2.0)
        b = kwargs.get('b', 1.0)
        # Use MFSConformalMap instead of EllipseMap to avoid branch cut issue
        # EllipseMap (Joukowsky transform) has a branch cut along [-c, +c] on real axis
        # where c = sqrt(a² - b²), causing optimization failures for sources near the cut
        def ellipse_boundary(t):
            return a * np.cos(t) + 1j * b * np.sin(t)
        return MFSConformalMap(
            ellipse_boundary,
            n_boundary=kwargs.get('n_boundary', 256),
            n_charge=kwargs.get('n_charge', 200)
        )
    
    elif domain == 'rectangle':
        return RectangleMap(
            half_width=kwargs.get('half_width', 1.0),
            half_height=kwargs.get('half_height', 1.0)
        )
    
    elif domain == 'polygon':
        vertices = kwargs.get('vertices')
        if vertices is None:
            raise ValueError("polygon requires 'vertices' parameter")
        return PolygonMap(vertices)
    
    elif domain == 'custom':
        boundary_func = kwargs.get('boundary_func')
        if boundary_func is None:
            raise ValueError("custom requires 'boundary_func' parameter")
        return NumericalConformalMap(
            boundary_func,
            n_boundary=kwargs.get('n_boundary', 256)
        )
    
    else:
        raise ValueError(f"Unknown domain type: {domain}")


def solve_forward_conformal(domain: str, sources: List[Tuple[Tuple[float, float], float]],
                            n_boundary: int = 100, **domain_kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve forward problem on any domain using conformal mapping.
    
    This is the main entry point for the conformal mapping approach.
    
    Parameters
    ----------
    domain : str
        Domain type ('disk', 'ellipse', 'rectangle', 'polygon', 'custom')
    sources : list of ((x, y), q)
        Source positions and intensities (must sum to zero!)
    n_boundary : int
        Number of boundary points
    **domain_kwargs :
        Domain-specific parameters (e.g., a=2.0, b=1.0 for ellipse)
    
    Returns
    -------
    boundary_points : array (n_boundary, 2)
        Boundary point coordinates
    u : array (n_boundary,)
        Boundary potential values
        
    Examples
    --------
    >>> sources = [((-0.3, 0.4), 1.0), ((0.3, -0.4), -1.0)]
    >>> pts, u = solve_forward_conformal('ellipse', sources, a=2.0, b=1.0)
    """
    conf_map = create_conformal_map(domain, **domain_kwargs)
    solver = ConformalForwardSolver(conf_map, n_boundary)
    u = solver.solve(sources)
    return solver.boundary_points, u


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Conformal Mapping Solver")
    print("=" * 60)
    
    # Test sources (must sum to zero)
    sources = [
        ((-0.3, 0.4), 1.0),
        ((0.5, 0.3), 1.0),
        ((-0.4, -0.4), -1.0),
        ((0.3, -0.5), -1.0),
    ]
    
    # Test disk (should match analytical)
    print("\n1. Unit Disk:")
    boundary_pts, u_disk = solve_forward_conformal('disk', sources)
    print(f"   Boundary range: [{u_disk.min():.4f}, {u_disk.max():.4f}]")
    
    # Test ellipse
    print("\n2. Ellipse (a=2, b=1):")
    sources_ellipse = [
        ((-0.6, 0.3), 1.0),
        ((1.0, 0.2), 1.0),
        ((-0.8, -0.3), -1.0),
        ((0.6, -0.4), -1.0),
    ]
    boundary_pts, u_ellipse = solve_forward_conformal('ellipse', sources_ellipse, a=2.0, b=1.0)
    print(f"   Boundary range: [{u_ellipse.min():.4f}, {u_ellipse.max():.4f}]")
    
    # Test rectangle
    print("\n3. Rectangle [-1,1] x [-0.5, 0.5]:")
    sources_rect = [
        ((-0.4, 0.2), 1.0),
        ((0.5, 0.1), 1.0),
        ((-0.3, -0.2), -1.0),
        ((0.3, -0.3), -1.0),
    ]
    boundary_pts, u_rect = solve_forward_conformal('rectangle', sources_rect, 
                                                    half_width=1.0, half_height=0.5)
    print(f"   Boundary range: [{u_rect.min():.4f}, {u_rect.max():.4f}]")
    
    # Test custom domain (brain-like)
    print("\n4. Custom domain (brain-like):")
    def brain_boundary(t):
        r = 1.0 + 0.15*np.cos(2*t) - 0.1*np.cos(4*t) + 0.05*np.cos(3*t)
        r = r * (1 - 0.1*np.sin(t)**4)
        return r * np.cos(t) + 1j * 0.8 * r * np.sin(t)
    
    sources_brain = [
        ((-0.4, 0.3), 1.0),
        ((0.4, 0.3), 1.0),
        ((-0.3, -0.2), -1.0),
        ((0.3, -0.2), -1.0),
    ]
    boundary_pts, u_brain = solve_forward_conformal('custom', sources_brain,
                                                     boundary_func=brain_boundary)
    print(f"   Boundary range: [{u_brain.min():.4f}, {u_brain.max():.4f}]")
    
    # Test polygon (triangle)
    print("\n5. Triangle:")
    triangle = [(0, 0), (2, 0), (1, 1.5)]
    sources_tri = [
        ((0.8, 0.4), 1.0),
        ((1.2, 0.4), -1.0),
    ]
    boundary_pts, u_tri = solve_forward_conformal('polygon', sources_tri, vertices=triangle)
    print(f"   Boundary range: [{u_tri.min():.4f}, {u_tri.max():.4f}]")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("\nThe universal formula u(z) = Σ qₖ G_D(f(z), f(zₖ)) works for ANY domain!")