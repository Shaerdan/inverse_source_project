"""
Unified Domain Abstraction for Inverse Source Localization
============================================================

This module provides a centralized domain registry that eliminates
scattered if/elif domain_type == ... branches throughout the codebase.

Each domain class encapsulates:
- Geometry (boundary, interior check, bounds)
- Source generation
- Sensor placement
- Solver compatibility

Usage:
    from domains import DomainRegistry, get_domain
    
    # Get a domain instance
    disk = get_domain('disk')
    ellipse = get_domain('ellipse', a=2.0, b=1.0)
    
    # Use unified interface
    sources = disk.generate_sources(n_sources=6, seed=42)
    sensors = disk.get_sensor_locations(n_sensors=100)
    bounds = disk.get_optimization_bounds(n_sources=4)
    is_inside = disk.contains_point(0.5, 0.3)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DomainConfig:
    """Shared configuration for all domains."""
    # Source placement
    default_n_sources: int = 4
    min_boundary_distance: float = 0.15  # Fraction of domain size
    max_boundary_distance: float = 0.35
    
    # Sensors
    default_n_sensors: int = 100
    
    # Optimization
    intensity_bound: float = 10.0
    position_margin: float = 0.05  # Stay this far from boundary
    
    # FEM settings
    default_forward_resolution: float = 0.1
    default_source_resolution: float = 0.15


# Global default config
DEFAULT_CONFIG = DomainConfig()


class Domain(ABC):
    """
    Abstract base class for all domains.
    
    Subclasses must implement geometry-specific methods.
    Common functionality is provided by the base class.
    """
    
    # Class-level attributes
    name: str = "abstract"
    supports_analytical: bool = False
    supports_conformal: bool = False
    supports_fem: bool = True
    
    def __init__(self, config: DomainConfig = None, **params):
        """
        Initialize domain with optional parameters.
        
        Parameters
        ----------
        config : DomainConfig, optional
            Shared configuration. Uses DEFAULT_CONFIG if not provided.
        **params : dict
            Domain-specific parameters (e.g., a, b for ellipse)
        """
        self.config = config or DEFAULT_CONFIG
        self.params = params
    
    # =========================================================================
    # ABSTRACT METHODS - Must be implemented by each domain
    # =========================================================================
    
    @abstractmethod
    def get_boundary_points(self, n_points: int) -> np.ndarray:
        """
        Get n_points evenly spaced on the boundary.
        
        Returns
        -------
        points : ndarray of shape (n_points, 2)
        """
        pass
    
    @abstractmethod
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point (x, y) is inside the domain."""
        pass
    
    @abstractmethod
    def get_optimization_bounds(self, n_sources: int) -> List[Tuple[float, float]]:
        """
        Get bounds for scipy.optimize for n_sources.
        
        Returns list of (min, max) tuples for each parameter:
        [x1_bounds, y1_bounds, q1_bounds, x2_bounds, y2_bounds, q2_bounds, ...]
        """
        pass
    
    @abstractmethod
    def get_characteristic_size(self) -> float:
        """Return characteristic length scale of domain."""
        pass
    
    # =========================================================================
    # COMMON METHODS - Shared implementation, can be overridden
    # =========================================================================
    
    def get_sensor_locations(self, n_sensors: int = None) -> np.ndarray:
        """
        Get sensor locations on boundary.
        
        Default implementation uses get_boundary_points.
        Override for non-uniform sensor placement.
        """
        n = n_sensors or self.config.default_n_sensors
        return self.get_boundary_points(n)
    
    def generate_sources(self, n_sources: int = None, seed: int = 42,
                         depth_range: Tuple[float, float] = None) -> List[Tuple[Tuple[float, float], float]]:
        """
        Generate test sources inside domain.
        
        Parameters
        ----------
        n_sources : int
            Number of sources
        seed : int
            Random seed for reproducibility
        depth_range : tuple, optional
            (min_depth, max_depth) from boundary as fraction of domain size
            
        Returns
        -------
        sources : list of ((x, y), intensity) tuples
        """
        np.random.seed(seed)
        n = n_sources or self.config.default_n_sources
        
        if depth_range is None:
            depth_range = (self.config.min_boundary_distance, 
                          self.config.max_boundary_distance)
        
        # Default: angular spread
        sources = self._generate_angular_spread(n, depth_range)
        
        # Enforce zero-sum constraint
        total = sum(s[1] for s in sources)
        if abs(total) > 1e-10:
            sources[-1] = (sources[-1][0], sources[-1][1] - total)
        
        return sources
    
    @abstractmethod
    def _generate_angular_spread(self, n_sources: int, 
                                  depth_range: Tuple[float, float]) -> List[Tuple[Tuple[float, float], float]]:
        """Domain-specific angular spread source generation."""
        pass
    
    def get_conformal_map(self):
        """
        Get conformal map for this domain if supported.
        
        Returns
        -------
        conformal_map : ConformalMap or None
        """
        if not self.supports_conformal:
            return None
        return self._get_conformal_map_impl()
    
    def _get_conformal_map_impl(self):
        """Override in subclasses that support conformal mapping."""
        return None
    
    def get_fem_mesh_params(self) -> Dict[str, Any]:
        """
        Get parameters for FEM mesh generation.
        
        Returns
        -------
        params : dict with keys like 'vertices', 'resolution', etc.
        """
        return {
            'resolution': self.config.default_forward_resolution,
            'source_resolution': self.config.default_source_resolution,
        }
    
    def __repr__(self):
        params_str = ', '.join(f'{k}={v}' for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})" if params_str else f"{self.__class__.__name__}()"


# =============================================================================
# CONCRETE DOMAIN IMPLEMENTATIONS
# =============================================================================

class DiskDomain(Domain):
    """Unit disk domain centered at origin."""
    
    name = "disk"
    supports_analytical = True
    supports_conformal = True  # Identity map
    supports_fem = True
    
    def __init__(self, config: DomainConfig = None, radius: float = 1.0):
        super().__init__(config, radius=radius)
        self.radius = radius
    
    def get_boundary_points(self, n_points: int) -> np.ndarray:
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        return np.column_stack([
            self.radius * np.cos(theta),
            self.radius * np.sin(theta)
        ])
    
    def contains_point(self, x: float, y: float) -> bool:
        return x**2 + y**2 < self.radius**2
    
    def get_optimization_bounds(self, n_sources: int) -> List[Tuple[float, float]]:
        r_max = self.radius * (1 - self.config.position_margin)
        q_max = self.config.intensity_bound
        bounds = []
        for _ in range(n_sources):
            bounds.extend([
                (-r_max, r_max),  # x
                (-r_max, r_max),  # y
                (-q_max, q_max),  # intensity
            ])
        return bounds
    
    def get_characteristic_size(self) -> float:
        return self.radius
    
    def _generate_angular_spread(self, n_sources: int, 
                                  depth_range: Tuple[float, float]) -> List[Tuple[Tuple[float, float], float]]:
        r_min = self.radius * (1 - depth_range[1])
        r_max = self.radius * (1 - depth_range[0])
        r_mean = (r_min + r_max) / 2
        
        angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
        sources = []
        for i, theta in enumerate(angles):
            x = r_mean * np.cos(theta)
            y = r_mean * np.sin(theta)
            intensity = 1.0 if i % 2 == 0 else -1.0
            sources.append(((x, y), intensity))
        return sources


class EllipseDomain(Domain):
    """Ellipse domain with semi-axes a, b."""
    
    name = "ellipse"
    supports_analytical = False
    supports_conformal = True  # Joukowsky map
    supports_fem = True
    
    def __init__(self, config: DomainConfig = None, a: float = 2.0, b: float = 1.0):
        super().__init__(config, a=a, b=b)
        self.a = a
        self.b = b
    
    def get_boundary_points(self, n_points: int) -> np.ndarray:
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        return np.column_stack([
            self.a * np.cos(theta),
            self.b * np.sin(theta)
        ])
    
    def contains_point(self, x: float, y: float) -> bool:
        return (x/self.a)**2 + (y/self.b)**2 < 1
    
    def get_optimization_bounds(self, n_sources: int) -> List[Tuple[float, float]]:
        margin = self.config.position_margin
        x_max = self.a * (1 - margin)
        y_max = self.b * (1 - margin)
        q_max = self.config.intensity_bound
        bounds = []
        for _ in range(n_sources):
            bounds.extend([
                (-x_max, x_max),
                (-y_max, y_max),
                (-q_max, q_max),
            ])
        return bounds
    
    def get_characteristic_size(self) -> float:
        return max(self.a, self.b)
    
    def _generate_angular_spread(self, n_sources: int,
                                  depth_range: Tuple[float, float]) -> List[Tuple[Tuple[float, float], float]]:
        scale = 1 - (depth_range[0] + depth_range[1]) / 2
        angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False) + 0.3
        sources = []
        for i, theta in enumerate(angles):
            x = scale * self.a * np.cos(theta)
            y = scale * self.b * np.sin(theta)
            intensity = 1.0 if i % 2 == 0 else -1.0
            sources.append(((x, y), intensity))
        return sources
    
    def _get_conformal_map_impl(self):
        try:
            from conformal_solver import EllipseMap
            return EllipseMap(self.a, self.b)
        except ImportError:
            return None


class StarDomain(Domain):
    """Star-shaped domain r(θ) = 1 + amplitude * cos(n_petals * θ)."""
    
    name = "star"
    supports_analytical = False
    supports_conformal = True  # Numerical MFS map
    supports_fem = True
    
    def __init__(self, config: DomainConfig = None, 
                 n_petals: int = 5, amplitude: float = 0.3):
        super().__init__(config, n_petals=n_petals, amplitude=amplitude)
        self.n_petals = n_petals
        self.amplitude = amplitude
    
    def _radius_at_angle(self, theta: float) -> float:
        return 1.0 + self.amplitude * np.cos(self.n_petals * theta)
    
    def get_boundary_points(self, n_points: int) -> np.ndarray:
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        r = self._radius_at_angle(theta)
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    def contains_point(self, x: float, y: float) -> bool:
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return r < self._radius_at_angle(theta)
    
    def get_optimization_bounds(self, n_sources: int) -> List[Tuple[float, float]]:
        # Use inner radius as safe bound
        r_inner = 1.0 - self.amplitude
        r_max = r_inner * (1 - self.config.position_margin)
        q_max = self.config.intensity_bound
        bounds = []
        for _ in range(n_sources):
            bounds.extend([
                (-r_max, r_max),
                (-r_max, r_max),
                (-q_max, q_max),
            ])
        return bounds
    
    def get_characteristic_size(self) -> float:
        return 1.0 + self.amplitude
    
    def _generate_angular_spread(self, n_sources: int,
                                  depth_range: Tuple[float, float]) -> List[Tuple[Tuple[float, float], float]]:
        target_depth = (depth_range[0] + depth_range[1]) / 2
        angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
        sources = []
        for i, theta in enumerate(angles):
            r_boundary = self._radius_at_angle(theta)
            r = max(0.2, r_boundary - target_depth)
            x, y = r * np.cos(theta), r * np.sin(theta)
            intensity = 1.0 if i % 2 == 0 else -1.0
            sources.append(((x, y), intensity))
        return sources
    
    def _get_conformal_map_impl(self):
        try:
            from conformal_solver import StarShapedMap
            return StarShapedMap(self._radius_at_angle, n_terms=32)
        except ImportError:
            return None


class SquareDomain(Domain):
    """Square domain [-1, 1]²."""
    
    name = "square"
    supports_analytical = False
    supports_conformal = True  # Schwarz-Christoffel (numerical)
    supports_fem = True
    
    def __init__(self, config: DomainConfig = None, half_width: float = 1.0):
        super().__init__(config, half_width=half_width)
        self.half_width = half_width
        self.vertices = [
            (-half_width, -half_width),
            (half_width, -half_width),
            (half_width, half_width),
            (-half_width, half_width)
        ]
    
    def get_boundary_points(self, n_points: int) -> np.ndarray:
        # Distribute points along 4 sides
        n_per_side = n_points // 4
        hw = self.half_width
        points = []
        
        # Bottom: left to right
        for i in range(n_per_side):
            t = i / n_per_side
            points.append((-hw + 2*hw*t, -hw))
        # Right: bottom to top
        for i in range(n_per_side):
            t = i / n_per_side
            points.append((hw, -hw + 2*hw*t))
        # Top: right to left
        for i in range(n_per_side):
            t = i / n_per_side
            points.append((hw - 2*hw*t, hw))
        # Left: top to bottom
        for i in range(n_points - 3*n_per_side):
            t = i / (n_points - 3*n_per_side)
            points.append((-hw, hw - 2*hw*t))
        
        return np.array(points)
    
    def contains_point(self, x: float, y: float) -> bool:
        hw = self.half_width
        return abs(x) < hw and abs(y) < hw
    
    def get_optimization_bounds(self, n_sources: int) -> List[Tuple[float, float]]:
        margin = self.config.position_margin
        xy_max = self.half_width * (1 - margin)
        q_max = self.config.intensity_bound
        bounds = []
        for _ in range(n_sources):
            bounds.extend([
                (-xy_max, xy_max),
                (-xy_max, xy_max),
                (-q_max, q_max),
            ])
        return bounds
    
    def get_characteristic_size(self) -> float:
        return self.half_width * 2
    
    def _generate_angular_spread(self, n_sources: int,
                                  depth_range: Tuple[float, float]) -> List[Tuple[Tuple[float, float], float]]:
        offset = self.half_width * (1 - (depth_range[0] + depth_range[1]) / 2)
        angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False) + np.pi/4
        sources = []
        for i, theta in enumerate(angles):
            x = offset * np.cos(theta)
            y = offset * np.sin(theta)
            # Clamp to square
            x = np.clip(x, -0.85 * self.half_width, 0.85 * self.half_width)
            y = np.clip(y, -0.85 * self.half_width, 0.85 * self.half_width)
            intensity = 1.0 if i % 2 == 0 else -1.0
            sources.append(((x, y), intensity))
        return sources
    
    def get_fem_mesh_params(self) -> Dict[str, Any]:
        params = super().get_fem_mesh_params()
        params['vertices'] = self.vertices
        return params


class BrainDomain(Domain):
    """Realistic brain-shaped domain."""
    
    name = "brain"
    supports_analytical = False
    supports_conformal = True  # MFS numerical
    supports_fem = True
    
    def __init__(self, config: DomainConfig = None, n_boundary_points: int = 100):
        super().__init__(config, n_boundary_points=n_boundary_points)
        self._n_boundary = n_boundary_points
        self._boundary_cache = None
    
    def _get_brain_boundary(self, n_points: int) -> np.ndarray:
        """Generate brain-shaped boundary."""
        try:
            from mesh import get_brain_boundary
            return get_brain_boundary(n_points)
        except ImportError:
            # Fallback: approximate brain as deformed ellipse
            theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
            # Brain shape: wider at top, narrower at bottom
            r_base = 0.8 + 0.2 * np.cos(theta)  # Basic shape
            r_brain = r_base * (1 + 0.1 * np.cos(2*theta) + 0.05 * np.cos(3*theta))
            x = r_brain * np.cos(theta)
            y = r_brain * np.sin(theta) * 0.7 + 0.05  # Compress vertically, shift up
            return np.column_stack([x, y])
    
    def get_boundary_points(self, n_points: int) -> np.ndarray:
        return self._get_brain_boundary(n_points)
    
    def contains_point(self, x: float, y: float) -> bool:
        # Approximate: use bounding ellipse check
        return (x/1.1)**2 + ((y-0.05)/0.65)**2 < 1
    
    def get_optimization_bounds(self, n_sources: int) -> List[Tuple[float, float]]:
        # Conservative bounds for brain interior
        q_max = self.config.intensity_bound
        bounds = []
        for _ in range(n_sources):
            bounds.extend([
                (-0.9, 0.9),   # x
                (-0.5, 0.6),   # y (brain is asymmetric)
                (-q_max, q_max),
            ])
        return bounds
    
    def get_characteristic_size(self) -> float:
        return 1.1  # Approximate max extent
    
    def _generate_angular_spread(self, n_sources: int,
                                  depth_range: Tuple[float, float]) -> List[Tuple[Tuple[float, float], float]]:
        x_scale = 0.7
        y_scale = 0.4
        y_offset = 0.05
        angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False) + 0.2
        sources = []
        for i, theta in enumerate(angles):
            x = x_scale * np.cos(theta)
            y = y_scale * np.sin(theta) + y_offset
            intensity = 1.0 if i % 2 == 0 else -1.0
            sources.append(((x, y), intensity))
        return sources
    
    def get_fem_mesh_params(self) -> Dict[str, Any]:
        params = super().get_fem_mesh_params()
        boundary = self.get_boundary_points(100)
        params['vertices'] = [tuple(p) for p in boundary]
        return params


class PolygonDomain(Domain):
    """Generic polygon domain (archived - use with caution)."""
    
    name = "polygon"
    supports_analytical = False
    supports_conformal = False  # MFS has issues with non-convex
    supports_fem = True
    
    def __init__(self, config: DomainConfig = None, 
                 vertices: List[Tuple[float, float]] = None):
        if vertices is None:
            # Default L-shape
            vertices = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        super().__init__(config, vertices=vertices)
        self.vertices = vertices
        self._verts_array = np.array(vertices)
    
    def get_boundary_points(self, n_points: int) -> np.ndarray:
        """Distribute points along polygon edges."""
        verts = self._verts_array
        n_verts = len(verts)
        
        # Compute total perimeter
        perimeter = 0
        edge_lengths = []
        for i in range(n_verts):
            length = np.linalg.norm(verts[(i+1) % n_verts] - verts[i])
            edge_lengths.append(length)
            perimeter += length
        
        # Distribute points proportionally
        points = []
        for i in range(n_verts):
            n_on_edge = max(1, int(n_points * edge_lengths[i] / perimeter))
            v1, v2 = verts[i], verts[(i+1) % n_verts]
            for j in range(n_on_edge):
                t = j / n_on_edge
                points.append(v1 + t * (v2 - v1))
        
        return np.array(points[:n_points])
    
    def contains_point(self, x: float, y: float) -> bool:
        """Ray casting algorithm."""
        verts = self._verts_array
        n = len(verts)
        inside = False
        j = n - 1
        for i in range(n):
            if ((verts[i, 1] > y) != (verts[j, 1] > y) and
                x < (verts[j, 0] - verts[i, 0]) * (y - verts[i, 1]) / 
                    (verts[j, 1] - verts[i, 1]) + verts[i, 0]):
                inside = not inside
            j = i
        return inside
    
    def get_optimization_bounds(self, n_sources: int) -> List[Tuple[float, float]]:
        verts = self._verts_array
        x_min, y_min = verts.min(axis=0)
        x_max, y_max = verts.max(axis=0)
        margin = 0.1
        q_max = self.config.intensity_bound
        bounds = []
        for _ in range(n_sources):
            bounds.extend([
                (x_min + margin, x_max - margin),
                (y_min + margin, y_max - margin),
                (-q_max, q_max),
            ])
        return bounds
    
    def get_characteristic_size(self) -> float:
        verts = self._verts_array
        return max(verts.max(axis=0) - verts.min(axis=0))
    
    def _generate_angular_spread(self, n_sources: int,
                                  depth_range: Tuple[float, float]) -> List[Tuple[Tuple[float, float], float]]:
        cx = np.mean(self._verts_array[:, 0])
        cy = np.mean(self._verts_array[:, 1])
        angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
        sources = []
        for i, theta in enumerate(angles):
            x = cx + 0.3 * np.cos(theta)
            y = cy + 0.3 * np.sin(theta)
            intensity = 1.0 if i % 2 == 0 else -1.0
            sources.append(((x, y), intensity))
        return sources
    
    def get_fem_mesh_params(self) -> Dict[str, Any]:
        params = super().get_fem_mesh_params()
        params['vertices'] = self.vertices
        return params


# =============================================================================
# DOMAIN REGISTRY
# =============================================================================

class DomainRegistry:
    """
    Central registry for all domain types.
    
    Usage:
        registry = DomainRegistry()
        disk = registry.get('disk')
        ellipse = registry.get('ellipse', a=2.0, b=1.0)
        
        # List available domains
        print(registry.list_domains())
        
        # Check capabilities
        print(registry.get_domains_supporting('conformal'))
    """
    
    _domains = {
        'disk': DiskDomain,
        'ellipse': EllipseDomain,
        'star': StarDomain,
        'square': SquareDomain,
        'brain': BrainDomain,
        'polygon': PolygonDomain,
    }
    
    # Domains included in standard tests (polygon archived)
    _active_domains = ['disk', 'ellipse', 'star', 'square', 'brain']
    
    def __init__(self, config: DomainConfig = None):
        self.config = config or DEFAULT_CONFIG
    
    def get(self, domain_type: str, **params) -> Domain:
        """
        Get a domain instance by type.
        
        Parameters
        ----------
        domain_type : str
            One of 'disk', 'ellipse', 'star', 'square', 'brain', 'polygon'
        **params : dict
            Domain-specific parameters
            
        Returns
        -------
        domain : Domain instance
        """
        if domain_type not in self._domains:
            available = ', '.join(self._domains.keys())
            raise ValueError(f"Unknown domain '{domain_type}'. Available: {available}")
        
        return self._domains[domain_type](config=self.config, **params)
    
    def list_domains(self, active_only: bool = True) -> List[str]:
        """List available domain types."""
        if active_only:
            return self._active_domains.copy()
        return list(self._domains.keys())
    
    def get_domains_supporting(self, feature: str) -> List[str]:
        """
        Get domains supporting a specific feature.
        
        Parameters
        ----------
        feature : str
            One of 'analytical', 'conformal', 'fem'
        """
        attr = f'supports_{feature}'
        return [name for name, cls in self._domains.items() 
                if getattr(cls, attr, False)]
    
    def get_all_active(self, **default_params) -> Dict[str, Domain]:
        """Get all active domain instances."""
        domains = {}
        for name in self._active_domains:
            params = default_params.get(name, {})
            domains[name] = self.get(name, **params)
        return domains


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global registry instance
_registry = DomainRegistry()


def get_domain(domain_type: str, **params) -> Domain:
    """
    Convenience function to get a domain instance.
    
    Examples
    --------
    >>> disk = get_domain('disk')
    >>> ellipse = get_domain('ellipse', a=2.0, b=1.0)
    >>> star = get_domain('star', n_petals=5, amplitude=0.3)
    """
    return _registry.get(domain_type, **params)


def list_domains(active_only: bool = True) -> List[str]:
    """List available domain types."""
    return _registry.list_domains(active_only)


def get_active_domains(**default_params) -> Dict[str, Domain]:
    """Get all active domain instances."""
    return _registry.get_all_active(**default_params)


# =============================================================================
# MIGRATION HELPERS - For gradual transition from old code
# =============================================================================

def create_domain_sources_unified(domain_type: str, domain_params: dict = None,
                                   n_sources: int = 4, **kwargs) -> List[Tuple[Tuple[float, float], float]]:
    """
    Drop-in replacement for comparison.create_domain_sources().
    
    This function bridges the old API to the new Domain abstraction.
    """
    # Map old params to new format
    params = {}
    if domain_params:
        if domain_type == 'ellipse':
            params = {'a': domain_params.get('a', 2.0), 'b': domain_params.get('b', 1.0)}
        elif domain_type == 'star':
            params = {'n_petals': domain_params.get('n_petals', 5),
                     'amplitude': domain_params.get('amplitude', 0.3)}
        elif domain_type in ['square', 'polygon']:
            if 'vertices' in domain_params:
                params = {'vertices': domain_params['vertices']}
    
    domain = get_domain(domain_type, **params)
    return domain.generate_sources(n_sources=n_sources, **kwargs)


def get_sensor_locations_unified(domain_type: str, domain_params: dict = None,
                                  n_sensors: int = 100) -> np.ndarray:
    """
    Drop-in replacement for comparison.get_sensor_locations().
    """
    params = {}
    if domain_params:
        if domain_type == 'ellipse':
            params = {'a': domain_params.get('a', 2.0), 'b': domain_params.get('b', 1.0)}
        elif domain_type == 'star':
            params = {'n_petals': domain_params.get('n_petals', 5),
                     'amplitude': domain_params.get('amplitude', 0.3)}
    
    domain = get_domain(domain_type, **params)
    return domain.get_sensor_locations(n_sensors)


if __name__ == "__main__":
    # Demo
    print("Domain Registry Demo")
    print("=" * 50)
    
    print("\nActive domains:", list_domains())
    print("All domains:", list_domains(active_only=False))
    print("Conformal-capable:", _registry.get_domains_supporting('conformal'))
    print("Analytical-capable:", _registry.get_domains_supporting('analytical'))
    
    print("\n" + "=" * 50)
    print("Testing source generation with n_sources=6:")
    print("=" * 50)
    
    for domain_name in list_domains():
        domain = get_domain(domain_name)
        sources = domain.generate_sources(n_sources=6, seed=42)
        total_intensity = sum(s[1] for s in sources)
        print(f"\n{domain_name}:")
        print(f"  {len(sources)} sources, sum(q) = {total_intensity:.2e}")
        for i, ((x, y), q) in enumerate(sources):
            print(f"    {i+1}: ({x:+.3f}, {y:+.3f}), q={q:+.2f}")
