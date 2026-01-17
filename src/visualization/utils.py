"""
Visualization Utilities
=======================

Helper functions for domain geometry, source matching, and plotting.
These utilities are used throughout the visualization toolbox.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Ellipse
from matplotlib.collections import LineCollection
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict, Optional, Union
import warnings

from .config import COLORS, MARKERS, LINESTYLES, get_source_color, get_source_marker


# =============================================================================
# DOMAIN GEOMETRY
# =============================================================================

def get_domain_boundary(domain_type: str, domain_params: dict = None,
                        n_points: int = 200) -> np.ndarray:
    """
    Get boundary points for a domain.

    Parameters
    ----------
    domain_type : str
        One of: 'disk', 'ellipse', 'square', 'rectangle', 'star', 'polygon', 'brain'
    domain_params : dict, optional
        Domain-specific parameters:
        - disk: {'radius': 1.0}
        - ellipse: {'a': 2.0, 'b': 1.0}
        - square: {'side': 2.0} or {'half_side': 1.0}
        - rectangle: {'width': 2.0, 'height': 1.0} or {'half_width': 1.0, 'half_height': 0.5}
        - star: {'n_points': 5, 'r_outer': 1.0, 'r_inner': 0.5}
        - polygon: {'vertices': [(x1,y1), (x2,y2), ...]}
        - brain: {'scale': 1.0}
    n_points : int
        Number of boundary points

    Returns
    -------
    boundary : ndarray, shape (n_points, 2)
        Boundary point coordinates (x, y)
    """
    if domain_params is None:
        domain_params = {}

    domain_type = domain_type.lower()

    if domain_type == 'disk':
        radius = domain_params.get('radius', 1.0)
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        boundary = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])

    elif domain_type == 'ellipse':
        a = domain_params.get('a', 2.0)
        b = domain_params.get('b', 1.0)
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        boundary = np.column_stack([a * np.cos(theta), b * np.sin(theta)])

    elif domain_type == 'square':
        side = domain_params.get('side', 2.0)
        half = domain_params.get('half_side', side / 2)
        boundary = _rectangle_boundary(half, half, n_points)

    elif domain_type == 'rectangle':
        width = domain_params.get('width', 2.0)
        height = domain_params.get('height', 1.0)
        half_w = domain_params.get('half_width', width / 2)
        half_h = domain_params.get('half_height', height / 2)
        boundary = _rectangle_boundary(half_w, half_h, n_points)

    elif domain_type == 'star':
        n_star = domain_params.get('n_points', 5)
        r_outer = domain_params.get('r_outer', 1.0)
        r_inner = domain_params.get('r_inner', 0.5)
        boundary = _star_boundary(n_star, r_outer, r_inner, n_points)

    elif domain_type == 'polygon':
        vertices = domain_params.get('vertices')
        if vertices is None:
            raise ValueError("polygon domain requires 'vertices' parameter")
        boundary = _polygon_boundary(vertices, n_points)

    elif domain_type == 'brain':
        scale = domain_params.get('scale', 1.0)
        boundary = _brain_boundary(scale, n_points)

    else:
        raise ValueError(f"Unknown domain type: {domain_type}")

    return boundary


def _rectangle_boundary(half_width: float, half_height: float, n_points: int) -> np.ndarray:
    """Generate rectangle boundary points."""
    n_per_side = n_points // 4
    pts = []

    # Bottom: -w to +w at y = -h
    pts.extend(np.column_stack([
        np.linspace(-half_width, half_width, n_per_side, endpoint=False),
        np.full(n_per_side, -half_height)
    ]))

    # Right: +w, -h to +h
    pts.extend(np.column_stack([
        np.full(n_per_side, half_width),
        np.linspace(-half_height, half_height, n_per_side, endpoint=False)
    ]))

    # Top: +w to -w at y = +h
    pts.extend(np.column_stack([
        np.linspace(half_width, -half_width, n_per_side, endpoint=False),
        np.full(n_per_side, half_height)
    ]))

    # Left: -w, +h to -h
    pts.extend(np.column_stack([
        np.full(n_per_side, -half_width),
        np.linspace(half_height, -half_height, n_per_side, endpoint=False)
    ]))

    return np.vstack(pts[:n_points])


def _star_boundary(n_star: int, r_outer: float, r_inner: float, n_points: int) -> np.ndarray:
    """Generate star-shaped boundary points."""
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    r = np.zeros(n_points)

    for i, t in enumerate(theta):
        # Angle within one star segment
        segment = (t * n_star / (2*np.pi)) % 1
        if segment < 0.5:
            # Going from outer to inner
            r[i] = r_outer - (r_outer - r_inner) * (segment * 2)
        else:
            # Going from inner to outer
            r[i] = r_inner + (r_outer - r_inner) * ((segment - 0.5) * 2)

    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _polygon_boundary(vertices: List[Tuple[float, float]], n_points: int) -> np.ndarray:
    """Generate polygon boundary by interpolating along edges."""
    vertices = np.array(vertices)
    n_verts = len(vertices)

    # Compute edge lengths
    edges = np.diff(np.vstack([vertices, vertices[0:1]]), axis=0)
    edge_lengths = np.linalg.norm(edges, axis=1)
    total_length = np.sum(edge_lengths)

    # Distribute points proportionally
    pts = []
    for i in range(n_verts):
        v_start = vertices[i]
        v_end = vertices[(i + 1) % n_verts]
        n_edge = max(1, int(n_points * edge_lengths[i] / total_length))
        t = np.linspace(0, 1, n_edge, endpoint=False)
        edge_pts = v_start + np.outer(t, v_end - v_start)
        pts.append(edge_pts)

    boundary = np.vstack(pts)

    # Ensure exactly n_points
    if len(boundary) > n_points:
        boundary = boundary[:n_points]
    elif len(boundary) < n_points:
        # Resample to exact count
        cumlen = np.cumsum(np.linalg.norm(np.diff(boundary, axis=0), axis=1))
        cumlen = np.insert(cumlen, 0, 0)
        target = np.linspace(0, cumlen[-1], n_points, endpoint=False)
        boundary = np.column_stack([
            np.interp(target, cumlen, boundary[:, 0]),
            np.interp(target, cumlen, boundary[:, 1])
        ])

    return boundary


def _brain_boundary(scale: float, n_points: int) -> np.ndarray:
    """Generate brain-like boundary (deformed ellipse)."""
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    r = 1.0 + 0.15*np.cos(2*theta) - 0.1*np.cos(4*theta) + 0.05*np.cos(3*theta)
    r = r * (1 - 0.1*np.sin(theta)**4)
    x = scale * r * np.cos(theta)
    y = scale * 0.8 * r * np.sin(theta)
    return np.column_stack([x, y])


def domain_mask(xx: np.ndarray, yy: np.ndarray,
                domain_type: str, domain_params: dict = None) -> np.ndarray:
    """
    Create boolean mask for points inside domain.

    Parameters
    ----------
    xx, yy : ndarray
        Meshgrid of x and y coordinates
    domain_type : str
        Domain type (see get_domain_boundary)
    domain_params : dict
        Domain parameters

    Returns
    -------
    mask : ndarray
        Boolean array, True for points inside domain
    """
    if domain_params is None:
        domain_params = {}

    domain_type = domain_type.lower()

    if domain_type == 'disk':
        radius = domain_params.get('radius', 1.0)
        return xx**2 + yy**2 < radius**2

    elif domain_type == 'ellipse':
        a = domain_params.get('a', 2.0)
        b = domain_params.get('b', 1.0)
        return (xx/a)**2 + (yy/b)**2 < 1.0

    elif domain_type in ('square', 'rectangle'):
        if domain_type == 'square':
            half_w = domain_params.get('half_side', domain_params.get('side', 2.0) / 2)
            half_h = half_w
        else:
            half_w = domain_params.get('half_width', domain_params.get('width', 2.0) / 2)
            half_h = domain_params.get('half_height', domain_params.get('height', 1.0) / 2)
        return (np.abs(xx) < half_w) & (np.abs(yy) < half_h)

    elif domain_type == 'star':
        # Use polygon approximation
        boundary = get_domain_boundary(domain_type, domain_params, n_points=200)
        return _points_in_polygon(xx, yy, boundary)

    elif domain_type == 'polygon':
        vertices = domain_params.get('vertices')
        if vertices is None:
            raise ValueError("polygon domain requires 'vertices' parameter")
        return _points_in_polygon(xx, yy, np.array(vertices))

    elif domain_type == 'brain':
        boundary = get_domain_boundary(domain_type, domain_params, n_points=200)
        return _points_in_polygon(xx, yy, boundary)

    else:
        raise ValueError(f"Unknown domain type: {domain_type}")


def _points_in_polygon(xx: np.ndarray, yy: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Check if meshgrid points are inside polygon using ray casting."""
    from matplotlib.path import Path
    path = Path(vertices)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    inside = path.contains_points(points)
    return inside.reshape(xx.shape)


def point_in_domain(x: float, y: float, domain_type: str,
                    domain_params: dict = None) -> bool:
    """
    Check if a single point is inside domain.

    Parameters
    ----------
    x, y : float
        Point coordinates
    domain_type : str
        Domain type
    domain_params : dict
        Domain parameters

    Returns
    -------
    bool
        True if point is inside domain
    """
    xx = np.array([[x]])
    yy = np.array([[y]])
    return domain_mask(xx, yy, domain_type, domain_params)[0, 0]


def get_bounding_box(domain_boundary: np.ndarray,
                     margin: float = 0.1) -> Tuple[float, float, float, float]:
    """
    Get bounding box for domain with margin.

    Parameters
    ----------
    domain_boundary : ndarray, shape (n, 2)
        Boundary points
    margin : float
        Fractional margin to add (0.1 = 10%)

    Returns
    -------
    x_min, x_max, y_min, y_max : float
        Bounding box coordinates
    """
    x_min, y_min = domain_boundary.min(axis=0)
    x_max, y_max = domain_boundary.max(axis=0)

    dx = (x_max - x_min) * margin
    dy = (y_max - y_min) * margin

    return x_min - dx, x_max + dx, y_min - dy, y_max + dy


# =============================================================================
# SOURCE UTILITIES
# =============================================================================

def sources_to_arrays(sources: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert sources list to position and intensity arrays.

    Parameters
    ----------
    sources : list of ((x, y), intensity) or Source objects
        Source list

    Returns
    -------
    positions : ndarray, shape (n, 2)
        Source positions
    intensities : ndarray, shape (n,)
        Source intensities
    """
    positions = []
    intensities = []

    for s in sources:
        if hasattr(s, 'x'):
            # Source dataclass
            positions.append([s.x, s.y])
            intensities.append(s.intensity)
        else:
            # Tuple format ((x, y), q)
            positions.append([s[0][0], s[0][1]])
            intensities.append(s[1])

    return np.array(positions), np.array(intensities)


def match_sources(sources_true: List[Tuple],
                  sources_recovered: List[Tuple]) -> List[Tuple[int, int, float]]:
    """
    Match true and recovered sources using Hungarian algorithm.

    Parameters
    ----------
    sources_true : list
        True source list
    sources_recovered : list
        Recovered source list

    Returns
    -------
    matching : list of (true_idx, recovered_idx, distance)
        Optimal matching with position distances
    """
    pos_true, _ = sources_to_arrays(sources_true)
    pos_rec, _ = sources_to_arrays(sources_recovered)

    n_true = len(pos_true)
    n_rec = len(pos_rec)

    if n_true == 0 or n_rec == 0:
        return []

    # Build cost matrix (Euclidean distance)
    cost_matrix = np.zeros((n_true, n_rec))
    for i in range(n_true):
        for j in range(n_rec):
            cost_matrix[i, j] = np.linalg.norm(pos_true[i] - pos_rec[j])

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matching = []
    for i, j in zip(row_ind, col_ind):
        matching.append((i, j, cost_matrix[i, j]))

    return matching


def compute_source_errors(sources_true: List[Tuple],
                          sources_recovered: List[Tuple]) -> Dict:
    """
    Compute error metrics after matching sources.

    Parameters
    ----------
    sources_true : list
        True source list
    sources_recovered : list
        Recovered source list

    Returns
    -------
    errors : dict
        Dictionary with keys:
        - 'position_rmse': RMSE of position errors
        - 'position_max': Maximum position error
        - 'intensity_rmse': RMSE of intensity errors
        - 'intensity_max': Maximum intensity error
        - 'matching': list of (true_idx, rec_idx, dist)
        - 'position_errors': list of individual position errors
        - 'intensity_errors': list of individual intensity errors
    """
    pos_true, int_true = sources_to_arrays(sources_true)
    pos_rec, int_rec = sources_to_arrays(sources_recovered)

    matching = match_sources(sources_true, sources_recovered)

    if not matching:
        return {
            'position_rmse': np.nan,
            'position_max': np.nan,
            'intensity_rmse': np.nan,
            'intensity_max': np.nan,
            'matching': [],
            'position_errors': [],
            'intensity_errors': [],
        }

    position_errors = []
    intensity_errors = []

    for true_idx, rec_idx, dist in matching:
        position_errors.append(dist)
        intensity_errors.append(abs(int_true[true_idx] - int_rec[rec_idx]))

    position_errors = np.array(position_errors)
    intensity_errors = np.array(intensity_errors)

    return {
        'position_rmse': np.sqrt(np.mean(position_errors**2)),
        'position_max': np.max(position_errors),
        'intensity_rmse': np.sqrt(np.mean(intensity_errors**2)),
        'intensity_max': np.max(intensity_errors),
        'matching': matching,
        'position_errors': position_errors.tolist(),
        'intensity_errors': intensity_errors.tolist(),
    }


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def add_domain_boundary(ax: plt.Axes, boundary: np.ndarray,
                        fill: bool = False, **kwargs) -> None:
    """
    Add domain boundary line (and optional fill) to axes.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    boundary : ndarray, shape (n, 2)
        Boundary points
    fill : bool
        Whether to fill the domain interior
    **kwargs
        Additional arguments for plot/fill
    """
    # Close the boundary
    boundary_closed = np.vstack([boundary, boundary[0:1]])

    if fill:
        fill_color = kwargs.pop('facecolor', COLORS['domain_fill'])
        fill_alpha = kwargs.pop('alpha', 0.3)
        ax.fill(boundary_closed[:, 0], boundary_closed[:, 1],
                facecolor=fill_color, alpha=fill_alpha, edgecolor='none')

    # Draw boundary line
    line_color = kwargs.pop('color', COLORS['boundary'])
    line_width = kwargs.pop('linewidth', LINESTYLES['boundary']['linewidth'])
    ax.plot(boundary_closed[:, 0], boundary_closed[:, 1],
            color=line_color, linewidth=line_width, **kwargs)


def add_source_markers(ax: plt.Axes, sources: List[Tuple],
                       style: str = 'true', show_labels: bool = False,
                       label_offset: Tuple[float, float] = (0.05, 0.05),
                       **kwargs) -> None:
    """
    Add source markers to axes.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    sources : list
        Source list
    style : str
        'true' (filled circles) or 'recovered' (hollow squares)
    show_labels : bool
        Whether to show source labels
    label_offset : tuple
        Offset for labels (dx, dy)
    **kwargs
        Additional marker arguments
    """
    positions, intensities = sources_to_arrays(sources)

    marker_style = get_source_marker(style)
    marker_style.update(kwargs)

    for i, (pos, intensity) in enumerate(zip(positions, intensities)):
        color = get_source_color(intensity)

        if style == 'recovered':
            ax.plot(pos[0], pos[1], color=color, markeredgecolor=color,
                    markerfacecolor='none', **marker_style)
        else:
            ax.plot(pos[0], pos[1], color=color, **marker_style)

        if show_labels:
            ax.annotate(f'{i+1}', (pos[0] + label_offset[0], pos[1] + label_offset[1]),
                       fontsize=8, ha='left', va='bottom')


def add_sensor_markers(ax: plt.Axes, sensor_locations: np.ndarray,
                       **kwargs) -> None:
    """
    Add small gray dots for sensor locations.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    sensor_locations : ndarray, shape (n, 2)
        Sensor positions
    **kwargs
        Additional marker arguments
    """
    marker_style = MARKERS['sensor'].copy()
    marker_style.update(kwargs)
    color = kwargs.pop('color', COLORS['sensor'])

    ax.plot(sensor_locations[:, 0], sensor_locations[:, 1],
            linestyle='none', color=color, **marker_style)


def add_colorbar(ax: plt.Axes, mappable, label: str = None,
                 **kwargs):
    """
    Add colorbar with label.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    mappable : ScalarMappable
        The mappable (e.g., from imshow or contourf)
    label : str, optional
        Colorbar label
    **kwargs
        Additional colorbar arguments

    Returns
    -------
    cbar : Colorbar
        The colorbar object
    """
    cbar = plt.colorbar(mappable, ax=ax, **kwargs)
    if label:
        cbar.set_label(label)
    return cbar


def set_domain_axes(ax: plt.Axes, domain_boundary: np.ndarray,
                    margin: float = 0.1) -> None:
    """
    Set equal aspect ratio and appropriate axis limits.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    domain_boundary : ndarray
        Boundary points
    margin : float
        Fractional margin
    """
    x_min, x_max, y_min, y_max = get_bounding_box(domain_boundary, margin)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')


def add_matching_arrows(ax: plt.Axes, sources_true: List[Tuple],
                        sources_recovered: List[Tuple],
                        matching: List[Tuple[int, int, float]] = None,
                        color_by_error: bool = True,
                        **kwargs) -> None:
    """
    Add arrows from recovered to true source positions.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    sources_true, sources_recovered : list
        Source lists
    matching : list, optional
        Pre-computed matching. If None, computed automatically.
    color_by_error : bool
        If True, color arrows by error magnitude
    **kwargs
        Additional arrow arguments
    """
    if matching is None:
        matching = match_sources(sources_true, sources_recovered)

    pos_true, _ = sources_to_arrays(sources_true)
    pos_rec, _ = sources_to_arrays(sources_recovered)

    max_error = max(m[2] for m in matching) if matching else 1.0

    for true_idx, rec_idx, error in matching:
        start = pos_rec[rec_idx]
        end = pos_true[true_idx]

        if color_by_error and max_error > 0:
            # Color from green (small error) to red (large error)
            intensity = error / max_error
            color = plt.cm.RdYlGn_r(intensity)
        else:
            color = COLORS['trajectory']

        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color=color,
                                  lw=1.5, **kwargs))


def format_axes_pi(ax: plt.Axes, axis: str = 'x') -> None:
    """
    Format axis ticks in multiples of pi.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes
    axis : str
        'x' or 'y'
    """
    import matplotlib.ticker as ticker

    def pi_formatter(x, pos):
        if x == 0:
            return '0'
        elif x == np.pi:
            return r'$\pi$'
        elif x == 2*np.pi:
            return r'$2\pi$'
        elif x == -np.pi:
            return r'$-\pi$'
        elif x % (np.pi/2) == 0:
            n = int(x / (np.pi/2))
            if n == 1:
                return r'$\pi/2$'
            elif n == -1:
                return r'$-\pi/2$'
            else:
                return rf'${n}\pi/2$'
        else:
            return f'{x:.2f}'

    if axis == 'x':
        ax.xaxis.set_major_locator(ticker.MultipleLocator(np.pi/2))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(pi_formatter))
    else:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(np.pi/2))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(pi_formatter))


def create_figure(layout: str = 'single', **kwargs) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """
    Create figure with standard sizing.

    Parameters
    ----------
    layout : str
        Layout name from FIGSIZE dict, or tuple like '2x2', '3x3'
    **kwargs
        Additional arguments for plt.subplots

    Returns
    -------
    fig : Figure
        Matplotlib figure
    axes : Axes or ndarray of Axes
        Matplotlib axes
    """
    from .config import FIGSIZE

    if layout in FIGSIZE:
        figsize = FIGSIZE[layout]
        return plt.subplots(figsize=figsize, **kwargs)
    elif 'x' in str(layout):
        # Parse '2x3' format
        rows, cols = map(int, str(layout).split('x'))
        figsize = (4 * cols, 3 * rows)
        return plt.subplots(rows, cols, figsize=figsize, **kwargs)
    else:
        return plt.subplots(**kwargs)
