"""
Visualization Configuration
===========================

Centralized styling configuration for the inverse source visualization toolbox.
Single source of truth for colors, markers, figure sizes, and matplotlib settings.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Union

# =============================================================================
# COLOR SCHEMES
# =============================================================================

COLORS = {
    # Sources
    'source_positive': '#d62728',      # Red for q > 0
    'source_negative': '#1f77b4',      # Blue for q < 0

    # Domain
    'boundary': '#2c3e50',             # Dark gray
    'domain_fill': '#ecf0f1',          # Light fill

    # Data comparison
    'measured': '#3498db',             # Blue
    'recovered': '#e74c3c',            # Red
    'residual': '#f39c12',             # Orange

    # Optimization
    'trajectory': '#8e44ad',           # Purple
    'initial': '#27ae60',              # Green
    'final': '#c0392b',                # Dark red

    # Sensors
    'sensor': '#7f8c8d',               # Gray

    # Additional useful colors
    'grid': '#bdc3c7',                 # Light gray for grids
    'text': '#2c3e50',                 # Dark text
    'highlight': '#f1c40f',            # Yellow highlight
    'success': '#27ae60',              # Green for success
    'warning': '#f39c12',              # Orange for warning
    'error': '#e74c3c',                # Red for error
}

# =============================================================================
# MARKER STYLES
# =============================================================================

MARKERS = {
    'source_true': {
        'marker': 'o',
        'markersize': 10,
        'markeredgewidth': 2,
        'fillstyle': 'full',
    },
    'source_recovered': {
        'marker': 's',
        'markersize': 10,
        'fillstyle': 'none',
        'markeredgewidth': 2,
    },
    'sensor': {
        'marker': '.',
        'markersize': 4,
    },
    'initial': {
        'marker': 'o',
        'markersize': 8,
        'fillstyle': 'full',
    },
    'final': {
        'marker': 'o',
        'markersize': 10,
        'fillstyle': 'full',
    },
    'peak': {
        'marker': '*',
        'markersize': 12,
        'fillstyle': 'full',
    },
}

# =============================================================================
# FIGURE SIZES
# =============================================================================

FIGSIZE = {
    'single': (8, 6),
    'wide': (12, 5),
    'square': (8, 8),
    'dashboard': (16, 12),
    'comparison': (14, 10),
    'animation': (10, 8),
    'small': (6, 4),
    'tall': (8, 10),
}

# =============================================================================
# COLORMAPS
# =============================================================================

COLORMAPS = {
    'solution': 'RdBu_r',      # Diverging for potential (symmetric around 0)
    'barrier': 'YlOrRd',       # Sequential for barrier values
    'error': 'hot',            # Sequential for errors
    'greens': 'PiYG',          # Diverging for Green's function
    'intensity': 'RdBu_r',     # Diverging for source intensities
    'jacobian': 'viridis',     # Sequential for Jacobian magnitude
    'convergence': 'plasma',   # Sequential for iteration coloring
}

# =============================================================================
# LINE STYLES
# =============================================================================

LINESTYLES = {
    'measured': {'linestyle': '-', 'linewidth': 2},
    'recovered': {'linestyle': '--', 'linewidth': 2},
    'reference': {'linestyle': ':', 'linewidth': 1.5},
    'trajectory': {'linestyle': '-', 'linewidth': 1},
    'boundary': {'linestyle': '-', 'linewidth': 2},
    'grid': {'linestyle': '--', 'linewidth': 0.5, 'alpha': 0.5},
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_source_color(intensity: Union[float, int]) -> str:
    """
    Return appropriate color based on source intensity sign.

    Parameters
    ----------
    intensity : float
        Source intensity value

    Returns
    -------
    str
        Hex color code (red for positive, blue for negative)
    """
    return COLORS['source_positive'] if intensity > 0 else COLORS['source_negative']


def get_source_marker(style: str = 'true') -> dict:
    """
    Get marker style dictionary for sources.

    Parameters
    ----------
    style : str
        'true' for filled circles, 'recovered' for hollow squares

    Returns
    -------
    dict
        Marker style parameters for matplotlib
    """
    if style == 'true':
        return MARKERS['source_true'].copy()
    elif style == 'recovered':
        return MARKERS['source_recovered'].copy()
    else:
        return MARKERS.get(style, MARKERS['source_true']).copy()


def get_trajectory_color(iteration: int, total_iterations: int) -> str:
    """
    Get color for trajectory point based on iteration progress.

    Parameters
    ----------
    iteration : int
        Current iteration number
    total_iterations : int
        Total number of iterations

    Returns
    -------
    str
        Hex color code (light early, dark late)
    """
    cmap = plt.cm.get_cmap(COLORMAPS['convergence'])
    return mpl.colors.to_hex(cmap(iteration / max(total_iterations, 1)))


def apply_style():
    """
    Apply consistent matplotlib style settings for all visualizations.

    Call this at module import or before creating figures.
    """
    # Use a clean style base
    plt.style.use('seaborn-v0_8-whitegrid')

    # Customize rcParams
    plt.rcParams.update({
        # Figure
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,

        # Axes
        'axes.facecolor': 'white',
        'axes.edgecolor': COLORS['text'],
        'axes.linewidth': 1.0,
        'axes.grid': True,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 10,
        'axes.labelweight': 'normal',
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Grid
        'grid.color': COLORS['grid'],
        'grid.linewidth': 0.5,
        'grid.alpha': 0.7,

        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.facecolor': 'white',
        'legend.edgecolor': COLORS['grid'],
        'legend.fontsize': 9,

        # Lines
        'lines.linewidth': 2.0,
        'lines.markersize': 6,

        # Font
        'font.family': 'sans-serif',
        'font.size': 10,

        # Ticks
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
    })


def reset_style():
    """Reset matplotlib to default style."""
    plt.rcdefaults()


# =============================================================================
# STYLE CONTEXT MANAGERS
# =============================================================================

class viz_style:
    """
    Context manager for temporarily applying visualization style.

    Usage
    -----
    with viz_style():
        fig, ax = plt.subplots()
        # plotting code
    """

    def __enter__(self):
        self._original_rcparams = plt.rcParams.copy()
        apply_style()
        return self

    def __exit__(self, *args):
        plt.rcParams.update(self._original_rcparams)


# =============================================================================
# CONSTANTS FOR SPECIFIC VISUALIZATIONS
# =============================================================================

# Barrier visualization
BARRIER_LEVELS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
BARRIER_MU_DEFAULT = 1e-4
BARRIER_R_MAX_DEFAULT = 0.95

# Convergence plot
CONVERGENCE_LOG_THRESHOLD = 1e-12  # Values below this are clipped

# Animation
ANIMATION_INTERVAL_MS = 100
ANIMATION_TRAIL_LENGTH = 20

# Dashboard
DASHBOARD_GRID_SPEC = {
    'diagnostic': (3, 3),
    'comparison': (3, 4),
    'domain': (3, 4),
}
