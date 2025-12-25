"""
Configuration System for Inverse Source Localization
=====================================================

JSON-based configuration management for flexible parameter control.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path


@dataclass
class ForwardConfig:
    """Configuration for forward solver."""
    method: str = "bem"  # "bem" or "fem"
    n_boundary_points: int = 100
    domain_type: str = "disk"  # "disk", "ellipse", "star"
    domain_params: Dict[str, float] = field(default_factory=lambda: {"radius": 1.0})


@dataclass
class InverseConfig:
    """Configuration for inverse solver."""
    method: str = "linear"  # "linear" or "nonlinear"
    regularization: str = "l1"  # "l1", "l2", "tv"
    alpha: float = 1e-4
    n_sources: int = 4  # For nonlinear method
    optimizer: str = "L-BFGS-B"  # For nonlinear method
    max_iter: int = 100
    tolerance: float = 1e-6


@dataclass
class GridConfig:
    """Configuration for source grid (linear method)."""
    n_radial: int = 10
    n_angular: int = 20
    r_min: float = 0.1
    r_max: float = 0.9


@dataclass
class TVConfig:
    """Configuration for Total Variation regularization."""
    algorithm: str = "chambolle_pock"  # "chambolle_pock" or "admm"
    tau: float = 0.1
    sigma: float = 0.1
    theta: float = 1.0
    max_iter: int = 500
    tol: float = 1e-5
    # ADMM parameters
    rho: float = 1.0


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    live_plot: bool = False
    plot_interval: int = 10
    save_figures: bool = True
    figure_dir: str = "results"
    dpi: int = 150


@dataclass
class Config:
    """Master configuration container."""
    forward: ForwardConfig = field(default_factory=ForwardConfig)
    inverse: InverseConfig = field(default_factory=InverseConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    tv: TVConfig = field(default_factory=TVConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'forward': asdict(self.forward),
            'inverse': asdict(self.inverse),
            'grid': asdict(self.grid),
            'tv': asdict(self.tv),
            'visualization': asdict(self.visualization),
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Config':
        """Create from dictionary."""
        return cls(
            forward=ForwardConfig(**d.get('forward', {})),
            inverse=InverseConfig(**d.get('inverse', {})),
            grid=GridConfig(**d.get('grid', {})),
            tv=TVConfig(**d.get('tv', {})),
            visualization=VisualizationConfig(**d.get('visualization', {})),
        )
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            d = json.load(f)
        print(f"Configuration loaded from {path}")
        return cls.from_dict(d)


def create_default_config(path: str = "config.json") -> Config:
    """Create and save default configuration file."""
    config = Config()
    config.save(path)
    return config


def get_config(path: Optional[str] = None) -> Config:
    """
    Get configuration, loading from file if provided.
    
    Parameters
    ----------
    path : str, optional
        Path to config file. If None, returns default config.
        
    Returns
    -------
    config : Config
    """
    if path is None:
        return Config()
    
    path = Path(path)
    if path.exists():
        return Config.load(str(path))
    else:
        print(f"Config file {path} not found, using defaults")
        return Config()


# Pre-defined configuration templates
TEMPLATES = {
    'default': Config(),
    
    'high_resolution': Config(
        forward=ForwardConfig(n_boundary_points=200),
        grid=GridConfig(n_radial=20, n_angular=40),
    ),
    
    'fast': Config(
        forward=ForwardConfig(n_boundary_points=50),
        grid=GridConfig(n_radial=5, n_angular=10),
        inverse=InverseConfig(max_iter=50),
    ),
    
    'tv_chambolle_pock': Config(
        inverse=InverseConfig(regularization='tv'),
        tv=TVConfig(algorithm='chambolle_pock', max_iter=1000),
    ),
    
    'tv_admm': Config(
        inverse=InverseConfig(regularization='tv'),
        tv=TVConfig(algorithm='admm', rho=1.0, max_iter=500),
    ),
    
    'nonlinear': Config(
        inverse=InverseConfig(
            method='nonlinear',
            n_sources=4,
            optimizer='L-BFGS-B',
            max_iter=200,
        ),
    ),
    
    'ellipse': Config(
        forward=ForwardConfig(
            domain_type='ellipse',
            domain_params={'a': 2.0, 'b': 1.0},
        ),
    ),
}


def get_template(name: str) -> Config:
    """Get a pre-defined configuration template."""
    if name not in TEMPLATES:
        available = ', '.join(TEMPLATES.keys())
        raise ValueError(f"Unknown template '{name}'. Available: {available}")
    return TEMPLATES[name]


if __name__ == "__main__":
    # Create default config file
    config = create_default_config("config.json")
    print("\nDefault configuration:")
    print(json.dumps(config.to_dict(), indent=2))
