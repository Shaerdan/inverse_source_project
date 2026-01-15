"""
Test Configuration System for Inverse Source Localization
==========================================================

Provides a flexible JSON-based configuration system for testing different
source configurations, optimizer settings, and measurement parameters.

Usage:
    python cli.py compare --domains disk --preset easy_validation
    python cli.py compare --domains disk ellipse star --preset six_sources
    python cli.py compare --list-presets

The configuration file (test_configurations.json) contains:
- active_preset: Master switch for default preset
- presets: Named test configurations
- domain_defaults: Default sources for each domain
- expected_performance: Pass/fail thresholds
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path


# Default config file location (relative to src/)
DEFAULT_CONFIG_PATH = Path(__file__).parent / "test_configurations.json"


@dataclass
class SourceConfig:
    """Configuration for source generation."""
    n_sources: int = 4
    placement_method: str = "domain_default"  # angular_spread, random, explicit, domain_default
    depth_range: Tuple[float, float] = (0.15, 0.35)
    angular_range: Tuple[float, float] = (0, 360)  # degrees
    angular_jitter: float = 0.0
    intensity_method: str = "alternating"  # alternating, random, explicit
    intensity_magnitude: float = 1.0
    intensity_magnitude_range: Tuple[float, float] = (0.5, 2.0)
    explicit_sources: List[Dict] = field(default_factory=list)


@dataclass
class MeasurementConfig:
    """Configuration for measurement settings."""
    n_sensors: int = 100
    noise_level: float = 0.001
    noise_type: str = "gaussian"


@dataclass
class OptimizerConfig:
    """Configuration for optimizer settings."""
    # L-BFGS-B settings
    lbfgsb_n_restarts: int = 5
    lbfgsb_maxiter: int = 2000
    lbfgsb_ftol: float = 2.220446049250313e-09
    lbfgsb_gtol: float = 1e-05
    
    # Differential Evolution settings
    de_maxiter: int = 200
    de_tol: float = 1e-06
    de_atol: float = 0
    de_mutation: Tuple[float, float] = (0.5, 1.0)
    de_recombination: float = 0.7
    de_polish: bool = True
    de_strategy: str = "best1bin"


@dataclass
class SolverConfig:
    """Configuration for which solvers to run."""
    run_linear: bool = True
    run_nonlinear: bool = True
    nonlinear_methods: List[str] = field(default_factory=lambda: ["L-BFGS-B", "differential_evolution"])
    prefer_fem_for_polygon: bool = True


@dataclass
class TestPreset:
    """Complete test preset configuration."""
    name: str
    description: str
    sources: SourceConfig
    measurement: MeasurementConfig
    optimizer: OptimizerConfig
    solvers: SolverConfig
    seed: int = 42


class TestConfigManager:
    """
    Manager for test configurations.
    
    Loads presets from JSON file and provides methods to generate
    sources and optimizer settings for testing.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the configuration manager.
        
        Parameters
        ----------
        config_path : str, optional
            Path to configuration JSON file. Uses default if not provided.
        """
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config = None
        self.presets = {}
        self.domain_defaults = {}
        self.expected_performance = {}
        
        if self.config_path.exists():
            self.load_config()
        else:
            print(f"Config file not found: {self.config_path}")
            print("Using built-in defaults.")
            self._use_builtin_defaults()
    
    def load_config(self):
        """Load configuration from JSON file."""
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        self.presets = self.config.get('presets', {})
        self.domain_defaults = self.config.get('domain_defaults', {})
        self.expected_performance = self.config.get('expected_performance', {})
        self.active_preset = self.config.get('active_preset', 'default')
    
    def _use_builtin_defaults(self):
        """Use built-in defaults when no config file exists."""
        self.active_preset = 'default'
        self.presets = {'default': self._get_default_preset_dict()}
        self.domain_defaults = {}
        self.expected_performance = {}
    
    def _get_default_preset_dict(self) -> dict:
        """Get the default preset as a dictionary."""
        return {
            "description": "Default configuration",
            "sources": {
                "n_sources": 4,
                "placement": {"method": "domain_default"},
                "intensities": {"method": "domain_default"}
            },
            "measurement": {
                "n_sensors": 100,
                "noise_level": 0.001,
                "noise_type": "gaussian"
            },
            "solvers": {
                "run_linear": True,
                "run_nonlinear": True,
                "nonlinear_methods": ["L-BFGS-B", "differential_evolution"],
                "prefer_fem_for_polygon": True
            },
            "optimizer": {
                "L-BFGS-B": {
                    "n_restarts": 5,
                    "maxiter": 2000
                },
                "differential_evolution": {
                    "maxiter": 200,
                    "tol": 1e-06,
                    "polish": True
                }
            },
            "seed": 42
        }
    
    def list_presets(self) -> List[str]:
        """List available preset names."""
        return list(self.presets.keys())
    
    def get_preset(self, name: str = None) -> dict:
        """
        Get a preset configuration by name.
        
        Parameters
        ----------
        name : str, optional
            Preset name. Uses active_preset if not provided.
            
        Returns
        -------
        dict : Preset configuration
        """
        if name is None:
            name = self.active_preset
        
        if name not in self.presets:
            available = ', '.join(self.presets.keys())
            raise ValueError(f"Unknown preset '{name}'. Available: {available}")
        
        return self.presets[name]
    
    def get_source_config(self, preset_name: str = None) -> SourceConfig:
        """Extract SourceConfig from a preset."""
        preset = self.get_preset(preset_name)
        src = preset.get('sources', {})
        placement = src.get('placement', {})
        intensities = src.get('intensities', {})
        
        depth_range = placement.get('depth_range', [0.15, 0.35])
        angular_range = placement.get('angular_range', [0, 360])
        
        return SourceConfig(
            n_sources=src.get('n_sources', 4),
            placement_method=placement.get('method', 'domain_default'),
            depth_range=tuple(depth_range) if depth_range else (0.15, 0.35),
            angular_range=tuple(angular_range) if angular_range else (0, 360),
            angular_jitter=placement.get('angular_jitter', 0.0),
            intensity_method=intensities.get('method', 'alternating'),
            intensity_magnitude=intensities.get('magnitude', 1.0),
            intensity_magnitude_range=tuple(intensities.get('magnitude_range', [0.5, 2.0])),
            explicit_sources=src.get('explicit', []) or []
        )
    
    def get_measurement_config(self, preset_name: str = None) -> MeasurementConfig:
        """Extract MeasurementConfig from a preset."""
        preset = self.get_preset(preset_name)
        meas = preset.get('measurement', {})
        
        return MeasurementConfig(
            n_sensors=meas.get('n_sensors', 100),
            noise_level=meas.get('noise_level', 0.001),
            noise_type=meas.get('noise_type', 'gaussian')
        )
    
    def get_optimizer_config(self, preset_name: str = None) -> OptimizerConfig:
        """Extract OptimizerConfig from a preset."""
        preset = self.get_preset(preset_name)
        opt = preset.get('optimizer', {})
        lbfgsb = opt.get('L-BFGS-B', {})
        de = opt.get('differential_evolution', {})
        
        de_mutation = de.get('mutation', [0.5, 1.0])
        
        return OptimizerConfig(
            lbfgsb_n_restarts=lbfgsb.get('n_restarts', 5),
            lbfgsb_maxiter=lbfgsb.get('maxiter', 2000),
            lbfgsb_ftol=lbfgsb.get('ftol', 2.220446049250313e-09),
            lbfgsb_gtol=lbfgsb.get('gtol', 1e-05),
            de_maxiter=de.get('maxiter', 200),
            de_tol=de.get('tol', 1e-06),
            de_atol=de.get('atol', 0),
            de_mutation=tuple(de_mutation) if isinstance(de_mutation, list) else (0.5, 1.0),
            de_recombination=de.get('recombination', 0.7),
            de_polish=de.get('polish', True),
            de_strategy=de.get('strategy', 'best1bin')
        )
    
    def get_solver_config(self, preset_name: str = None) -> SolverConfig:
        """Extract SolverConfig from a preset."""
        preset = self.get_preset(preset_name)
        sol = preset.get('solvers', {})
        
        return SolverConfig(
            run_linear=sol.get('run_linear', True),
            run_nonlinear=sol.get('run_nonlinear', True),
            nonlinear_methods=sol.get('nonlinear_methods', ['L-BFGS-B', 'differential_evolution']),
            prefer_fem_for_polygon=sol.get('prefer_fem_for_polygon', True)
        )
    
    def get_seed(self, preset_name: str = None) -> int:
        """Get random seed from preset."""
        preset = self.get_preset(preset_name)
        return preset.get('seed', 42)
    
    def get_expected_performance(self, preset_name: str = None) -> Dict[str, float]:
        """Get expected performance thresholds for a preset."""
        if preset_name is None:
            preset_name = self.active_preset
        return self.expected_performance.get(preset_name, {})
    
    def get_domain_default_sources(self, domain: str) -> List[Tuple[Tuple[float, float], float]]:
        """Get default sources for a domain from config."""
        if domain not in self.domain_defaults:
            return None
        
        domain_config = self.domain_defaults[domain]
        sources_list = domain_config.get('sources', [])
        
        sources = []
        for src in sources_list:
            pos = tuple(src['position'])
            intensity = src['intensity']
            sources.append((pos, intensity))
        
        return sources
    
    def print_preset_summary(self, preset_name: str = None):
        """Print a summary of a preset configuration."""
        preset = self.get_preset(preset_name)
        name = preset_name or self.active_preset
        
        print(f"\nPreset: {name}")
        print(f"  Description: {preset.get('description', 'N/A')}")
        
        src = preset.get('sources', {})
        print(f"  Sources: {src.get('n_sources', 4)}")
        placement = src.get('placement', {})
        print(f"    Placement: {placement.get('method', 'domain_default')}")
        if placement.get('depth_range'):
            print(f"    Depth range: {placement['depth_range']}")
        if placement.get('angular_range'):
            print(f"    Angular range: {placement['angular_range']}°")
        
        intensities = src.get('intensities', {})
        print(f"    Intensities: {intensities.get('method', 'alternating')}")
        
        meas = preset.get('measurement', {})
        print(f"  Measurement:")
        print(f"    Sensors: {meas.get('n_sensors', 100)}")
        print(f"    Noise: {meas.get('noise_level', 0.001)}")
        
        sol = preset.get('solvers', {})
        print(f"  Solvers:")
        print(f"    Linear: {sol.get('run_linear', True)}")
        print(f"    Nonlinear: {sol.get('run_nonlinear', True)}")
        if sol.get('run_nonlinear', True):
            print(f"    Methods: {sol.get('nonlinear_methods', ['L-BFGS-B', 'differential_evolution'])}")
        
        print(f"  Seed: {preset.get('seed', 42)}")


def generate_sources_from_config(
    domain_type: str,
    source_config: SourceConfig,
    domain_params: dict = None,
    seed: int = 42,
    domain_defaults_func=None
) -> List[Tuple[Tuple[float, float], float]]:
    """
    Generate sources based on configuration.
    
    Parameters
    ----------
    domain_type : str
        Domain type ('disk', 'ellipse', 'star', 'square', 'brain')
    source_config : SourceConfig
        Source configuration from preset
    domain_params : dict, optional
        Domain-specific parameters
    seed : int
        Random seed
    domain_defaults_func : callable, optional
        Function to get domain defaults (create_domain_sources)
        
    Returns
    -------
    List of ((x, y), intensity) tuples
    """
    np.random.seed(seed)
    
    n = source_config.n_sources
    method = source_config.placement_method
    
    # Handle domain_default - use existing create_domain_sources behavior
    if method == 'domain_default':
        if domain_defaults_func:
            return domain_defaults_func(domain_type, domain_params, n_sources=n)
        else:
            # Fallback to angular_spread with default depths
            method = 'angular_spread'
    
    # Handle explicit sources
    if method == 'explicit' and source_config.explicit_sources:
        sources = []
        for src in source_config.explicit_sources:
            pos = tuple(src['position'])
            intensity = src['intensity']
            sources.append((pos, intensity))
        return sources
    
    # Generate positions using angular_spread or random
    positions = _generate_positions(
        domain_type=domain_type,
        n_sources=n,
        depth_range=source_config.depth_range,
        angular_range=source_config.angular_range,
        angular_jitter=source_config.angular_jitter,
        method=method,
        domain_params=domain_params,
        seed=seed
    )
    
    # Generate intensities
    intensities = _generate_intensities(
        n_sources=n,
        method=source_config.intensity_method,
        magnitude=source_config.intensity_magnitude,
        magnitude_range=source_config.intensity_magnitude_range,
        seed=seed
    )
    
    # Combine
    sources = [(pos, intensity) for pos, intensity in zip(positions, intensities)]
    
    return sources


def _generate_positions(
    domain_type: str,
    n_sources: int,
    depth_range: Tuple[float, float],
    angular_range: Tuple[float, float],
    angular_jitter: float,
    method: str,
    domain_params: dict = None,
    seed: int = 42
) -> List[Tuple[float, float]]:
    """Generate source positions based on configuration."""
    np.random.seed(seed)
    
    # Convert angular range from degrees to radians
    ang_min = np.deg2rad(angular_range[0])
    ang_max = np.deg2rad(angular_range[1])
    
    # Mean depth
    mean_depth = (depth_range[0] + depth_range[1]) / 2
    
    positions = []
    
    if domain_type == 'disk':
        radius = 1.0
        # depth is fraction from boundary, so r = radius * (1 - depth)
        r_min = radius * (1 - depth_range[1])
        r_max = radius * (1 - depth_range[0])
        r_mean = (r_min + r_max) / 2
        
        if method == 'angular_spread':
            angles = np.linspace(ang_min, ang_max, n_sources, endpoint=False)
            for i, theta in enumerate(angles):
                theta += np.random.uniform(-angular_jitter, angular_jitter)
                r = np.random.uniform(r_min, r_max)
                positions.append((r * np.cos(theta), r * np.sin(theta)))
        else:  # random
            for _ in range(n_sources):
                theta = np.random.uniform(ang_min, ang_max)
                r = np.random.uniform(r_min, r_max)
                positions.append((r * np.cos(theta), r * np.sin(theta)))
    
    elif domain_type == 'ellipse':
        a = domain_params.get('a', 2.0) if domain_params else 2.0
        b = domain_params.get('b', 1.0) if domain_params else 1.0
        
        # Scale factors for ellipse
        scale_min = 1 - depth_range[1]
        scale_max = 1 - depth_range[0]
        
        if method == 'angular_spread':
            angles = np.linspace(ang_min, ang_max, n_sources, endpoint=False)
            for theta in angles:
                theta += np.random.uniform(-angular_jitter, angular_jitter)
                scale = np.random.uniform(scale_min, scale_max)
                x = scale * a * np.cos(theta)
                y = scale * b * np.sin(theta)
                positions.append((x, y))
        else:
            for _ in range(n_sources):
                theta = np.random.uniform(ang_min, ang_max)
                scale = np.random.uniform(scale_min, scale_max)
                x = scale * a * np.cos(theta)
                y = scale * b * np.sin(theta)
                positions.append((x, y))
    
    elif domain_type == 'star':
        n_petals = domain_params.get('n_petals', 5) if domain_params else 5
        amplitude = domain_params.get('amplitude', 0.3) if domain_params else 0.3
        
        if method == 'angular_spread':
            angles = np.linspace(ang_min, ang_max, n_sources, endpoint=False)
            for theta in angles:
                theta += np.random.uniform(-angular_jitter, angular_jitter)
                r_boundary = 1.0 + amplitude * np.cos(n_petals * theta)
                depth = np.random.uniform(depth_range[0], depth_range[1])
                r = r_boundary - depth
                r = max(0.2, r)  # Don't go too close to center
                positions.append((r * np.cos(theta), r * np.sin(theta)))
        else:
            for _ in range(n_sources):
                theta = np.random.uniform(ang_min, ang_max)
                r_boundary = 1.0 + amplitude * np.cos(n_petals * theta)
                depth = np.random.uniform(depth_range[0], depth_range[1])
                r = r_boundary - depth
                r = max(0.2, r)
                positions.append((r * np.cos(theta), r * np.sin(theta)))
    
    elif domain_type == 'square':
        # Square [-1, 1]^2
        offset = 1.0 - mean_depth
        
        if method == 'angular_spread':
            angles = np.linspace(ang_min, ang_max, n_sources, endpoint=False)
            for theta in angles:
                theta += np.random.uniform(-angular_jitter, angular_jitter)
                depth = np.random.uniform(depth_range[0], depth_range[1])
                r = 1.0 - depth
                # Project onto square boundary
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                # Clamp to square interior
                x = np.clip(x, -0.95, 0.95)
                y = np.clip(y, -0.95, 0.95)
                positions.append((x, y))
        else:
            for _ in range(n_sources):
                depth = np.random.uniform(depth_range[0], depth_range[1])
                margin = depth
                x = np.random.uniform(-1 + margin, 1 - margin)
                y = np.random.uniform(-1 + margin, 1 - margin)
                positions.append((x, y))
    
    elif domain_type == 'brain':
        # Brain domain: x in [-1.1, 1.1], y in [-0.6, 0.7]
        # Approximate as scaled ellipse for positioning
        x_scale = 1.0
        y_scale = 0.65
        
        if method == 'angular_spread':
            angles = np.linspace(ang_min, ang_max, n_sources, endpoint=False)
            for theta in angles:
                theta += np.random.uniform(-angular_jitter, angular_jitter)
                depth = np.random.uniform(depth_range[0], depth_range[1])
                scale = 1.0 - depth
                x = scale * x_scale * np.cos(theta)
                y = scale * y_scale * np.sin(theta) + 0.05  # Slight offset for brain shape
                positions.append((x, y))
        else:
            for _ in range(n_sources):
                depth = np.random.uniform(depth_range[0], depth_range[1])
                scale = 1.0 - depth
                theta = np.random.uniform(ang_min, ang_max)
                x = scale * x_scale * np.cos(theta)
                y = scale * y_scale * np.sin(theta) + 0.05
                positions.append((x, y))
    
    else:
        raise ValueError(f"Unknown domain type: {domain_type}")
    
    return positions


def _generate_intensities(
    n_sources: int,
    method: str,
    magnitude: float,
    magnitude_range: Tuple[float, float],
    seed: int = 42
) -> List[float]:
    """Generate source intensities based on configuration."""
    np.random.seed(seed + 1000)  # Different seed for intensities
    
    if method == 'alternating':
        intensities = [magnitude if i % 2 == 0 else -magnitude for i in range(n_sources)]
    
    elif method == 'random':
        # Generate random magnitudes
        mags = np.random.uniform(magnitude_range[0], magnitude_range[1], n_sources)
        signs = np.random.choice([-1, 1], n_sources)
        intensities = list(mags * signs)
    
    elif method == 'explicit':
        # Should be handled at higher level
        intensities = [1.0 if i % 2 == 0 else -1.0 for i in range(n_sources)]
    
    else:
        # Default to alternating
        intensities = [magnitude if i % 2 == 0 else -magnitude for i in range(n_sources)]
    
    # Enforce zero-sum constraint
    total = sum(intensities)
    if abs(total) > 1e-10:
        # Adjust last intensity to make sum zero
        intensities[-1] -= total
    
    return intensities


def list_presets(config_path: str = None) -> None:
    """Print list of available presets."""
    manager = TestConfigManager(config_path)
    
    print("\nAvailable Test Presets:")
    print("=" * 70)
    
    for name in sorted(manager.list_presets()):
        preset = manager.get_preset(name)
        desc = preset.get('description', 'No description')
        n_sources = preset.get('sources', {}).get('n_sources', 4)
        noise = preset.get('measurement', {}).get('noise_level', 0.001)
        
        print(f"\n  {name}:")
        print(f"    {desc}")
        print(f"    Sources: {n_sources}, Noise: {noise}")
    
    print("\n" + "=" * 70)
    print(f"Active preset: {manager.active_preset}")
    print("\nUsage: python cli.py compare --domains disk --preset <preset_name>")


# Convenience function for CLI
def get_config_manager(config_path: str = None) -> TestConfigManager:
    """Get a TestConfigManager instance."""
    return TestConfigManager(config_path)


if __name__ == "__main__":
    # Demo usage
    list_presets()
    
    print("\n\nExample: Generating sources with 'easy_validation' preset for disk domain")
    manager = TestConfigManager()
    
    if 'easy_validation' in manager.list_presets():
        manager.print_preset_summary('easy_validation')
        
        src_config = manager.get_source_config('easy_validation')
        sources = generate_sources_from_config('disk', src_config, seed=42)
        
        print("\nGenerated sources:")
        for i, ((x, y), intensity) in enumerate(sources):
            print(f"  {i+1}: ({x:+.3f}, {y:+.3f}), intensity = {intensity:+.3f}")
