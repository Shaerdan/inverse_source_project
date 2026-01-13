"""
Experiment Tracking Module for Inverse Source Localization
============================================================

Provides comprehensive experiment tracking:
1. Organized output folders with hash + timestamp
2. SQLite database for all experiment metadata
3. Non-blocking plot generation (saves to file, no GUI)
4. Full reproducibility tracking

Usage:
    from inverse_source.experiment_tracker import ExperimentTracker
    
    with ExperimentTracker(base_dir='results') as tracker:
        tracker.log_params(domain='disk', noise=0.001, ...)
        tracker.log_metrics(localization=0.85, rmse=0.12, ...)
        tracker.save_figure(fig, 'comparison.png')
        tracker.log_artifact('mesh.msh', artifact_type='mesh')

Author: Claude (Anthropic)
Date: January 2026
"""

import os
import sys
import json
import sqlite3
import hashlib
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
import numpy as np

# Disable interactive plotting globally
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt


def get_timestamp() -> str:
    """Get formatted timestamp: YYYY-MM-DD_THHhMMmin"""
    now = datetime.now()
    return now.strftime("%Y-%m-%d_T%Hh%Mmin")


def get_short_hash(length: int = 8) -> str:
    """Generate random hash for unique identification."""
    import secrets
    return secrets.token_hex(length // 2)


def get_deterministic_hash(params: dict, length: int = 8) -> str:
    """Generate deterministic hash from parameters for reproducibility."""
    param_str = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(param_str.encode()).hexdigest()[:length]


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    # Identifiers
    experiment_id: str = ""
    timestamp: str = ""
    hash_id: str = ""
    
    # Domain configuration
    domain_type: str = ""
    domain_params: dict = field(default_factory=dict)
    
    # Mesh configuration
    forward_mesh_resolution: float = 0.0
    source_grid_resolution: float = 0.0
    n_boundary_points: int = 0
    n_interior_points: int = 0
    n_elements: int = 0
    
    # Problem configuration
    n_sources: int = 0
    sources_true: list = field(default_factory=list)
    noise_level: float = 0.0
    seed: int = 0
    
    # Solver configuration
    solver_type: str = ""
    method: str = ""  # l1, l2, tv, nonlinear
    alpha: float = 0.0
    alpha_selection: str = ""  # 'fixed', 'lcurve', 'calibrated'
    
    # Optimization parameters
    max_iterations: int = 0
    tolerance: float = 0.0
    optimizer: str = ""
    n_restarts: int = 0
    
    # Additional solver params
    solver_params: dict = field(default_factory=dict)
    
    # Environment
    python_version: str = ""
    numpy_version: str = ""
    scipy_version: str = ""
    platform_info: str = ""
    
    # Paths
    output_dir: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentMetrics:
    """Metrics from an experiment run."""
    # Position metrics
    position_rmse: float = 0.0
    position_mae: float = 0.0
    
    # Intensity metrics
    intensity_rmse: float = 0.0
    intensity_mae: float = 0.0
    intensity_correlation: float = 0.0
    
    # Boundary fit
    boundary_residual: float = 0.0
    boundary_relative_error: float = 0.0
    
    # Source recovery quality
    localization_score: float = 0.0
    sparsity_ratio: float = 0.0
    n_peaks_detected: int = 0
    n_sources_recovered: int = 0
    
    # Computational
    time_seconds: float = 0.0
    iterations: int = 0
    converged: bool = False
    
    # L-curve info (if applicable)
    lcurve_residual: float = 0.0
    lcurve_regularizer: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentDatabase:
    """SQLite database for experiment tracking."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        """Initialize database with schema."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Main experiments table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                hash_id TEXT NOT NULL,
                
                -- Domain
                domain_type TEXT,
                domain_params TEXT,  -- JSON
                
                -- Mesh
                forward_mesh_resolution REAL,
                source_grid_resolution REAL,
                n_boundary_points INTEGER,
                n_interior_points INTEGER,
                n_elements INTEGER,
                
                -- Problem
                n_sources INTEGER,
                sources_true TEXT,  -- JSON
                noise_level REAL,
                seed INTEGER,
                
                -- Solver
                solver_type TEXT,
                method TEXT,
                alpha REAL,
                alpha_selection TEXT,
                
                -- Optimization
                max_iterations INTEGER,
                tolerance REAL,
                optimizer TEXT,
                n_restarts INTEGER,
                solver_params TEXT,  -- JSON
                
                -- Environment
                python_version TEXT,
                numpy_version TEXT,
                scipy_version TEXT,
                platform_info TEXT,
                
                -- Paths
                output_dir TEXT,
                
                -- Timestamps
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Metrics table (one-to-one with experiments)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                
                -- Position metrics
                position_rmse REAL,
                position_mae REAL,
                
                -- Intensity metrics
                intensity_rmse REAL,
                intensity_mae REAL,
                intensity_correlation REAL,
                
                -- Boundary fit
                boundary_residual REAL,
                boundary_relative_error REAL,
                
                -- Source recovery
                localization_score REAL,
                sparsity_ratio REAL,
                n_peaks_detected INTEGER,
                n_sources_recovered INTEGER,
                
                -- Computational
                time_seconds REAL,
                iterations INTEGER,
                converged INTEGER,
                
                -- L-curve
                lcurve_residual REAL,
                lcurve_regularizer REAL,
                
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)
        
        # Artifacts table (one-to-many)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                artifact_type TEXT,  -- 'figure', 'mesh', 'solution', 'config', etc.
                filename TEXT,
                filepath TEXT,
                description TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)
        
        # Calibration runs table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS calibrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                calibration_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                
                domain_type TEXT,
                domain_params TEXT,  -- JSON
                
                forward_mesh_resolution REAL,
                source_grid_resolution REAL,
                
                alpha_l1 REAL,
                alpha_l2 REAL,
                alpha_tv REAL,
                
                forward_convergence TEXT,  -- JSON
                source_convergence TEXT,   -- JSON
                lcurve_l1 TEXT,  -- JSON
                lcurve_l2 TEXT,  -- JSON
                lcurve_tv TEXT,  -- JSON
                
                calibration_time_seconds REAL,
                config_path TEXT,
                
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indices for faster queries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_domain ON experiments(domain_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_method ON experiments(method)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_timestamp ON experiments(timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cal_domain ON calibrations(domain_type)")
        
        self.conn.commit()
    
    def insert_experiment(self, config: ExperimentConfig) -> int:
        """Insert experiment configuration."""
        cursor = self.conn.execute("""
            INSERT INTO experiments (
                experiment_id, timestamp, hash_id,
                domain_type, domain_params,
                forward_mesh_resolution, source_grid_resolution,
                n_boundary_points, n_interior_points, n_elements,
                n_sources, sources_true, noise_level, seed,
                solver_type, method, alpha, alpha_selection,
                max_iterations, tolerance, optimizer, n_restarts, solver_params,
                python_version, numpy_version, scipy_version, platform_info,
                output_dir
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            config.experiment_id, config.timestamp, config.hash_id,
            config.domain_type, json.dumps(config.domain_params),
            config.forward_mesh_resolution, config.source_grid_resolution,
            config.n_boundary_points, config.n_interior_points, config.n_elements,
            config.n_sources, json.dumps(config.sources_true), config.noise_level, config.seed,
            config.solver_type, config.method, config.alpha, config.alpha_selection,
            config.max_iterations, config.tolerance, config.optimizer, config.n_restarts,
            json.dumps(config.solver_params),
            config.python_version, config.numpy_version, config.scipy_version, config.platform_info,
            config.output_dir
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_metrics(self, experiment_id: str, metrics: ExperimentMetrics) -> int:
        """Insert experiment metrics."""
        # Sanitize values - ensure all are scalars, not lists/tuples/arrays
        def to_scalar(val, default=0):
            """Recursively unwrap containers and convert to scalar."""
            if val is None:
                return default
            # Unwrap containers recursively
            while isinstance(val, (list, tuple)):
                if len(val) == 0:
                    return default
                val = val[0]
            # Handle numpy arrays
            if isinstance(val, np.ndarray):
                if val.size == 0:
                    return default
                val = val.flat[0]
            # Convert to Python float/int
            try:
                return float(val)
            except (TypeError, ValueError):
                return default
        
        def to_int(val, default=0):
            """Convert to int, handling edge cases."""
            scalar = to_scalar(val, default)
            try:
                return int(scalar)
            except (TypeError, ValueError):
                return default
        
        cursor = self.conn.execute("""
            INSERT INTO metrics (
                experiment_id,
                position_rmse, position_mae,
                intensity_rmse, intensity_mae, intensity_correlation,
                boundary_residual, boundary_relative_error,
                localization_score, sparsity_ratio, n_peaks_detected, n_sources_recovered,
                time_seconds, iterations, converged,
                lcurve_residual, lcurve_regularizer
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id,
            to_scalar(metrics.position_rmse), to_scalar(metrics.position_mae),
            to_scalar(metrics.intensity_rmse), to_scalar(metrics.intensity_mae), 
            to_scalar(metrics.intensity_correlation),
            to_scalar(metrics.boundary_residual), to_scalar(metrics.boundary_relative_error),
            to_scalar(metrics.localization_score), to_scalar(metrics.sparsity_ratio),
            to_int(metrics.n_peaks_detected), to_int(metrics.n_sources_recovered),
            to_scalar(metrics.time_seconds), to_int(metrics.iterations), 
            to_int(metrics.converged),
            to_scalar(metrics.lcurve_residual), to_scalar(metrics.lcurve_regularizer)
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_artifact(self, experiment_id: str, artifact_type: str,
                        filename: str, filepath: str, description: str = "") -> int:
        """Insert artifact reference."""
        cursor = self.conn.execute("""
            INSERT INTO artifacts (experiment_id, artifact_type, filename, filepath, description)
            VALUES (?, ?, ?, ?, ?)
        """, (experiment_id, artifact_type, filename, filepath, description))
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_calibration(self, cal_data: dict) -> int:
        """Insert calibration run."""
        cursor = self.conn.execute("""
            INSERT INTO calibrations (
                calibration_id, timestamp, domain_type, domain_params,
                forward_mesh_resolution, source_grid_resolution,
                alpha_l1, alpha_l2, alpha_tv,
                forward_convergence, source_convergence,
                lcurve_l1, lcurve_l2, lcurve_tv,
                calibration_time_seconds, config_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cal_data.get('calibration_id', ''),
            cal_data.get('timestamp', ''),
            cal_data.get('domain_type', ''),
            json.dumps(cal_data.get('domain_params', {})),
            cal_data.get('forward_mesh_resolution', 0),
            cal_data.get('source_grid_resolution', 0),
            cal_data.get('alpha_l1', 0),
            cal_data.get('alpha_l2', 0),
            cal_data.get('alpha_tv', 0),
            json.dumps(cal_data.get('forward_convergence', {})),
            json.dumps(cal_data.get('source_convergence', {})),
            json.dumps(cal_data.get('lcurve_l1', {})),
            json.dumps(cal_data.get('lcurve_l2', {})),
            json.dumps(cal_data.get('lcurve_tv', {})),
            cal_data.get('calibration_time_seconds', 0),
            cal_data.get('config_path', '')
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_experiment(self, experiment_id: str) -> Optional[dict]:
        """Retrieve experiment by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_metrics(self, experiment_id: str) -> Optional[dict]:
        """Retrieve metrics for experiment."""
        cursor = self.conn.execute(
            "SELECT * FROM metrics WHERE experiment_id = ?", (experiment_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_artifacts(self, experiment_id: str) -> List[dict]:
        """Retrieve all artifacts for experiment."""
        cursor = self.conn.execute(
            "SELECT * FROM artifacts WHERE experiment_id = ?", (experiment_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    def query_experiments(self, domain_type: str = None, method: str = None,
                          limit: int = 100) -> List[dict]:
        """Query experiments with filters."""
        query = "SELECT * FROM experiments WHERE 1=1"
        params = []
        
        if domain_type:
            query += " AND domain_type = ?"
            params.append(domain_type)
        if method:
            query += " AND method = ?"
            params.append(method)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


class ExperimentTracker:
    """
    Main experiment tracking class.
    
    Usage:
        with ExperimentTracker(base_dir='results') as tracker:
            tracker.log_params(domain='disk', noise=0.001)
            # ... run experiment ...
            tracker.log_metrics(localization=0.85)
            tracker.save_figure(fig, 'result.png')
    """
    
    def __init__(self, base_dir: str = 'results', 
                 experiment_name: str = None,
                 use_timestamp: bool = True,
                 use_hash: bool = True):
        """
        Initialize experiment tracker.
        
        Parameters
        ----------
        base_dir : str
            Base directory for all outputs
        experiment_name : str
            Optional name prefix for experiment folder
        use_timestamp : bool
            Include timestamp in folder name
        use_hash : bool
            Include random hash in folder name
        """
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name or "exp"
        self.use_timestamp = use_timestamp
        self.use_hash = use_hash
        
        # Generate identifiers
        self.timestamp = get_timestamp()
        self.hash_id = get_short_hash()
        self.experiment_id = self._create_experiment_id()
        
        # Create output directory
        self.output_dir = self._create_output_dir()
        
        # Initialize database (shared across all experiments)
        self.db_path = self.base_dir / 'experiments.db'
        self.db = ExperimentDatabase(str(self.db_path))
        
        # Config and metrics
        self.config = ExperimentConfig(
            experiment_id=self.experiment_id,
            timestamp=self.timestamp,
            hash_id=self.hash_id,
            output_dir=str(self.output_dir)
        )
        self.metrics = ExperimentMetrics()
        
        # Track environment
        self._log_environment()
        
        # Artifacts list
        self.artifacts = []
    
    def _create_experiment_id(self) -> str:
        """Create unique experiment ID."""
        parts = [self.experiment_name]
        if self.use_timestamp:
            parts.append(self.timestamp)
        if self.use_hash:
            parts.append(self.hash_id)
        return "_".join(parts)
    
    def _create_output_dir(self) -> Path:
        """Create output directory with organized structure."""
        output_dir = self.base_dir / self.experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (output_dir / 'figures').mkdir(exist_ok=True)
        (output_dir / 'data').mkdir(exist_ok=True)
        (output_dir / 'meshes').mkdir(exist_ok=True)
        
        return output_dir
    
    def _log_environment(self):
        """Log environment information."""
        self.config.python_version = sys.version.split()[0]
        self.config.platform_info = platform.platform()
        
        try:
            import numpy as np
            self.config.numpy_version = np.__version__
        except:
            pass
        
        try:
            import scipy
            self.config.scipy_version = scipy.__version__
        except:
            pass
    
    def log_params(self, **kwargs):
        """Log experiment parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                # Store in solver_params
                self.config.solver_params[key] = value
    
    def log_domain(self, domain_type: str, domain_params: dict = None):
        """Log domain configuration."""
        self.config.domain_type = domain_type
        self.config.domain_params = domain_params or {}
    
    def log_mesh(self, forward_resolution: float = None, 
                 source_resolution: float = None,
                 n_boundary: int = None, n_interior: int = None,
                 n_elements: int = None):
        """Log mesh configuration."""
        if forward_resolution is not None:
            self.config.forward_mesh_resolution = forward_resolution
        if source_resolution is not None:
            self.config.source_grid_resolution = source_resolution
        if n_boundary is not None:
            self.config.n_boundary_points = n_boundary
        if n_interior is not None:
            self.config.n_interior_points = n_interior
        if n_elements is not None:
            self.config.n_elements = n_elements
    
    def log_sources(self, sources_true: list, noise_level: float = 0.0, seed: int = 0):
        """Log source configuration."""
        # Convert sources to serializable format
        serializable = []
        for s in sources_true:
            pos, intensity = s
            serializable.append({'position': list(pos), 'intensity': intensity})
        
        self.config.n_sources = len(sources_true)
        self.config.sources_true = serializable
        self.config.noise_level = noise_level
        self.config.seed = seed
    
    def log_solver(self, solver_type: str, method: str = None,
                   alpha: float = None, alpha_selection: str = 'fixed',
                   optimizer: str = None, max_iterations: int = None,
                   tolerance: float = None, n_restarts: int = None,
                   **kwargs):
        """Log solver configuration."""
        self.config.solver_type = solver_type
        if method:
            self.config.method = method
        if alpha is not None:
            self.config.alpha = alpha
        self.config.alpha_selection = alpha_selection
        if optimizer:
            self.config.optimizer = optimizer
        if max_iterations:
            self.config.max_iterations = max_iterations
        if tolerance:
            self.config.tolerance = tolerance
        if n_restarts:
            self.config.n_restarts = n_restarts
        
        # Additional params
        self.config.solver_params.update(kwargs)
    
    def log_metrics(self, **kwargs):
        """Log experiment metrics."""
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
    
    def save_figure(self, fig, filename: str, description: str = "",
                    dpi: int = 150, close: bool = True) -> str:
        """
        Save figure to file (non-blocking).
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save
        filename : str
            Filename (will be saved in figures/ subdirectory)
        description : str
            Description of the figure
        dpi : int
            Resolution
        close : bool
            Close figure after saving
            
        Returns
        -------
        filepath : str
            Full path to saved figure
        """
        filepath = self.output_dir / 'figures' / filename
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        
        if close:
            plt.close(fig)
        
        self.artifacts.append({
            'type': 'figure',
            'filename': filename,
            'filepath': str(filepath),
            'description': description
        })
        
        return str(filepath)
    
    def save_data(self, data: Union[dict, np.ndarray], filename: str,
                  description: str = "") -> str:
        """
        Save data to file.
        
        Parameters
        ----------
        data : dict or np.ndarray
            Data to save
        filename : str
            Filename (will be saved in data/ subdirectory)
        description : str
            Description of the data
            
        Returns
        -------
        filepath : str
            Full path to saved data
        """
        filepath = self.output_dir / 'data' / filename
        
        if isinstance(data, np.ndarray):
            np.save(filepath, data)
            if not filename.endswith('.npy'):
                filepath = Path(str(filepath) + '.npy')
        else:
            # Save as JSON
            if not filename.endswith('.json'):
                filepath = Path(str(filepath) + '.json')
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        self.artifacts.append({
            'type': 'data',
            'filename': filename,
            'filepath': str(filepath),
            'description': description
        })
        
        return str(filepath)
    
    def save_solution(self, q: np.ndarray, interior_points: np.ndarray,
                      filename: str = "solution.npz") -> str:
        """Save solution data."""
        filepath = self.output_dir / 'data' / filename
        np.savez(filepath, q=q, interior_points=interior_points)
        
        self.artifacts.append({
            'type': 'solution',
            'filename': filename,
            'filepath': str(filepath),
            'description': 'Solution and interior points'
        })
        
        return str(filepath)
    
    def finalize(self):
        """Finalize experiment - save all data to database."""
        # Insert config
        self.db.insert_experiment(self.config)
        
        # Insert metrics
        self.db.insert_metrics(self.experiment_id, self.metrics)
        
        # Insert artifacts
        for artifact in self.artifacts:
            self.db.insert_artifact(
                self.experiment_id,
                artifact['type'],
                artifact['filename'],
                artifact['filepath'],
                artifact.get('description', '')
            )
        
        # Save config as JSON in output dir
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2, default=str)
        
        # Save metrics as JSON
        metrics_path = self.output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics.to_dict(), f, indent=2, default=str)
        
        print(f"Experiment saved: {self.experiment_id}")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Database: {self.db_path}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
        self.db.close()


class CalibrationTracker(ExperimentTracker):
    """Specialized tracker for calibration runs."""
    
    def __init__(self, base_dir: str = 'results/calibration', **kwargs):
        super().__init__(base_dir=base_dir, experiment_name='calibration', **kwargs)
    
    def log_calibration(self, domain_type: str, calibration_data: dict):
        """Log calibration results for a domain."""
        cal_entry = {
            'calibration_id': f"{self.experiment_id}_{domain_type}",
            'timestamp': self.timestamp,
            'domain_type': domain_type,
            'config_path': str(self.output_dir / 'calibration_config.json'),
            **calibration_data
        }
        self.db.insert_calibration(cal_entry)


# Utility functions for non-blocking plots
def savefig_no_show(fig, filepath: str, dpi: int = 150, close: bool = True):
    """Save figure without displaying (non-blocking)."""
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    if close:
        plt.close(fig)


def create_figure(*args, **kwargs):
    """Create figure with non-interactive backend."""
    return plt.figure(*args, **kwargs)


def create_subplots(*args, **kwargs):
    """Create subplots with non-interactive backend."""
    return plt.subplots(*args, **kwargs)


# Query utilities
def list_experiments(base_dir: str = 'results', domain_type: str = None,
                     method: str = None, limit: int = 50) -> List[dict]:
    """List experiments from database."""
    db_path = Path(base_dir) / 'experiments.db'
    if not db_path.exists():
        return []
    
    db = ExperimentDatabase(str(db_path))
    results = db.query_experiments(domain_type=domain_type, method=method, limit=limit)
    db.close()
    return results


def get_experiment_details(base_dir: str, experiment_id: str) -> dict:
    """Get full details of an experiment."""
    db_path = Path(base_dir) / 'experiments.db'
    db = ExperimentDatabase(str(db_path))
    
    exp = db.get_experiment(experiment_id)
    if exp:
        exp['metrics'] = db.get_metrics(experiment_id)
        exp['artifacts'] = db.get_artifacts(experiment_id)
    
    db.close()
    return exp


if __name__ == '__main__':
    # Demo usage
    print("Experiment Tracker Demo")
    print("=" * 50)
    
    with ExperimentTracker(base_dir='results', experiment_name='demo') as tracker:
        # Log configuration
        tracker.log_domain('disk', {})
        tracker.log_mesh(forward_resolution=0.1, source_resolution=0.15)
        tracker.log_sources([((0.3, 0.4), 1.0), ((-0.3, -0.4), -1.0)], 
                           noise_level=0.001, seed=42)
        tracker.log_solver('FEM', method='l2', alpha=1e-4, alpha_selection='lcurve')
        
        # Simulate metrics
        tracker.log_metrics(
            position_rmse=0.12,
            localization_score=0.85,
            time_seconds=1.5,
            converged=True
        )
        
        # Create and save a test figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_title('Test Plot')
        tracker.save_figure(fig, 'test_plot.png', description='Demo test plot')
    
    print("\nExperiment completed!")
