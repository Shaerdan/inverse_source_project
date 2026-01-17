# Domain Architecture Migration Guide

## Overview

This document describes the migration from scattered domain-specific code to the unified `domains.py` architecture.

**Problem**: The codebase had 40+ scattered `if domain_type == 'disk': ... elif domain_type == 'ellipse': ...` blocks. When a bug was fixed for one domain, it often wasn't fixed for others.

**Solution**: Centralized `Domain` class hierarchy with a registry pattern.

---

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `domains.py` | **NEW** | Unified domain abstraction |
| `comparison.py` | Modified | Use domains.py for source/sensor generation |
| `cli.py` | No change | Already uses comparison.py functions |
| `conformal_solver.py` | Future | Can accept Domain objects |
| `fem_solver.py` | Future | Can accept Domain objects |

---

## Quick Start

### Before (Old Way)
```python
# Scattered in comparison.py, cli.py, etc.
if domain_type == 'disk':
    sources = [((0.5, 0.5), 1.0), ...]
elif domain_type == 'ellipse':
    a, b = params['a'], params['b']
    sources = [((a*0.5, b*0.5), 1.0), ...]
elif domain_type == 'star':
    # Different logic again...
```

### After (New Way)
```python
from domains import get_domain

domain = get_domain(domain_type, **params)
sources = domain.generate_sources(n_sources=6, seed=42)
sensors = domain.get_sensor_locations(100)
bounds = domain.get_optimization_bounds(n_sources=6)
```

---

## Domain Class API

### Core Methods (All Domains)

```python
class Domain:
    # Geometry
    def get_boundary_points(n_points: int) -> np.ndarray
    def contains_point(x: float, y: float) -> bool
    def get_characteristic_size() -> float
    
    # Source/Sensor Generation
    def generate_sources(n_sources: int, seed: int) -> List[((x,y), intensity)]
    def get_sensor_locations(n_sensors: int) -> np.ndarray
    
    # Optimization Support
    def get_optimization_bounds(n_sources: int) -> List[Tuple[min, max]]
    
    # Solver Compatibility
    def get_conformal_map() -> ConformalMap or None
    def get_fem_mesh_params() -> dict
```

### Capability Flags

```python
class DiskDomain(Domain):
    supports_analytical = True   # AnalyticalSolver works
    supports_conformal = True    # ConformalSolver works  
    supports_fem = True          # FEMSolver works

class PolygonDomain(Domain):
    supports_analytical = False
    supports_conformal = False   # MFS issues with non-convex
    supports_fem = True
```

---

## Available Domains

| Domain | Class | Analytical | Conformal | FEM | Notes |
|--------|-------|------------|-----------|-----|-------|
| disk | `DiskDomain` | ✅ | ✅ | ✅ | Gold standard |
| ellipse | `EllipseDomain` | ❌ | ✅ | ✅ | Joukowsky map |
| star | `StarDomain` | ❌ | ✅ | ✅ | Numerical MFS |
| square | `SquareDomain` | ❌ | ✅ | ✅ | Schwarz-Christoffel |
| brain | `BrainDomain` | ❌ | ✅ | ✅ | Realistic shape |
| polygon | `PolygonDomain` | ❌ | ❌ | ✅ | **Archived** |

---

## Migration Steps

### Step 1: Add domains.py to your project
Copy `domains.py` to `src/domains.py`

### Step 2: Update imports in comparison.py
```python
# Add at top of comparison.py
from domains import (
    get_domain, 
    DomainRegistry,
    create_domain_sources_unified,
    get_sensor_locations_unified
)
```

### Step 3: Replace scattered functions
The three main functions to replace:

#### get_sensor_locations()
```python
# Old (90 lines of if/elif):
def get_sensor_locations(domain_type, domain_params, n_sensors):
    if domain_type == 'disk':
        ...
    elif domain_type == 'ellipse':
        ...
    # ... 6 more cases

# New (3 lines):
def get_sensor_locations(domain_type, domain_params=None, n_sensors=100):
    domain = get_domain(domain_type, **_extract_params(domain_type, domain_params))
    return domain.get_sensor_locations(n_sensors)
```

#### create_domain_sources()
```python
# Old (150 lines of if/elif):
def create_domain_sources(domain_type, domain_params, n_sources, ...):
    if domain_type == 'disk':
        ...
    elif domain_type == 'ellipse':
        ...
    # ... 6 more cases with different bugs in each

# New (3 lines):
def create_domain_sources(domain_type, domain_params=None, n_sources=4, **kwargs):
    domain = get_domain(domain_type, **_extract_params(domain_type, domain_params))
    return domain.generate_sources(n_sources=n_sources, **kwargs)
```

#### get_conformal_map()
```python
# Old (50 lines):
def get_conformal_map(domain_type, domain_params):
    if domain_type == 'disk':
        ...
    # ...

# New (3 lines):
def get_conformal_map(domain_type, domain_params=None):
    domain = get_domain(domain_type, **_extract_params(domain_type, domain_params))
    return domain.get_conformal_map()
```

### Step 4: Helper function for param extraction
```python
def _extract_params(domain_type: str, domain_params: dict) -> dict:
    """Convert old-style domain_params dict to new kwargs."""
    if not domain_params:
        return {}
    
    if domain_type == 'ellipse':
        return {'a': domain_params.get('a', 2.0), 
                'b': domain_params.get('b', 1.0)}
    elif domain_type == 'star':
        return {'n_petals': domain_params.get('n_petals', 5),
                'amplitude': domain_params.get('amplitude', 0.3)}
    elif domain_type in ['square', 'polygon']:
        if 'vertices' in domain_params:
            return {'vertices': domain_params['vertices']}
    return {}
```

---

## Testing the Migration

```python
# Test that old and new produce same results
from comparison import create_domain_sources as old_func
from domains import create_domain_sources_unified as new_func

for domain in ['disk', 'ellipse', 'star', 'square', 'brain']:
    old_sources = old_func(domain, n_sources=6)
    new_sources = new_func(domain, n_sources=6)
    
    assert len(old_sources) == len(new_sources)
    print(f"{domain}: OK")
```

---

## Adding a New Domain

With the old architecture, adding a domain required changes in 10+ places.

With the new architecture:

```python
# 1. Create domain class in domains.py
class HeartDomain(Domain):
    name = "heart"
    supports_analytical = False
    supports_conformal = True
    supports_fem = True
    
    def get_boundary_points(self, n_points):
        # Heart curve parametrization
        t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        x = 16 * np.sin(t)**3
        y = 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)
        return np.column_stack([x/17, y/17])  # Normalize
    
    # ... implement other abstract methods

# 2. Register in DomainRegistry
class DomainRegistry:
    _domains = {
        # ... existing domains
        'heart': HeartDomain,
    }

# 3. Done! All solvers automatically work with the new domain
```

---

## Configuration via DomainConfig

Global settings that apply to all domains:

```python
from domains import DomainConfig, DomainRegistry

config = DomainConfig(
    default_n_sources=4,
    min_boundary_distance=0.15,
    max_boundary_distance=0.35,
    default_n_sensors=100,
    intensity_bound=10.0,
    position_margin=0.05,
    default_forward_resolution=0.1,
    default_source_resolution=0.15,
)

registry = DomainRegistry(config)
disk = registry.get('disk')
```

---

## Phase 2: Solver Integration (Future)

Eventually, solvers can accept Domain objects directly:

```python
# Future API
from domains import get_domain
from conformal_solver import ConformalNonlinearInverseSolver

domain = get_domain('ellipse', a=2.0, b=1.0)
solver = ConformalNonlinearInverseSolver(domain, n_sources=4)
# Solver automatically gets correct:
# - conformal map
# - bounds
# - sensor locations
```

---

## Troubleshooting

### "Domain not found" error
```python
# Check available domains
from domains import list_domains
print(list_domains())  # ['disk', 'ellipse', 'star', 'square', 'brain']
print(list_domains(active_only=False))  # includes 'polygon'
```

### Sources not respecting n_sources
Make sure you're using the new functions:
```python
# Wrong - may use old hardcoded version
from comparison import create_domain_sources

# Right - uses new unified version  
from domains import get_domain
domain = get_domain('brain')
sources = domain.generate_sources(n_sources=6)
```

### Conformal map returns None
```python
domain = get_domain('polygon')
cmap = domain.get_conformal_map()  # Returns None - polygon doesn't support conformal

# Check capability first
if domain.supports_conformal:
    cmap = domain.get_conformal_map()
```

---

## Version History

- **v7.47**: Initial domains.py with unified architecture
- **v7.46**: Scattered domain logic (40+ if/elif blocks)
