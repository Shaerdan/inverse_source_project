# Session Summary: January 1, 2026

## Overview
Extended comprehensive solver comparison to all geometries, plus fixed several bugs identified from user testing.

---

## Bug Fixes (v7.4)

### 1. Point-in-Polygon Constraint for Nonlinear Solver ✅

**Problem**: FEM nonlinear solver for polygons used rectangular bounding box, allowing sources outside non-convex domains (L-shape).

**Fix**: Added point-in-polygon test using ray casting algorithm in `fem_solver.py`

**Result**: Polygon nonlinear RMSE improved from 0.85 → 0.07

### 2. Star Domain Support ✅

**Problem**: `create_domain_sources('star')` raised "Unknown domain type"

**Fix**: Added star sources to `create_domain_sources()` in `comparison.py`

### 3. TV Regularization Alpha Scaling ✅

**Problem**: TV used same α as L1/L2, producing more peaks and worse results

**Fix**: Added 100x alpha scaling for TV method:
```python
def get_alpha_for_method(base_alpha, method):
    if method == 'tv':
        return base_alpha * 100  # TV needs larger alpha
    return base_alpha
```

**Result**: TV now produces fewer peaks and better RMSE

### 4. Ellipse/Polygon Domain Constraints ✅

**Fix**: Updated `_objective()` in `FEMNonlinearInverseSolver` to properly check:
- Polygon: point-in-polygon test
- Ellipse: (x/a)² + (y/b)² < 0.85²
- Disk: x² + y² < 0.85²

---

## Test Results After Fixes

### Polygon (L-shape)
```
FEM Polygon Linear (L1): Pos RMSE=0.2646
FEM Polygon Linear (L2): Pos RMSE=0.1803
FEM Polygon Linear (TV): Pos RMSE=0.1173  ← TV now best!
FEM Polygon Nonlinear:   Pos RMSE=0.0686  ← Fixed from 0.85
```

---

## Methods Run Per Domain

| Domain | Linear (L1, L2, TV) | Nonlinear |
|--------|---------------------|-----------|
| **disk** | Analytical + FEM | Analytical + FEM |
| **ellipse** | Conformal + FEM | Conformal + FEM |
| **star** | Conformal | Conformal |
| **polygon/square** | FEM | FEM |

---

## Files Modified

| File | Changes |
|------|---------|
| `fem_solver.py` | Point-in-polygon, domain_type/params, domain-aware objective |
| `comparison.py` | Star sources, TV alpha scaling, FEM ellipse wrappers |
| `conformal_solver.py` | Added `solve_tv()` |
| `cli.py` | Unified comparison |

---

## Package Version: v7.4

## Deployment
```bash
cd ~/Downloads && unzip -o inverse_source_v7.4.zip -d temp && \
cp temp/*.py ~/Projects/inverse_source_project/src/ && \
cp temp/*.md ~/Projects/inverse_source_project/docs/ && \
cp temp/*.tex temp/*.bib ~/Projects/inverse_source_project/docs/ && \
cp temp/setup.py temp/requirements.txt ~/Projects/inverse_source_project/ && \
cp temp/*.yaml temp/*.json ~/Projects/inverse_source_project/ && \
rm -rf temp && \
cd ~/Projects/inverse_source_project && pip install -e . --break-system-packages
```
