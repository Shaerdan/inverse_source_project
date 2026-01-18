# SUMMARY OF FIXES

## Files to Replace

Replace these 4 files in your `src/` directory:

1. **comparison.py**
2. **fem_solver.py** 
3. **analytical_solver.py**
4. **conformal_solver.py**

---

## What Was Fixed

### 1. comparison.py - CRITICAL BUG FIX

**`run_bem_nonlinear` (line ~749)** and **`run_fem_nonlinear` (line ~958)**:
- BEFORE: Manual `differential_evolution()` call with WRONG parameterization
  - Created INTERLEAVED bounds with n-1 intensities: `[x0,y0,q0,x1,y1,q1,...,xn,yn]`
  - But `_params_to_sources()` expects GROUPED layout: `[x0,y0,x1,y1,...,q0,q1,...,qn]`
  - This completely corrupted positions and intensities for DE
- AFTER: Uses `inverse.solve(method='differential_evolution')` which has correct parameterization

**All DE maxiter values increased to 2000**:
- `run_bem_nonlinear`: 200 → 2000
- `run_fem_nonlinear`: 100 → 2000
- `run_bem_numerical_nonlinear`: 100 → 2000
- `run_fem_polygon_nonlinear`: 500 → 2000
- `run_fem_ellipse_nonlinear`: 500 → 2000

### 2. fem_solver.py

**DE minimum maxiter increased**:
- Line ~1258: `max(1000, maxiter)` → `max(2000, maxiter)`

### 3. analytical_solver.py

**DE minimum maxiter increased**:
- Line ~727: `max(1000, maxiter)` → `max(2000, maxiter)`

### 4. conformal_solver.py

**DE maxiter increased**:
- Line ~1492: `maxiter=1000` → `maxiter=2000`

---

## Root Cause Explanation

The bug was in `comparison.py` where DE was called manually with bounds that didn't match the solver's internal parameterization:

```python
# BUGGY CODE in comparison.py created WRONG bounds:
for i in range(n):
    bounds.extend([(-0.8, 0.8), (-0.8, 0.8)])  # x, y
    if i < n - 1:  # BUG: n-1 intensities, INTERLEAVED
        bounds.append((-5.0, 5.0))
# Result: 17 params in layout [x0,y0,q0,x1,y1,q1,...,x5,y5]

# But _params_to_sources expects 18 params in GROUPED layout:
# [x0,y0,x1,y1,x2,y2,...,x5,y5,q0,q1,q2,q3,q4,q5]
```

This caused DE to optimize in a completely different parameter space than what the objective function expected, resulting in garbage results.

L-BFGS-B worked because it used `inverse.solve()` which has correct parameterization internally.

---

## Expected Results After Fix

For the six_sources preset on disk domain:
- BEFORE: FEM DE RMSE ≈ 0.175 (27x worse than L-BFGS-B)
- AFTER: FEM DE RMSE should be ≈ 0.005-0.02 (comparable to L-BFGS-B)

---

## Regarding BEM Naming

The BEM naming (run_bem_*, BEMSolver, etc.) is legacy but internally consistent:
- `analytical_solver.py` has correct Analytical* names with BEM* aliases for backward compatibility
- `bem_solver.py` is a separate TRUE BEM implementation for distributed sources
- Renaming these requires changes across many files and test scripts

If you want a complete rename, let me know and I can do that separately.
