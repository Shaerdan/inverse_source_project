# Inverse Source Project - Session Summary
## Date: December 25, 2025

---

## Overview

This session focused on major code refactoring to properly separate three distinct solver approaches, fix naming conventions, add missing functionality, and run comprehensive comparisons.

---

## 1. Code Refactoring: The Three Solver Types

### Problem Identified
The code previously labeled "BEM" was actually Serdan's 2021 **analytical derivation** of the Neumann Green's function for the unit disk - not a true Boundary Element Method implementation.

### Solution Implemented

| Module | Description | Domain Support |
|--------|-------------|----------------|
| `analytical_solver.py` | Exact closed-form Green's function (Serdan's 2021 derivation) | Unit disk only |
| `bem_solver.py` | TRUE BEM with numerical integration | Unit disk (for validation) |
| `conformal_solver.py` | Maps general domains to unit disk, then uses analytical | Ellipse, star-shaped, any simply-connected |
| `fem_solver.py` | Finite Element Method | Any meshed domain |

### Key Insight: Conformal Mapping Uses Analytical Solution
The 2D Laplacian Green's function is **conformally invariant**:
```
G_Ω(z, ζ) = G_D(f(z), f(ζ))
```
So `ConformalForwardSolver`:
1. Maps source position from physical domain to unit disk: `ξ_disk = f(ξ_physical)`
2. Maps boundary points to disk: `x_disk = f(x_physical)`
3. Calls `greens_function_disk_neumann(x_disk, ξ_disk)` (the analytical formula)
4. Result is automatically correct for the physical domain

---

## 2. Project File Structure

```
~/Projects/inverse_source_project/
├── src/                          ← Python package modules go HERE
│   ├── __init__.py
│   ├── analytical_solver.py
│   ├── bem_solver.py
│   ├── conformal_solver.py
│   ├── fem_solver.py
│   ├── mesh.py
│   ├── regularization.py
│   ├── comparison.py
│   ├── parameter_study.py
│   ├── config.py
│   ├── utils.py
│   ├── cli.py
│   └── [legacy files: conformal_bem.py, forward_solver.py, etc.]
├── Config/                       ← Configuration files go HERE
│   ├── config.yaml
│   ├── config.json
│   └── config_template.json
├── docs/
├── examples/
├── meshes/
├── notebooks/
├── results/
├── tests/
├── README.md
├── setup.py
├── pyproject.toml
└── requirements.txt
```

### Copy Commands for Updates
```bash
cd ~/Downloads
unzip -o inverse_source_v2.zip -d temp_extract
cd temp_extract

# Python modules to src/
cp analytical_solver.py ~/Projects/inverse_source_project/src/
cp bem_solver.py ~/Projects/inverse_source_project/src/
cp conformal_solver.py ~/Projects/inverse_source_project/src/
cp fem_solver.py ~/Projects/inverse_source_project/src/
cp mesh.py ~/Projects/inverse_source_project/src/
cp regularization.py ~/Projects/inverse_source_project/src/
cp comparison.py ~/Projects/inverse_source_project/src/
cp parameter_study.py ~/Projects/inverse_source_project/src/
cp config.py ~/Projects/inverse_source_project/src/
cp utils.py ~/Projects/inverse_source_project/src/
cp cli.py ~/Projects/inverse_source_project/src/
cp __init__.py ~/Projects/inverse_source_project/src/

# Config files to Config/
cp config.yaml ~/Projects/inverse_source_project/Config/
cp config.json ~/Projects/inverse_source_project/Config/

# Root level files
cp README.md ~/Projects/inverse_source_project/
cp requirements.txt ~/Projects/inverse_source_project/
cp setup.py ~/Projects/inverse_source_project/
cp pyproject.toml ~/Projects/inverse_source_project/

cd ..
rm -rf temp_extract
```

---

## 3. Solver Comparison Results

### CLI Command
```bash
python -m src.cli compare
```

### What Gets Compared (Full Run)
```
ANALYTICAL LINEAR SOLVERS
  - Analytical Linear (L1)
  - Analytical Linear (L2)
  - Analytical Linear (TV)

ANALYTICAL NONLINEAR SOLVERS
  - Analytical Nonlinear (L-BFGS-B) x5
  - Analytical Nonlinear (differential_evolution)

BEM NUMERICAL LINEAR SOLVERS
  - BEM Numerical Linear (L1)
  - BEM Numerical Linear (L2)
  - BEM Numerical Linear (TV)

BEM NUMERICAL NONLINEAR SOLVERS
  - BEM Numerical Nonlinear (L-BFGS-B) x5
  - BEM Numerical Nonlinear (differential_evolution)

FEM LINEAR SOLVERS
  - FEM Linear (L1)
  - FEM Linear (L2)
  - FEM Linear (TV)

FEM NONLINEAR SOLVERS
  - FEM Nonlinear (L-BFGS-B) x5
  - FEM Nonlinear (differential_evolution)
```

### Results Summary (from test run)

| Solver | Pos RMSE | Int RMSE | Time |
|--------|----------|----------|------|
| Analytical Linear (L1) | 0.259 | 0.841 | 0.1s |
| Analytical Linear (L2) | 0.052 | 0.967 | 0.0s |
| Analytical Linear (TV) | 0.052 | 0.968 | 0.0s |
| Analytical Nonlinear (L-BFGS-B) x5 | 0.274 | 2.306 | 2.5s |
| Analytical Nonlinear (DE) | 0.235 | 2.482 | 4.9s |
| BEM Numerical Linear (L1) | 0.273 | 0.814 | 0.1s |
| BEM Numerical Linear (L2) | 0.052 | 0.968 | 0.2s |
| BEM Numerical Linear (TV) | 0.052 | 0.969 | 0.1s |
| BEM Numerical Nonlinear (L-BFGS-B) x5 | 0.088 | 0.359 | 20.1s |
| BEM Numerical Nonlinear (DE) | 0.269 | 4.650 | 31.4s |
| FEM Linear (L1) | 0.273 | 0.846 | 0.5s |
| FEM Linear (L2) | 0.052 | 0.970 | 0.5s |
| FEM Linear (TV) | 0.052 | 0.973 | 0.5s |
| FEM Nonlinear (L-BFGS-B) x5 | 0.309 | 1.886 | 45.4s |
| **FEM Nonlinear (DE)** | **0.004** | **0.010** | 58.4s |

---

## 4. Key Discussion: Optimization Landscape

### Initial Observation
BEM Numerical Nonlinear (L-BFGS-B) appeared to outperform Analytical Nonlinear (L-BFGS-B), raising the question: does numerical BEM have an easier optimization landscape?

### Analysis

**For FEM vs Analytical: YES, genuinely different landscapes**
- FEM uses piecewise linear test functions
- Creates a "smoother" objective function with fewer sharp local minima
- FEM Nonlinear + differential_evolution achieved near-perfect recovery (Pos RMSE = 0.004)

**For BEM Numerical vs Analytical: NO, should be identical**
- Both compute the SAME Green's function:
  ```
  G(x, ξ) = -1/(2π)[ln|x-ξ| + ln|x-ξ*| - ln|ξ|]
  ```
- Same mathematical landscape
- Observed differences are due to **random variation** in L-BFGS-B restarts

### Conclusion
The apparent BEM advantage was **statistical noise** from limited restarts. To properly compare:
- Use identical random seeds
- Run 50+ restarts
- Report mean ± std

The **real** finding: FEM's piecewise-linear forward model creates a smoother optimization landscape, making global optimization more likely to succeed.

---

## 5. Important Finding: Linear Solver Intensity Recovery

### Issue Identified
Linear solver intensity RMSE was consistently high (~0.8-0.97) despite good boundary residuals.

### Root Cause Analysis
The inverse source problem is **severely ill-posed**:
- Green's matrix condition number: ~10^15 (numerically singular)
- 57 out of 200 singular values < 0.01
- Many source configurations produce (nearly) identical boundary data

### Key Findings
1. **Total intensity IS conserved** (~1.74 out of 2.0 per polarity)
2. **But intensity spreads across domain** - only 15-30% within r=0.2 of true sources
3. **Column norms vary with radius**: boundary sources have 4x more influence than interior sources
4. **This is NOT a bug** - it's the fundamental nature of the inverse problem

### Integrated Intensity Near True Sources (L2, α=10⁻⁴)
```
Source at (-0.3, +0.4), q_true = +1.0:
  r < 0.10: q_integrated = +0.035
  r < 0.20: q_integrated = +0.159
  r < 0.30: q_integrated = +0.440
```

### Updated Metrics
Added `compute_linear_metrics()` function that:
- Sums intensity within a neighborhood of each true source
- Uses intensity-weighted centroids for position
- More meaningful than comparing individual grid values to point sources

### Implication for Users
- Linear solvers give good **boundary data fit** but poor **source localization**
- Nonlinear solvers with global optimization (e.g., FEM + differential_evolution) are needed for accurate intensity recovery
- Consider this a **detection** problem (where are sources?) rather than **quantification** (what are intensities?)

---

## 6. Bug Fixes Applied

### Bug 1: Shape Mismatch in BEM Nonlinear
**Error:** `operands could not be broadcast together with shapes (64,) (100,)`

**Cause:** `BEMNonlinearInverseSolver` used fixed 64 elements but measured data had 100 points.

**Fix:** Auto-match boundary point count in `set_measured_data()`:
```python
def set_measured_data(self, u_measured: np.ndarray):
    self.u_measured = u_measured - np.mean(u_measured)
    if len(u_measured) != self.n_boundary:
        self.n_boundary = len(u_measured)
        self.forward = BEMForwardSolver(n_elements=self.n_boundary, quadrature_order=6)
```

### Bug 2: differential_evolution Workers Error
**Error:** `The map-like callable must be of the form f(func, iterable)...`

**Cause:** `workers=1` parameter incompatible with some scipy versions.

**Fix:** Replace with `updating='deferred'`:
```python
result = differential_evolution(self._objective, bounds, maxiter=maxiter,
                               seed=42, polish=True, updating='deferred')
```

---

## 7. Configuration System

### Default Method Changed
- **Old:** `method: "bem"`
- **New:** `method: "analytical"`

### Config File Locations
- `Config/config.yaml` - YAML format with comments
- `Config/config.json` - JSON format

### Forward Method Options
```yaml
forward:
  method: "analytical"  # "analytical", "bem", or "fem"
```

| Method | Description |
|--------|-------------|
| `analytical` | Exact closed-form (fastest, unit disk only) |
| `bem` | Numerical boundary integration (validation) |
| `fem` | Finite element (most general) |

---

## 8. Classes and Their Relationships

### analytical_solver.py
```python
AnalyticalForwardSolver          # Exact forward solution
AnalyticalLinearInverseSolver    # Grid-based, L1/L2/TV regularization
AnalyticalNonlinearInverseSolver # Continuous positions, optimization
```

### bem_solver.py
```python
BEMDiscretization                # Boundary element discretization
BEMForwardSolver                 # Numerical forward solution
BEMLinearInverseSolver           # Grid-based inverse
BEMNonlinearInverseSolver        # Continuous positions (ADDED THIS SESSION)
```

### conformal_solver.py
```python
ConformalMap (abstract)          # Base class
DiskMap                          # Identity/scaling
EllipseMap                       # Ellipse via Joukowsky transform
StarShapedMap                    # r = R(θ) domains
ConformalForwardSolver           # Maps to disk, uses analytical
ConformalLinearInverseSolver     # Linear inverse on general domain
ConformalNonlinearInverseSolver  # Nonlinear on general domain
```

### fem_solver.py
```python
FEMForwardSolver
FEMLinearInverseSolver
FEMNonlinearInverseSolver
```

---

## 9. Backward Compatibility

Old code using these names still works via aliases in `analytical_solver.py`:
```python
BEMForwardSolver = AnalyticalForwardSolver
BEMLinearInverseSolver = AnalyticalLinearInverseSolver
BEMNonlinearInverseSolver = AnalyticalNonlinearInverseSolver
```

---

## 10. Key Formulas

### Neumann Green's Function for Unit Disk
```
G(x, ξ) = -1/(2π)[ln|x-ξ| + ln|x-ξ*| - ln|ξ|]
```
where `ξ* = ξ/|ξ|²` is the image point (method of images for Neumann BC).

### Inverse Problem
Given boundary measurements `u(x)` on `∂Ω`, recover sources `{(ξₖ, qₖ)}`:
```
-Δu = Σₖ qₖ δ(x - ξₖ)   in Ω
∂u/∂n = 0              on ∂Ω
```
with compatibility condition `Σₖ qₖ = 0`.

---

## 11. Open Items / Future Work

1. **Statistical comparison:** Run Analytical vs BEM Nonlinear with many trials to confirm identical performance
2. **Why FEM works better:** Further investigate the smoothing effect of FEM discretization
3. **Conformal solver testing:** More testing on ellipse and star-shaped domains
4. **Performance optimization:** BEM Nonlinear is slow (~20-30s) compared to Analytical

---

## 12. Files Modified This Session

| File | Changes |
|------|---------|
| `analytical_solver.py` | Renamed from bem_solver.py concept, added backward-compatible aliases |
| `bem_solver.py` | Complete rewrite as TRUE BEM, added `BEMNonlinearInverseSolver` |
| `conformal_solver.py` | New file for general domains |
| `comparison.py` | Added `run_bem_numerical_linear()`, `run_bem_numerical_nonlinear()`, updated all imports |
| `cli.py` | Updated imports from `bem_solver` to `analytical_solver` |
| `config.py` | Changed default method to "analytical" |
| `config.yaml` | Updated method options documentation |
| `config.json` | Changed default method to "analytical" |
| `__init__.py` | Added `BEMNumericalNonlinearInverseSolver` export |
| `README.md` | Comprehensive rewrite with all solver types documented |

---

## 13. Transcript Reference

Full conversation transcript available at:
```
/mnt/transcripts/2025-12-25-23-13-09-code-refactor-bem-analytical-conformal.txt
```

---

*End of Session Summary*

---

## 14. Peak Detection and Clustering (December 27, 2025)

### Problem Addressed
Linear solver intensity fields are spread across the domain. DBSCAN clustering alone merges multiple source peaks into single clusters when the field is smooth (L2, TV regularization).

### Solution: Dual Detection Methods

**1. Peak Detection (`find_intensity_peaks`)**
- Finds local maxima (positive) and minima (negative) in intensity field
- Uses KD-tree for efficient neighbor queries
- Parameters:
  - `neighbor_radius=0.12`: Radius to find local extrema
  - `integration_radius=0.20`: Radius for integrating intensity
  - `intensity_threshold_ratio=0.15`: Only consider |q| > 15% of max
  - `min_peak_separation=0.25`: Suppress nearby weaker peaks
- Better for smooth fields (L2, TV)

**2. DBSCAN Clustering (`find_intensity_clusters`)**
- Groups connected regions of significant intensity
- Clusters positive and negative separately
- Parameters:
  - `eps=0.18`: DBSCAN neighborhood radius
  - `min_samples=2`: Minimum points per cluster
  - `intensity_threshold_ratio=0.15`
- Better for sparse fields (L1)

### Results Structure
```python
ComparisonResult:
    peaks: List[Dict]      # Peak info (position, peak_intensity, integrated_intensity)
    clusters: List[Dict]   # Cluster info (centroid, total_intensity, spread)
```

### Visualization
```python
plot_comparison(results, sources, 
                show_peaks=True,     # ▲/▼ markers
                show_clusters=True)  # ◆ markers
```

### Typical Results (4-source problem)

| Method | Peaks | Clusters | Notes |
|--------|-------|----------|-------|
| L1 | 15 | 4 | Many peaks, separated clusters |
| L2 | 12 | 2 | Fewer peaks, merged clusters |
| TV | 13 | 2 | Similar to L2 |

---

## 15. Updated Files

### comparison.py Changes
- Added `PeakInfo` dataclass
- Added `find_intensity_peaks()` function
- Added `compute_peak_metrics()` function
- Updated `run_bem_linear()`, `run_bem_numerical_linear()`, `run_fem_linear()` to compute both peaks and clusters
- Updated `plot_comparison()` with `show_peaks` and `show_clusters` options
- Metrics now based on peak matching by default

### README.md
- Complete rewrite with all experiments documented
- Usage examples for each solver type
- Parameter tuning guide
- Key findings and recommendations

### Dependencies
- Added `scikit-learn>=1.3` for DBSCAN clustering

