# Session Summary: January 1, 2026 (v7.6)

## Key Insight: RMSE Metric Was Misleading

The peak-based position RMSE was **fundamentally flawed**:
- Threshold-dependent (changing threshold changes RMSE dramatically)
- Low α gives many spurious peaks scattered everywhere
- Some peaks happen to be near true sources by chance → artificially good RMSE
- The "optimal" RMSE was actually from garbage solutions

## New Proper Metrics

### 1. Localization Score (0 to 1)
**"How much intensity is near true sources?"**
- Uses Gaussian weighting, no thresholding needed
- 1.0 = all intensity exactly at true sources
- Our linear methods get 0.1-0.3 (most intensity is NOT near true sources)

### 2. Sparsity Ratio (0 to 1)
**"How concentrated is the solution?"**
- Measures if intensity is in few points (point-like) vs spread out
- 1.0 = 90% of intensity in exactly 4 points
- Our linear methods get 0.03-0.14 (very spread out)

## Results with Proper Metrics at L-curve Corner

| Method | α | Localization | Sparsity |
|--------|---|--------------|----------|
| L1 | 1.2e-04 | 0.12 | **0.14** |
| L2 | 2.0e-04 | **0.29** | 0.03 |
| TV | 7.5e-05 | **0.27** | 0.03 |

**Interpretation:**
- **L1**: Higher sparsity (more concentrated), but LOWER localization (wrong places)
- **L2/TV**: Lower sparsity (more spread), but HIGHER localization (spread around true sources)

## --compare Now Uses L-curve by Default

```bash
# Default: automatic L-curve optimal α for each method
python -m inverse_source compare --domain disk

# Override with fixed α
python -m inverse_source compare --domain disk --fixed-alpha --alpha 1e-4
```

Output shows optimal α found for each method:
```
Finding optimal α for L1...   L1: α = 1.27e-04
Finding optimal α for L2...   L2: α = 2.34e-04
Finding optimal α for TV...   TV: α = 6.95e-05
```

## Files Added/Modified

| File | Changes |
|------|---------|
| `parameter_selection.py` | Added `localization_score()`, `sparsity_ratio()`, updated `ParameterSweepResult` |
| `analytical_solver.py` | TV default changed to cvxpy for accuracy |
| `comparison.py` | Added L-curve auto-alpha in `compare_all_solvers_general()` |
| `cli.py` | `--compare` now uses L-curve by default, `--fixed-alpha` to override |
| `__init__.py` | Export new metrics |

## Package Version: v7.6

## Usage

```python
from inverse_source import (
    localization_score, sparsity_ratio,
    param_sweep_lcurve, AnalyticalLinearInverseSolver
)

# Run sweep
result = param_sweep_lcurve(
    solver.G, u_noisy, solver.interior_points,
    method='l1',
    sources_true=sources,  # For localization score
    verbose=True
)

# Get metrics at L-curve corner
loc = result.localization_scores[result.idx_lcurve]
spar = result.sparsity_ratios[result.idx_lcurve]
print(f"Localization: {loc:.3f}, Sparsity: {spar:.3f}")
```

## Deployment

```bash
cd ~/Downloads && unzip -o inverse_source_v7.6.zip -d temp && \
cp temp/*.py ~/Projects/inverse_source_project/src/ && \
rm -rf temp && \
cd ~/Projects/inverse_source_project && pip install -e . --break-system-packages
```

## Important Note

The **nonlinear** solver (recovering exact point source positions) should still use position RMSE since it finds discrete source locations, not a distributed field. The new metrics are primarily for evaluating **linear** solver outputs.
