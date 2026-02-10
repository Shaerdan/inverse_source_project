# Linear Methods Boundary Bias Analysis

Publication materials for demonstrating and correcting boundary bias in 
grid-based linear inverse methods for source localization.

## The Problem

Standard linear methods (L2, L1, TV regularization) for inverse source problems
exhibit **boundary bias**: reconstructed intensity concentrates near the domain
boundary regardless of true source locations.

**Root cause**: Green's function column norms vary ~20× between boundary and 
center sources: `||G_j|| ~ (1 - ρ_j)` where `ρ_j` is the conformal radius. 
Standard regularization penalizes `||q||` uniformly, but achieving the same 
data fit costs more intensity for center sources than boundary sources.

## The Fix

**Depth-weighted regularization**: Replace `α||q||` with `α Σ w_j |q_j|` where:

```
w_j = (1 - ρ_j)^(-β)    with β = 1.0 (empirically validated)
```

This compensates for the sensitivity variation, giving approximately uniform 
penalty across depths.

**Alpha selection**: Use the discrepancy principle instead of L-curve:
- Target residual: `||Gq - u|| ≈ σ√M × 1.3`
- Select smallest `α` achieving this residual

## Results

| Method | Original Target% | Weighted Target% | Improvement |
|--------|------------------|------------------|-------------|
| L2     | 23%              | 34%              | +11%        |
| **L1** | **1%**           | **76%**          | **+75%**    |
| TV     | 18%              | 33%              | +15%        |

L1 shows the most dramatic improvement because it's designed to be sparse,
making it most susceptible to the boundary bias (and most responsive to the fix).

## File Structure

```
linear_methods_publication/
├── src/
│   └── depth_weighted_solvers.py    # Core weighted solver implementations
├── generate_boundary_bias_figures.py # Main figure generation script
├── run_multiseed_validation.py       # Statistical validation across seeds
├── generate_slurm_jobs.py            # SLURM job generator for cluster
└── README.md                          # This file
```

## Quick Start (Local)

```bash
# Generate all figures for disk domain
python generate_boundary_bias_figures.py --domain disk --seed 42 --output-dir ./figs

# Run multi-seed validation
python run_multiseed_validation.py --domain disk --n-seeds 20 --output-dir ./figs
```

## Cluster Execution (SLURM)

```bash
# Generate SLURM scripts
python generate_slurm_jobs.py --output-dir ./slurm_jobs --n-seeds 50

# Submit jobs
cd slurm_jobs
bash submit_figures.sh       # 3 jobs (one per domain)
bash submit_multiseed.sh     # 3 × 50 = 150 jobs

# After completion, aggregate results
python aggregate_multiseed.py --domain disk --results-dir ../results_boundary_bias --n-seeds 50
```

## Output Figures

1. **Figure A** (`fig_A_column_norm_*.png`): Column norm sensitivity curve
   - Shows ||G_j|| vs ρ_j with ~20× variation
   - Explains the physics of boundary bias

2. **Figure B** (`fig_B_heatmaps_*.png`): 3×2 heatmap grid
   - Rows: L2, L1, TV
   - Columns: Original, Depth-Weighted
   - True sources marked with ×
   - Visual proof of bias correction

3. **Figure C** (`fig_C_intensity_dist_*.png`): Intensity distribution bar chart
   - Shows intensity fraction by radial zone
   - Compares original vs weighted

4. **Figure D** (`fig_D_boxplots_*.png`): Multi-seed box plots
   - Statistical comparison across seeds
   - Shows robustness of the fix

## Dependencies

- numpy
- scipy
- matplotlib
- (Optional) cvxpy for comparison with convex solvers

The scripts also require the project's `analytical_solver.py` and 
`conformal_solver.py` modules.

## Integration with Existing Codebase

The `depth_weighted_solvers.py` module is designed to integrate with 
`run_comparison_job.py`:

```python
from depth_weighted_solvers import (
    compute_conformal_radii_disk,
    compute_depth_weights,
    solve_l1_weighted_admm,
    compute_l_curve_weighted
)

# In run_comparison_job.py:
radii = compute_conformal_radii_disk(grid_points)
weights = compute_depth_weights(radii, beta=1.0)
alpha, _, _ = compute_l_curve_weighted(G, u, 'l1', weights, noise_level=sigma_noise)
q = solve_l1_weighted_admm(G, u, alpha, A_eq, b_eq, weights)
```

## Mathematical Details

See the main paper, Section 7 (Linear vs Nonlinear Methods), for full derivation
and theoretical justification.

Key insight: The condition `||G_j|| ~ (1 - ρ_j)` means that to produce the 
same boundary measurement, a source at ρ=0.9 needs 10× less intensity than 
one at ρ=0.1. Standard regularization sees the boundary source as "cheaper" 
and preferentially places intensity there.

## Citation

[Paper citation TBD]
