#!/usr/bin/env python3
"""
Aggregate Perturbation Test Results (Test A)
=============================================

Loads per-seed perturbation test results and generates:
  - Plot 1 (KEY): Fraction of null-space directions vs N - N_max
  - Plot 2: Minimum forward change vs N - N_max
  - Plot 3: Predicted vs observed null space dimension (from SVD)
  - Plot 4: SVD singular value spectra
  - Summary statistics

Usage:
    python aggregate_perturbation_results.py --results-dir perturbation_results_general_random_intensity_random_rho
    python aggregate_perturbation_results.py --all-cases
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# CONSTANTS
# =============================================================================

TEST_CASES = ['same_radius', 'same_angle', 'general', 'general_random_intensity']


# =============================================================================
# DATA LOADING
# =============================================================================

def find_result_folders(base_dir: str) -> List[str]:
    """Find all result folders containing results.json"""
    if not os.path.exists(base_dir):
        return []
    folders = []
    for name in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, name)
        if os.path.isdir(folder_path):
            result_file = os.path.join(folder_path, 'results.json')
            if os.path.exists(result_file):
                folders.append(folder_path)
    return folders


def load_results(base_dir: str) -> List[dict]:
    """Load all results from a directory."""
    folders = find_result_folders(base_dir)
    results = []
    for folder in folders:
        result_file = os.path.join(folder, 'results.json')
        try:
            with open(result_file) as f:
                data = json.load(f)
            if 'error' not in data:
                results.append(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  WARNING: Skipping {folder}: {e}")
    return results


# =============================================================================
# POOLED DATA EXTRACTION
# =============================================================================

def extract_pooled_data(results: List[dict]) -> list:
    """
    Extract pooled (seed, N) data points for analysis.

    Returns list of dicts with per-record data.
    """
    records = []

    for res in results:
        seed = res['seed']
        n_star = res['n_star']
        N_max = res['N_max']

        for nr in res['results_by_N']:
            record = {
                'seed': seed,
                'N': nr['N'],
                'n_star': n_star,
                'N_max': N_max,
                'N_minus_Nmax': nr['N_minus_Nmax'],
                'dim_params': nr['dim_params'],
                'dim_equations': nr['dim_equations'],
                'svd_rank': nr['svd_rank'],
                'svd_null_dim': nr['svd_null_dim'],
                'null_dim_predicted': nr['null_dim_predicted'],
                'svd_condition_number': nr['svd_condition_number'],
            }

            # Extract per-epsilon metrics
            for eps_key, eps_data in nr['perturbation'].items():
                record[f'{eps_key}_null_fwd_max'] = eps_data.get('null_fwd_max')
                record[f'{eps_key}_range_fwd_median'] = eps_data.get('range_fwd_median')
                record[f'{eps_key}_separation_ratio'] = eps_data.get('separation_ratio')
                record[f'{eps_key}_random_fwd_min'] = eps_data.get('random_fwd_min')

            records.append(record)

    return records


# =============================================================================
# PLOTTING
# =============================================================================

def generate_plots(records: List[dict], test_case: str, output_dir: str,
                   eps_key: str = 'eps_1e-07'):
    """Generate all plots for the perturbation test."""

    if not HAS_MATPLOTLIB:
        print("  matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    N_minus_Nmax = np.array([r['N_minus_Nmax'] for r in records])

    # ------------------------------------------------------------------
    # Plot 1 (KEY): Separation ratio vs N - N_max
    # Shows how cleanly null-space directions are separated from range.
    # Only points where null space exists (N > N_max).
    # ------------------------------------------------------------------
    sep_ratios = []
    has_null_x = []
    for r in records:
        sep = r.get(f'{eps_key}_separation_ratio')
        if sep is not None and sep > 0:
            sep_ratios.append(sep)
            has_null_x.append(r['N_minus_Nmax'])

    if sep_ratios:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(has_null_x, sep_ratios, alpha=0.4, s=15, c='steelblue',
                   edgecolors='none')
        ax.set_yscale('log')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5,
                   label='$N = N_{max}$')
        ax.set_xlabel('$N - N_{max}$', fontsize=12)
        ax.set_ylabel('Separation ratio (range median / null max)', fontsize=12)
        ax.set_title(f'Null vs Range Separation — {test_case}', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, f'plot1_separation_ratio_{test_case}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

    # ------------------------------------------------------------------
    # Plot 2: Null vs range ||Δh||/ε on same axes (log scale) vs N - N_max
    # Orange = null-space max, blue = range-space median.
    # ------------------------------------------------------------------
    null_max_vals = []
    null_max_x = []
    range_med_vals = []
    range_med_x = []

    for r in records:
        nm = r.get(f'{eps_key}_null_fwd_max')
        rm = r.get(f'{eps_key}_range_fwd_median')
        if nm is not None:
            null_max_vals.append(nm)
            null_max_x.append(r['N_minus_Nmax'])
        if rm is not None:
            range_med_vals.append(rm)
            range_med_x.append(r['N_minus_Nmax'])

    fig, ax = plt.subplots(figsize=(8, 5))
    if range_med_vals:
        ax.scatter(range_med_x, range_med_vals, alpha=0.3, s=12, c='steelblue',
                   edgecolors='none', label='Range (median $\\|\\Delta h\\|/\\epsilon$)')
    if null_max_vals:
        ax.scatter(null_max_x, null_max_vals, alpha=0.4, s=15, c='darkorange',
                   edgecolors='none', label='Null (max $\\|\\Delta h\\|/\\epsilon$)')
    ax.set_yscale('log')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5,
               label='$N = N_{max}$')
    ax.set_xlabel('$N - N_{max}$', fontsize=12)
    ax.set_ylabel('$\\|\\Delta h\\| / \\epsilon$', fontsize=12)
    ax.set_title(f'Forward Map Sensitivity — {test_case}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f'plot2_null_vs_range_{test_case}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # ------------------------------------------------------------------
    # Plot 3: Predicted vs observed null space dimension (SVD)
    # ------------------------------------------------------------------
    predicted = np.array([r['null_dim_predicted'] for r in records])
    observed = np.array([r['svd_null_dim'] for r in records])

    fig, ax = plt.subplots(figsize=(6, 6))
    jitter = 0.15 * np.random.randn(len(predicted))
    ax.scatter(predicted + jitter, observed + jitter, alpha=0.3, s=15,
               c='seagreen', edgecolors='none')
    max_dim = max(max(predicted), max(observed)) + 1
    ax.plot([0, max_dim], [0, max_dim], 'r--', linewidth=1.5, label='$y = x$')
    ax.set_xlabel('Predicted null dim: $\\max(0, 3N - 2n^*)$', fontsize=12)
    ax.set_ylabel('Observed null dim (SVD)', fontsize=12)
    ax.set_title(f'Null Space Dimension — {test_case}', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f'plot3_null_dim_pred_vs_obs_{test_case}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def generate_plots_single_seed(result: dict, output_dir: str):
    """Generate detailed plots for a single seed's results."""

    if not HAS_MATPLOTLIB:
        return

    os.makedirs(output_dir, exist_ok=True)
    test_case = result['test_case']
    seed = result['seed']
    n_star = result['n_star']
    N_max = result['N_max']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: SVD singular value spectra for each N
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(result['results_by_N'])))

    for idx, nr in enumerate(result['results_by_N']):
        N = nr['N']
        svs = np.array(nr['svd_singular_values'])
        label = f"N={N} (3N={3*N})"
        marker = 'o' if N <= N_max else 'x'
        ax.semilogy(range(1, len(svs) + 1), svs, '-' + marker,
                     color=colors[idx], markersize=4, label=label, alpha=0.8)

    ax.axhline(y=1e-10, color='gray', linestyle=':', label='Rank threshold')
    ax.set_xlabel('Singular value index', fontsize=11)
    ax.set_ylabel('Singular value', fontsize=11)
    ax.set_title(f'SVD Spectra — seed {seed}, $n^*$={n_star}, '
                 f'$N_{{max}}$={N_max:.1f}', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: null dimension vs N
    ax = axes[1]
    N_vals = [nr['N'] for nr in result['results_by_N']]
    null_pred = [nr['null_dim_predicted'] for nr in result['results_by_N']]
    null_obs = [nr['svd_null_dim'] for nr in result['results_by_N']]

    ax.plot(N_vals, null_pred, 'rs--', markersize=8, label='Predicted: max(0, 3N-2n*)')
    ax.plot(N_vals, null_obs, 'bo-', markersize=8, label='Observed (SVD)')
    ax.axvline(x=N_max, color='green', linestyle=':', linewidth=1.5,
               label=f'$N_{{max}}$ = {N_max:.1f}')
    ax.set_xlabel('N (number of sources)', fontsize=11)
    ax.set_ylabel('Null space dimension', fontsize=11)
    ax.set_title(f'Null Space Dimension — seed {seed}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir,
                        f'single_seed_{seed}_svd_{test_case}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def print_summary(records: List[dict], test_case: str,
                  eps_key: str = 'eps_1e-07'):
    """Print summary statistics."""

    print(f"\n{'='*70}")
    print(f"PERTURBATION TEST SUMMARY: {test_case}")
    print(f"{'='*70}")

    n_seeds = len(set(r['seed'] for r in records))
    n_records = len(records)
    print(f"  Seeds: {n_seeds}")
    print(f"  Total (seed, N) pairs: {n_records}")
    print(f"  Epsilon for analysis: {eps_key}")

    # Split by below/above N_max
    below = [r for r in records if r['N_minus_Nmax'] <= 0]
    above = [r for r in records if r['N_minus_Nmax'] > 0]

    print(f"\n  Below N_max: {len(below)} points")
    print(f"  Above N_max: {len(above)} points")

    # Below N_max: should have NO null space
    if below:
        null_dims_below = [r['svd_null_dim'] for r in below]
        print(f"\n  BELOW N_max:")
        print(f"    All svd_null_dim == 0: {all(d == 0 for d in null_dims_below)}")
        print(f"    Expected: True (no null space)")

    # Above N_max: should have null space with huge separation
    if above:
        seps = [r.get(f'{eps_key}_separation_ratio') for r in above
                if r.get(f'{eps_key}_separation_ratio') is not None]
        null_maxs = [r.get(f'{eps_key}_null_fwd_max') for r in above
                     if r.get(f'{eps_key}_null_fwd_max') is not None]
        print(f"\n  ABOVE N_max:")
        if seps:
            print(f"    Separation ratio: median={np.median(seps):.2e}, "
                  f"min={np.min(seps):.2e}")
        if null_maxs:
            print(f"    Null ||Δh||/ε:    median={np.median(null_maxs):.2e}, "
                  f"max={np.max(null_maxs):.2e}")
        print(f"    Expected: separation >> 1, null ||Δh||/ε ≈ O(ε)")

    # SVD null dimension accuracy
    all_pred = [r['null_dim_predicted'] for r in records]
    all_obs = [r['svd_null_dim'] for r in records]
    matches = sum(1 for p, o in zip(all_pred, all_obs) if p == o)
    print(f"\n  SVD null dim matches prediction: {matches}/{n_records} "
          f"({100*matches/n_records:.1f}%)")

    # Breakdown by N value
    by_N = defaultdict(list)
    for r in records:
        by_N[r['N']].append(r)

    print(f"\n  Per-N breakdown:")
    print(f"  {'N':>4s}  {'3N':>4s}  {'2n*':>6s}  {'pred_null':>10s}  "
          f"{'obs_null':>9s}  {'null_max':>10s}  {'range_med':>10s}  {'sep_ratio':>10s}")
    print(f"  {'-'*75}")

    for N_val in sorted(by_N.keys()):
        recs = by_N[N_val]
        pred = np.mean([r['null_dim_predicted'] for r in recs])
        obs = np.mean([r['svd_null_dim'] for r in recs])
        dim_eq = np.mean([r['dim_equations'] for r in recs])

        null_maxs = [r.get(f'{eps_key}_null_fwd_max') for r in recs
                     if r.get(f'{eps_key}_null_fwd_max') is not None]
        range_meds = [r.get(f'{eps_key}_range_fwd_median') for r in recs
                      if r.get(f'{eps_key}_range_fwd_median') is not None]
        seps = [r.get(f'{eps_key}_separation_ratio') for r in recs
                if r.get(f'{eps_key}_separation_ratio') is not None]

        nm_str = f"{np.median(null_maxs):.2e}" if null_maxs else "N/A"
        rm_str = f"{np.median(range_meds):.2e}" if range_meds else "N/A"
        sep_str = f"{np.median(seps):.2e}" if seps else "N/A"

        print(f"  {N_val:4d}  {3*N_val:4d}  {dim_eq:6.1f}  {pred:10.1f}  "
              f"{obs:9.1f}  {nm_str:>10s}  {rm_str:>10s}  {sep_str:>10s}")

    print(f"{'='*70}\n")


# =============================================================================
# MAIN
# =============================================================================

def process_one_case(results_dir: str, test_case: str, output_base: str):
    """Process results for one test case."""

    print(f"\nProcessing: {test_case}")
    print(f"  Results dir: {results_dir}")

    results = load_results(results_dir)
    if not results:
        print(f"  No results found in {results_dir}")
        return

    print(f"  Loaded {len(results)} seed results")

    # Extract pooled data
    records = extract_pooled_data(results)
    print(f"  Pooled {len(records)} (seed, N) data points")

    # Determine which epsilon keys are available
    eps_keys = [k for k in records[0].keys() if k.endswith('_null_fwd_max')]
    eps_keys = [k.replace('_null_fwd_max', '') for k in eps_keys]
    print(f"  Available epsilons: {eps_keys}")

    # Use smallest epsilon for primary analysis
    primary_eps = eps_keys[0] if eps_keys else 'eps_1e-07'

    # Summary statistics
    print_summary(records, test_case, primary_eps)

    # Plots
    plot_dir = os.path.join(output_base, f'plots_{test_case}')
    generate_plots(records, test_case, plot_dir, primary_eps)

    # Save pooled data as JSON
    summary_path = os.path.join(output_base, f'perturbation_summary_{test_case}.json')
    summary = {
        'test_case': test_case,
        'n_seeds': len(results),
        'n_records': len(records),
        'timestamp': datetime.now().isoformat(),
    }

    # Aggregate stats
    below = [r for r in records if r['N_minus_Nmax'] <= 0]
    above = [r for r in records if r['N_minus_Nmax'] > 0]

    if below:
        summary['below_Nmax'] = {
            'count': len(below),
            'all_null_dim_zero': bool(all(r['svd_null_dim'] == 0 for r in below)),
        }
    if above:
        seps = [r.get(f'{primary_eps}_separation_ratio') for r in above
                if r.get(f'{primary_eps}_separation_ratio') is not None]
        null_maxs = [r.get(f'{primary_eps}_null_fwd_max') for r in above
                     if r.get(f'{primary_eps}_null_fwd_max') is not None]
        summary['above_Nmax'] = {
            'count': len(above),
            'median_separation_ratio': float(np.median(seps)) if seps else None,
            'min_separation_ratio': float(np.min(seps)) if seps else None,
            'median_null_fwd_max': float(np.median(null_maxs)) if null_maxs else None,
        }

    svd_matches = sum(1 for r in records
                      if r['null_dim_predicted'] == r['svd_null_dim'])
    summary['svd_match_rate'] = float(svd_matches / len(records))

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate perturbation test (Test A) results"
    )
    parser.add_argument('--results-dir', type=str,
                        help='Results directory for one test case')
    parser.add_argument('--all-cases', action='store_true',
                        help='Process all available test cases')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for plots and summaries')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plot generation')

    args = parser.parse_args()

    if args.no_plots:
        global HAS_MATPLOTLIB
        HAS_MATPLOTLIB = False

    if args.results_dir:
        # Infer test case from directory name
        dirname = os.path.basename(args.results_dir.rstrip('/'))
        test_case = 'unknown'
        for tc in TEST_CASES:
            if tc in dirname:
                test_case = tc
                break
        process_one_case(args.results_dir, test_case, args.output_dir)

    elif args.all_cases:
        for tc in TEST_CASES:
            for suffix in ['_random_rho', '']:
                results_dir = f"perturbation_results_{tc}{suffix}"
                if os.path.exists(results_dir):
                    process_one_case(results_dir, tc, args.output_dir)
    else:
        print("Specify --results-dir or --all-cases")
        parser.print_help()


if __name__ == '__main__':
    main()
